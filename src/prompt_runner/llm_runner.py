"""LLM evaluation: run chats on managed llama.cpp servers."""

import json
import os
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llama_server import SAMPLING_FIELDS, LlamaServer
from .server_process import ServerError, engine_version
from .utils import (
    append_jsonl,
    generate_run_identifiers,
    merge_options,
    merged_server_args,
    random_seed,
    sanitize_fs_name,
)

# Ollama options that llama-server would silently ignore, and what to use instead.
OLLAMA_ONLY = {
    "num_predict": "max_tokens",
    "num_ctx": "the -c flag in server_args",
    "num_batch": "the -b flag in server_args",
    "num_gpu": "the -ngl flag in server_args",
    "num_thread": "the -t flag in server_args",
}
# server_args that give a model its vision projector (-hf finds one by itself).
MMPROJ_FLAGS = ("--mmproj", "-mm", "-hf")


@dataclass
class LlmJob:
    """
    One chat to run.

    Attributes:
        model: An llm_models.yaml entry (name, server_args, optional
            generation_options). Not modified.
        messages: OpenAI-style messages; a message can have `images`, a list of image file
            paths sent with it (the log records the paths, not the images)
        out: Where to write the answer (the final text, as markdown)
        options: Request fields for this chat (temperature, max_tokens, seed,
            chat_template_kwargs, ...); they win over the model's
        tags: Free-form values copied into the job's log rows (prompt id, ...).
            Must be JSON-serializable.
    """

    model: dict[str, Any]
    messages: list[dict[str, Any]]
    out: Path
    options: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.out = Path(self.out)


def prompt_messages(prompt: dict[str, Any]) -> list[dict[str, Any]]:
    """The messages of an llm_prompts.yaml entry: its `messages`, or its `prompt` (with its `images`) as one user message."""
    if "messages" in prompt:
        messages: list[dict[str, Any]] = prompt["messages"]
        return messages
    message: dict[str, Any] = {"role": "user", "content": prompt["prompt"]}
    if prompt.get("images"):
        message["images"] = prompt["images"]
    return [message]


def reply_metrics(timings: dict[str, Any]) -> dict[str, Any]:
    """
    Token counts and speeds from the server's `timings`.

    The input includes the tokens the server took from its prompt cache
    (`cached_tokens`), so a repeated prompt shows a tiny `input_seconds`.
    """

    def seconds(ms: Any) -> float | None:
        return round(ms / 1000, 3) if isinstance(ms, (int, float)) else None

    fresh, cached = timings.get("prompt_n"), timings.get("cache_n")
    output_tokens = timings.get("predicted_n")
    output_seconds = seconds(timings.get("predicted_ms"))
    return {
        "input_tokens": None if fresh is None else fresh + (cached or 0),
        "cached_tokens": cached,
        "output_tokens": output_tokens,
        "input_seconds": seconds(timings.get("prompt_ms")),
        "output_seconds": output_seconds,
        "output_tokens_per_second": (
            round(output_tokens / output_seconds, 3)
            if output_tokens and output_seconds
            else None
        ),
    }


def _check_options(options: dict[str, Any], model: str) -> None:
    """Refuse Ollama-only options: llama-server would ignore them without a word."""
    for name, instead in OLLAMA_ONLY.items():
        if name in options:
            raise ValueError(
                f"Option '{name}' (model '{model}') is an Ollama option that llama-server "
                f"ignores; use {instead}"
            )


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _Run:
    """State shared by the groups of one run_llm_jobs() call."""

    def __init__(
        self,
        run_id: str,
        binary: str,
        engine: str,
        out_dir: Path,
        generation_defaults: dict[str, Any] | None,
        write_log: bool,
        total: int,
        open_server: Callable[[str, dict[str, Any], Path], LlamaServer] | None = None,
    ) -> None:
        self.run_id = run_id
        self.binary = binary
        self.engine = engine
        self.out_dir = out_dir
        self.generation_defaults = generation_defaults
        self.log_path = out_dir / "runs.jsonl" if write_log else None
        self.total = total
        self.open_server = open_server
        self.done = 0
        self.rows: list[dict[str, Any]] = []
        self.server_log = ""  # of the group being run, relative to out_dir
        self.server_sampling: dict[str, Any] = (
            {}
        )  # what its server uses when a chat sets none

    def rel(self, path: Path) -> str:
        """Path relative to out_dir, or as-is when it lies outside it."""
        path = Path(os.path.abspath(path))
        base = Path(os.path.abspath(self.out_dir))
        return str(path.relative_to(base)) if path.is_relative_to(base) else str(path)

    def options_for(self, job: LlmJob) -> dict[str, Any]:
        """The job's request fields: defaults < model generation_options < job options."""
        return merge_options(
            self.generation_defaults,
            job.model.get("generation_options"),
            job.options,
        )

    def emit(self, job: LlmJob, status: str, **fields: Any) -> None:
        """Record one row in the results and in runs.jsonl."""
        row = {
            "ts": _now(),
            "run_id": self.run_id,
            "status": status,
            "model": job.model["name"],
            "output_file": None,
            "seed": None,
            "request": None,
            "output": None,
            "sampling": None,
            "finish_reason": None,
            "metrics": None,
            "server_args": None,
            "engine": self.engine,
            "server_start_s": None,
            "elapsed_s": None,
            "index_in_server": None,
            "tags": job.tags,
            "kind": None,
            "error": None,
            "exit_code": None,
            "server_log": None,
            **fields,
        }
        self.rows.append(row)
        if self.log_path:
            append_jsonl(self.log_path, row)

    def request_of(self, job: LlmJob, options: dict[str, Any]) -> dict[str, Any]:
        """What was asked: the messages and the request fields."""
        return {"messages": job.messages, **options}

    def failed(
        self,
        job: LlmJob,
        args: dict[str, Any],
        request: dict[str, Any],
        index: int,
        kind: str,
        error: str,
        exit_code: int | None = None,
    ) -> None:
        self.done += 1
        print(
            f"  [{self.done}/{self.total}] {self.rel(job.out)}: FAILED ({kind}): {error}"
        )
        self.emit(
            job,
            "failed",
            request=request,
            seed=request.get("seed"),
            server_args=args,
            index_in_server=index,
            kind=kind,
            error=error,
            exit_code=exit_code,
            server_log=self.server_log,
        )

    def skipped(self, job: LlmJob, args: dict[str, Any]) -> None:
        self.done += 1
        print(
            f"  [{self.done}/{self.total}] {self.rel(job.out)}: skipped (server failed)"
        )
        request = self.request_of(job, self.options_for(job))
        self.emit(
            job,
            "skipped",
            request=request,
            seed=request.get("seed"),
            server_args=args,
            error="not run: an earlier job on this server failed",
            server_log=self.server_log,
        )

    def run_job(
        self,
        server: LlamaServer,
        job: LlmJob,
        args: dict[str, Any],
        index: int,
        start_seconds: float | None,
    ) -> bool:
        """Run one job. Returns True if the server died."""
        options = self.options_for(job)
        if options.get("seed") in (None, -1):
            options["seed"] = random_seed()  # recorded, so the chat can be repeated
        request = self.request_of(job, options)

        start = time.time()
        try:
            reply = server.chat(job.messages, options)
        except ServerError as e:
            self.failed(job, args, request, index, e.kind, e.message, e.exit_code)
            return e.kind == "server_died"
        except OSError as e:  # an image file that cannot be read: nothing was sent
            self.failed(job, args, request, index, "input_error", f"input error: {e}")
            return False
        elapsed = round(time.time() - start, 3)

        job.out.parent.mkdir(parents=True, exist_ok=True)
        job.out.write_text(reply.text, encoding="utf-8")
        self.done += 1
        note = " (stopped at max_tokens)" if reply.finish_reason == "length" else ""
        print(f"  [{self.done}/{self.total}] {self.rel(job.out)}: ok {elapsed}s{note}")
        self.emit(
            job,
            "ok",
            output_file=self.rel(job.out),
            seed=options["seed"],
            request=request,
            output={"text": reply.text, "reasoning": reply.reasoning},
            sampling={
                name: options.get(name, self.server_sampling.get(name))
                for name in SAMPLING_FIELDS
            },
            finish_reason=reply.finish_reason,
            metrics=reply_metrics(reply.timings),
            server_args=args,
            server_start_s=start_seconds,
            elapsed_s=elapsed,
            index_in_server=index,
            server_log=self.server_log,
        )
        return False

    def run_group(self, number: int, args: dict[str, Any], group: list[LlmJob]) -> None:
        """Run jobs that share server args on one server; if it dies, skip the rest."""
        name = group[0].model["name"]
        log_file = (
            self.out_dir
            / "server_logs"
            / f"{sanitize_fs_name(self.run_id)}__{number}__{sanitize_fs_name(name)}.log"
        )
        self.server_log = self.rel(log_file)
        print(f"Starting server: {name} ({len(group)} job(s))")

        rest: list[LlmJob] = []
        # LlamaServer is looked up here, not at import, so tests can patch it.
        open_server = self.open_server or LlamaServer
        opened = time.time()
        try:
            with open_server(self.binary, args, log_file) as server:
                self.server_log = self.rel(server.log_path)
                start_seconds = round(time.time() - opened, 3)  # the model load
                self.server_sampling = server.sampling_defaults()
                for index, job in enumerate(group):
                    if self.run_job(
                        server, job, args, index, start_seconds if index == 0 else None
                    ):
                        rest = group[index + 1 :]
                        break
        except ServerError as e:
            # The server failed to start (or died on its way up).
            first = group[0]
            request = self.request_of(first, self.options_for(first))
            self.failed(first, args, request, 0, e.kind, e.message, e.exit_code)
            rest = group[1:]
        for job in rest:
            self.skipped(job, args)


def run_llm_jobs(
    jobs: list[LlmJob],
    *,
    binary: str,
    out_dir: Path,
    server_defaults: dict[str, Any] | None = None,
    generation_defaults: dict[str, Any] | None = None,
    write_log: bool = True,
    open_server: Callable[[str, dict[str, Any], Path], LlamaServer] | None = None,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Run chats on managed llama servers, one per distinct model setup.

    Jobs are grouped by their resolved server args (`server_defaults` merged
    under the model's own `server_args`), in order of first appearance, so each
    model loads once even if the jobs are interleaved. A failed job is
    recorded and the run continues; if a server dies, the rest of its group
    is recorded as skipped. There is no retry.

    Every attempt (ok, failed or skipped) is appended as one line to
    out_dir/runs.jsonl, flushed immediately. Server output goes to
    out_dir/server_logs/. A random seed is chosen (and recorded) for every
    chat that has none.

    Args:
        jobs: The chats to run
        binary: Path to the `llama` executable (`serve` is added)
        out_dir: Where runs.jsonl and server_logs/ go (job outputs can be anywhere)
        server_defaults: `llama serve` flags added to every model's server_args
        generation_defaults: Request fields under every model's generation_options
        write_log: Set False to skip runs.jsonl (server logs are still written)
        open_server: Called as open_server(binary, server_args, log_path) to get the
            server for a group, used as a context manager (default: start a new
            LlamaServer, stopped when the group is done)
        run_id: The id recorded in the rows (default: a new one)

    Returns:
        The log rows of this run

    Raises:
        ValueError: If binary is empty, two jobs have the same output path, a
            request option is an Ollama option that llama-server would ignore, an
            image file does not exist, or a job has images for a model without --mmproj
    """
    if not binary:
        raise ValueError("binary (path to the llama executable) is required")
    outs = Counter(os.path.abspath(job.out) for job in jobs)
    duplicates = sorted(path for path, count in outs.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"Duplicate output paths (jobs would overwrite each other): {duplicates}"
        )
    missing: list[str] = []
    for job in jobs:
        options = merge_options(
            generation_defaults, job.model.get("generation_options"), job.options
        )
        _check_options(options, job.model["name"])
        images = [p for message in job.messages for p in message.get("images") or []]
        missing += [p for p in images if not Path(p).is_file()]
        args = merged_server_args(server_defaults, job.model)
        if images and not any(args.get(flag) for flag in MMPROJ_FLAGS):
            raise ValueError(
                f"Job {job.out} has images but model '{job.model['name']}' has no --mmproj in its server_args"
            )
    if missing:
        raise ValueError(f"Image file(s) not found: {sorted(set(missing))}")
    if not jobs:
        return []

    groups: dict[str, tuple[dict[str, Any], list[LlmJob]]] = {}
    for job in jobs:
        args = merged_server_args(server_defaults, job.model)
        key = json.dumps(args, sort_keys=True, default=str)
        groups.setdefault(key, (args, []))[1].append(job)

    run = _Run(
        run_id or generate_run_identifiers()[0],
        binary,
        engine_version(binary),
        Path(out_dir),
        generation_defaults,
        write_log,
        len(jobs),
        open_server,
    )
    for number, (args, group) in enumerate(groups.values()):
        run.run_group(number, args, group)
    return run.rows


def run_llm_eval(
    config: dict[str, Any],
    prompts: list[dict[str, Any]],
    models: list[dict[str, Any]],
    options: dict[str, Any] | None = None,
) -> str:
    """
    Run every prompt on every model and save the results under config's results_dir.

    Results go to <results_dir>/<run folder>/llm/: runs.jsonl (one row per
    prompt and model), <prompt id>/<model>.md (the answer) and server_logs/.

    Args:
        config: Configuration with 'results_dir' and 'llama_server', and optionally
            'llm_server_args' and 'llm_generation_defaults'
        prompts: Prompt dictionaries with 'id' and 'prompt' or 'messages' (and 'options')
        models: Model dictionaries with 'name' and 'server_args' (and 'generation_options')
        options: Request fields for every chat, winning over the prompts' options and the
            models' generation_options (e.g. {"temperature": 0.7} to compare all models at one value)

    Returns:
        run_id: The ISO-8601 formatted run identifier

    Raises:
        ValueError: If config has no llama_server, or an option is an Ollama option
    """
    binary = config.get("llama_server")
    if not binary:
        raise ValueError("config has no 'llama_server' (path to the llama executable)")
    run_id, run_dir_name, _ = generate_run_identifiers()
    llm_dir = Path(config["results_dir"]) / run_dir_name / "llm"
    jobs = [
        LlmJob(
            model,
            prompt_messages(prompt),
            llm_dir / prompt["id"] / f"{sanitize_fs_name(model['name'])}.md",
            merge_options(prompt.get("options"), options),
            {"prompt_id": prompt["id"]},
        )
        for model in models
        for prompt in prompts
    ]
    run_llm_jobs(
        jobs,
        binary=binary,
        out_dir=llm_dir,
        server_defaults=config.get("llm_server_args"),
        generation_defaults=config.get("llm_generation_defaults"),
        run_id=run_id,
    )
    print(f"\n✓ Evaluation complete. Run ID: {run_id}")
    return run_id
