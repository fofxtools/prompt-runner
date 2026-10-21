"""Image generation: run a list of jobs on managed sd-server processes."""

import base64
import json
import os
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .sdcpp_server import SdCppServer
from .server_process import ServerError, engine_version
from .utils import (
    append_jsonl,
    generate_run_identifiers,
    merge_options,
    merged_server_args,
    random_seed,
    sanitize_fs_name,
)

# Request fields that hold one image; they are file paths in a job and base64 on the wire.
IMAGE_FIELDS = ("init_image", "mask_image", "control_image", "ip_adapter_image")


@dataclass
class Job:
    """
    One image to generate.

    Attributes:
        model: An image_models.yaml entry (name, server_args, optional
            generation_options). Not modified.
        options: Native sd-server request fields for this image: prompt, seed,
            lora, init_image (a file path), ref_images (file paths), etc.
        out: Where to write the PNG. The extra images of a batch are written
            next to it as <stem>_1.png, <stem>_2.png, ...
        tags: Free-form values copied into the job's log rows (prompt id,
            variant, ...). Must be JSON-serializable.
    """

    model: dict[str, Any]
    options: dict[str, Any]
    out: Path
    tags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.out = Path(self.out)


def prepare_request(options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Resolve the seed and inline the input images of a native request.

    `init_image`, `mask_image`, `control_image`, `ip_adapter_image` and
    `ref_images` are file paths in a job; the server wants base64.

    Args:
        options: The merged request fields (not modified)

    Returns:
        (request, recorded): the request to send, and the same request with the
        images left as the given paths (what gets logged).

    Raises:
        OSError: If an input file cannot be read
    """
    recorded = dict(options)
    if recorded.get("seed") in (None, -1):
        recorded["seed"] = random_seed()
    request = dict(recorded)
    for key in IMAGE_FIELDS:
        if recorded.get(key):
            request[key] = _b64(recorded[key])
    if recorded.get("ref_images"):
        request["ref_images"] = [_b64(path) for path in recorded["ref_images"]]
    return request, recorded


def _b64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def _batch_path(out: Path, index: int) -> Path:
    """First image at `out`; later images of a batch as <stem>_<index><suffix>."""
    return out if index == 0 else out.with_name(f"{out.stem}_{index}{out.suffix}")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _Run:
    """State shared by the groups of one run_jobs() call."""

    def __init__(
        self,
        run_id: str,
        run_dir_name: str,
        binary: str,
        engine: str,
        out_dir: Path,
        generation_defaults: dict[str, Any] | None,
        write_log: bool,
        total: int,
        open_server: Callable[[str, dict[str, Any], Path], SdCppServer] | None = None,
    ) -> None:
        self.run_id = run_id
        self.run_dir_name = run_dir_name
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

    def rel(self, path: Path) -> str:
        """Path relative to out_dir, or as-is when it lies outside it."""
        path = Path(os.path.abspath(path))
        base = Path(os.path.abspath(self.out_dir))
        return str(path.relative_to(base)) if path.is_relative_to(base) else str(path)

    def options_for(self, job: Job) -> dict[str, Any]:
        """The job's request fields: defaults < model generation_options < job options."""
        return merge_options(
            self.generation_defaults,
            job.model.get("generation_options"),
            job.options,
        )

    def emit(self, job: Job, status: str, **fields: Any) -> None:
        """Record one row in the results and in runs.jsonl."""
        row = {
            "ts": _now(),
            "run_id": self.run_id,
            "status": status,
            "model": job.model["name"],
            "image": None,
            "batch_index": None,
            "batch_count": None,
            "seed": None,
            "request": None,
            "server_args": None,
            "engine": self.engine,
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

    def failed(
        self,
        job: Job,
        args: dict[str, Any],
        request: dict[str, Any],
        index: int,
        kind: str,
        error: str,
        exit_code: int | None = None,
    ) -> None:
        self.done += 1
        print(f"  [{self.done}/{self.total}] {job.out.name}: FAILED ({kind}): {error}")
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

    def skipped(self, job: Job, args: dict[str, Any]) -> None:
        self.done += 1
        print(f"  [{self.done}/{self.total}] {job.out.name}: skipped (server failed)")
        request = self.options_for(job)
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
        self, server: SdCppServer, job: Job, args: dict[str, Any], index: int
    ) -> bool:
        """Run one job. Returns True if the server died."""
        options = self.options_for(job)
        try:
            request, recorded = prepare_request(options)
        except OSError as e:
            self.failed(job, args, options, index, "input_error", f"input error: {e}")
            return False

        start = time.time()
        try:
            images = server.generate(request)
        except ServerError as e:
            self.failed(job, args, recorded, index, e.kind, e.message, e.exit_code)
            return e.kind == "server_died"
        elapsed = round(time.time() - start, 3)

        self.done += 1
        print(f"  [{self.done}/{self.total}] {job.out.name}: ok {elapsed}s")
        for i, png in enumerate(images):
            path = _batch_path(job.out, i)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(png)
            self.emit(
                job,
                "ok",
                image=self.rel(path),
                batch_index=i,
                batch_count=len(images),
                seed=recorded["seed"] + i,  # a batch uses seed, seed+1, ...
                request=recorded,
                server_args=args,
                elapsed_s=elapsed,
                index_in_server=index,
            )
        return False

    def run_group(self, number: int, args: dict[str, Any], group: list[Job]) -> None:
        """Run jobs that share server args on one server; if it dies, skip the rest."""
        name = group[0].model["name"]
        log_file = (
            self.out_dir
            / "server_logs"
            / f"{self.run_dir_name}__{number}__{sanitize_fs_name(name)}.log"
        )
        self.server_log = self.rel(log_file)
        print(f"Starting server: {name} ({len(group)} job(s))")

        rest: list[Job] = []
        # SdCppServer is looked up here, not at import, so tests can patch it.
        open_server = self.open_server or SdCppServer
        try:
            with open_server(self.binary, args, log_file) as server:
                self.server_log = self.rel(server.log_path)
                for index, job in enumerate(group):
                    if self.run_job(server, job, args, index):
                        rest = group[index + 1 :]
                        break
        except ServerError as e:
            # The server failed to start (or died on its way up).
            first = group[0]
            options = self.options_for(first)
            self.failed(first, args, options, 0, e.kind, e.message, e.exit_code)
            rest = group[1:]
        for job in rest:
            self.skipped(job, args)


def run_jobs(
    jobs: list[Job],
    *,
    binary: str,
    out_dir: Path,
    server_defaults: dict[str, Any] | None = None,
    generation_defaults: dict[str, Any] | None = None,
    skip_existing: bool = False,
    write_log: bool = True,
    open_server: Callable[[str, dict[str, Any], Path], SdCppServer] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate the images of a list of jobs, one managed sd-server per distinct model setup.

    Jobs are grouped by their resolved server args (`server_defaults` merged
    under the model's own `server_args`), in order of first appearance, so each
    model loads once even if the jobs are interleaved. A failed job is
    recorded and the run continues; if a server dies, the rest of its group
    is recorded as skipped. There is no retry.

    Every attempt (ok, failed or skipped) is appended as one line per image to
    out_dir/runs.jsonl, flushed immediately. Server output goes to
    out_dir/server_logs/.

    Args:
        jobs: The images to generate
        binary: Path to the sd-server binary
        out_dir: Where runs.jsonl and server_logs/ go (job outputs can be anywhere)
        server_defaults: sd-server flags added to every model's server_args
        generation_defaults: Request fields under every model's generation_options
        skip_existing: Skip jobs whose output file already exists
        write_log: Set False to skip runs.jsonl (server logs are still written)
        open_server: Called as open_server(binary, server_args, log_path) to get the
            server for a group, used as a context manager (default: start a new
            SdCppServer, stopped when the group is done). A kept or attached server
            can be returned here; its own `log_path` is what the log rows record.

    Returns:
        The log rows of this run (jobs skipped for an existing file have none)

    Raises:
        ValueError: If binary is empty or two jobs have the same output path
    """
    if not binary:
        raise ValueError("binary (path to the sd-server binary) is required")
    outs = Counter(os.path.abspath(job.out) for job in jobs)
    duplicates = sorted(path for path, count in outs.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"Duplicate output paths (jobs would overwrite each other): {duplicates}"
        )

    todo = [job for job in jobs if not (skip_existing and job.out.exists())]
    if len(todo) < len(jobs):
        print(f"Skipping {len(jobs) - len(todo)} existing image(s)")
    if not todo:
        return []

    groups: dict[str, tuple[dict[str, Any], list[Job]]] = {}
    for job in todo:
        args = merged_server_args(server_defaults, job.model)
        key = json.dumps(args, sort_keys=True, default=str)
        groups.setdefault(key, (args, []))[1].append(job)

    run_id, run_dir_name, _ = generate_run_identifiers()
    run = _Run(
        run_id,
        run_dir_name,
        binary,
        engine_version(binary),
        Path(out_dir),
        generation_defaults,
        write_log,
        len(todo),
        open_server,
    )
    for number, (args, group) in enumerate(groups.values()):
        run.run_group(number, args, group)
    return run.rows
