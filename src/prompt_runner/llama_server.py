"""llama.cpp's `llama serve` and its OpenAI-compatible chat API (/v1/chat/completions)."""

import base64
import http.client
import json
import mimetypes
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn

from .server_process import DEFAULT_HOST, ServerError, ServerProcess, http_error_text

# Seconds a request may go without a single byte before the server counts as hung. The
# server sends a comment line every 30 s while it works (--sse-ping-interval), so a
# healthy server never stays silent this long, even on a very long prompt.
IDLE_TIMEOUT = 1800.0

DeltaKind = Literal["reasoning", "text"]
# The request fields that make up the sampling, under the names of the server's `/props`.
SAMPLING_FIELDS = (
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "repeat_penalty",
    "presence_penalty",
)
# Which field of a streamed delta holds which kind of piece.
DELTA_FIELDS: tuple[tuple[DeltaKind, str], ...] = (
    ("reasoning", "reasoning_content"),
    ("text", "content"),
)


def image_part(path: str | Path) -> dict[str, Any]:
    """
    An `image_url` content part holding the image file as a data URI.

    The server wants a MIME type that starts with `image/` and does not care which
    (a JPEG labelled image/png works), so an unknown suffix is labelled image/png.

    Raises:
        OSError: If the file cannot be read
    """
    mime = mimetypes.guess_type(str(path))[0] or ""
    if not mime.startswith("image/"):
        mime = "image/png"
    encoded = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def with_image_parts(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    The messages as the server wants them: a message's `images` (file paths) become image parts.

    The images go before the text, as the Gemma and Qwen cards advise; `content` can be
    a string or a list of parts. A message without `images` is left as it is.

    Raises:
        OSError: If an image cannot be read
    """
    result = []
    for message in messages:
        paths = message.get("images")
        if not paths:
            result.append(message)
            continue
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}] if content else []
        parts = [image_part(path) for path in paths] + list(content or [])
        result.append(
            {**{k: v for k, v in message.items() if k != "images"}, "content": parts}
        )
    return result


@dataclass(frozen=True)
class ChatReply:
    """
    A finished chat reply.

    Attributes:
        text: The answer (`content`)
        reasoning: The thinking (`reasoning_content`); empty if the model did not think
        finish_reason: "stop" or "length", None if the server sent none. A reply that ran
            into the context size also says "stop".
        timings: The server's `timings` block (prompt_n, cache_n, predicted_n,
            prompt_ms, predicted_ms, predicted_per_second, ...); empty if it sent none
    """

    text: str
    reasoning: str
    finish_reason: str | None
    timings: dict[str, Any]


class LlamaServer(ServerProcess):
    """
    One `llama serve` process serving one model.

    The model (and its --mmproj) load before the server answers /health, which can take
    a minute or more for a big model. `binary` is the `llama` executable; the `serve`
    subcommand is added when the command line is built.
    """

    NAME = "llama-server"
    READY_PATH = "/health"  # 503 while the model loads, 200 once it is ready
    STARTUP_TIMEOUT = 600.0

    @classmethod
    def listen_argv(
        cls, argv: list[str], port: int, host: str = DEFAULT_HOST
    ) -> list[str]:
        return [argv[0], "serve", *argv[1:], "--host", host, "--port", str(port)]

    def loaded_file(self) -> str | None:
        """
        The model file the server says it serves, or None if it names none.

        Raises:
            OSError: If the server does not answer
        """
        path = self._http("GET", "/props").get("model_path")
        return path if isinstance(path, str) and path else None

    def build_info(self) -> str | None:
        """The server's build, e.g. 'b11630-6868a52cb', or None. Raises OSError if it does not answer."""
        info = self._http("GET", "/props").get("build_info")
        return info if isinstance(info, str) and info else None

    def sampling_defaults(self) -> dict[str, Any]:
        """
        The sampling the server uses for a request that sets none of it.

        These are llama.cpp's own defaults, or the ones the model's GGUF carries from its
        authors. Empty if the server does not say.
        """
        try:
            settings = self._http("GET", "/props")["default_generation_settings"]
            params = settings["params"]
        except (OSError, ValueError, KeyError, TypeError):
            return {}
        # The server holds 32-bit floats: 0.8 comes back as 0.800000011920929.
        return {
            name: (
                round(params[name], 4)
                if isinstance(params[name], float)
                else params[name]
            )
            for name in SAMPLING_FIELDS
            if name in params
        }

    def chat(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
        on_delta: Callable[[DeltaKind, str], None] | None = None,
    ) -> ChatReply:
        """
        Run one chat completion and return the whole reply.

        The reply is always streamed from the server, so `on_delta(kind, piece)` can show it as it
        arrives; it is called for each piece of reasoning and of the answer.

        Args:
            messages: OpenAI-style messages. A message can also have `images`, a list of
                file paths: they are sent as images before its text (the server must have
                been started with --mmproj). A content part {"type": "image_url", ...}
                works as well.
            options: More request fields, sent as they are: max_tokens (-1 for no limit), temperature,
                top_p, top_k, seed, chat_template_kwargs ({"enable_thinking": false}), ... The
                server silently ignores fields it does not know (Ollama's `num_predict`, for one).
            on_delta: Called as pieces arrive

        Raises:
            ServerError: `job_failed` if the server rejected or failed the request (it stays
                usable); `server_died` if the process is gone, the connection broke and the
                server no longer answers, or nothing arrived for IDLE_TIMEOUT (the server is
                stopped).
            OSError: If an image file cannot be read (nothing is sent)
        """
        body = {
            **(options or {}),
            "messages": with_image_parts(messages),
            "stream": True,
        }
        request = urllib.request.Request(
            self._base_url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        pieces: dict[DeltaKind, list[str]] = {"reasoning": [], "text": []}
        finish_reason: str | None = None
        timings: dict[str, Any] = {}
        done = False
        try:
            with urllib.request.urlopen(request, timeout=IDLE_TIMEOUT) as response:
                for raw in response:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue  # blank separators and the ":" comment pings
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        error = chunk["error"]
                        message = (
                            error.get("message") if isinstance(error, dict) else None
                        )
                        raise ServerError("job_failed", message or str(error))
                    timings = chunk.get("timings") or timings
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        for kind, key in DELTA_FIELDS:
                            if delta.get(key):
                                pieces[kind].append(delta[key])
                                if on_delta is not None:
                                    on_delta(kind, delta[key])
                        finish_reason = choice.get("finish_reason") or finish_reason
        except urllib.error.HTTPError as e:
            raise ServerError("job_failed", http_error_text(e)) from e
        except TimeoutError as e:
            stopped = " (server stopped)" if self.pid else ""
            self.stop()
            raise ServerError(
                "server_died", f"no output for {IDLE_TIMEOUT:.0f}s{stopped}"
            ) from e
        except ValueError as e:
            raise ServerError(
                "job_failed", f"unreadable reply from the server: {e}"
            ) from e
        except (OSError, http.client.HTTPException) as e:
            self._lost_connection(str(e))
        if not done and finish_reason is None:
            self._lost_connection("the reply ended early")
        return ChatReply(
            "".join(pieces["text"]),
            "".join(pieces["reasoning"]),
            finish_reason,
            timings,
        )

    def _lost_connection(self, why: str) -> NoReturn:
        """Raise `server_died` if the server is gone, else `job_failed`."""
        self._raise_if_dead()
        if not self.ready():
            raise ServerError(
                "server_died",
                f"lost the connection and {self.NAME} does not answer ({why})",
            )
        raise ServerError("job_failed", f"lost the connection: {why}")
