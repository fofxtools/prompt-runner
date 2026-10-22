"""stable-diffusion.cpp's `sd-server` and its native async API (/sdcpp/v1/...)."""

import base64
import time
import urllib.error
from typing import Any

from .server_process import (
    DEFAULT_HOST,
    POLL_INTERVAL,
    ServerError,
    ServerProcess,
    http_error_text,
)

REQUEST_TIMEOUT = 1800.0  # seconds per generation, including the lazy model load


class SdCppServer(ServerProcess):
    """
    One sd-server process serving one model.

    The server cannot switch models, so start one per model. Its model loads lazily,
    on the first request.
    """

    NAME = "sd-server"
    READY_PATH = "/sdcpp/v1/capabilities"

    @classmethod
    def listen_argv(
        cls, argv: list[str], port: int, host: str = DEFAULT_HOST
    ) -> list[str]:
        return [*argv, "-l", host, "--listen-port", str(port)]

    def loaded_file(self) -> str | None:
        """
        The model file the server says it serves, or None if it names none.

        It is the file given on the command line, reported before the lazy load, so asking costs the
        GPU nothing. Works for `--diffusion-model` and `--model` alike.

        Raises:
            OSError: If the server does not answer
        """
        model = self._http("GET", self.READY_PATH).get("model")
        path = model.get("path") if isinstance(model, dict) else None
        return path if isinstance(path, str) and path else None

    def generate(self, request: dict[str, Any]) -> list[bytes]:
        """
        Run one `img_gen` job and return the decoded images.

        Raises:
            ServerError: `job_failed` if the server rejected or failed the job
                (it stays usable), `server_died` if the process is gone or the
                request timed out (the server is killed).
        """
        try:
            job = self._http("POST", "/sdcpp/v1/img_gen", request)
        except urllib.error.HTTPError as e:
            raise ServerError("job_failed", http_error_text(e)) from e
        except OSError as e:
            self._raise_if_dead()
            raise ServerError("job_failed", f"submit failed: {e}") from e

        deadline = time.monotonic() + REQUEST_TIMEOUT
        while True:
            self._raise_if_dead()
            try:
                status = self._http("GET", f"/sdcpp/v1/jobs/{job['id']}")
            except OSError:
                status = {}
            state = status.get("status")
            if state == "completed":
                return [
                    base64.b64decode(img["b64_json"])
                    for img in status["result"]["images"]
                ]
            if state in ("failed", "cancelled"):
                error = status.get("error") or {}
                raise ServerError("job_failed", error.get("message") or f"job {state}")
            if time.monotonic() > deadline:
                stopped = " (server stopped)" if self.pid else ""
                self.stop()
                raise ServerError(
                    "server_died",
                    f"request timed out after {REQUEST_TIMEOUT:.0f}s{stopped}",
                )
            time.sleep(POLL_INTERVAL)
