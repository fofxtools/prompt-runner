"""Diagnostics for running prompt-runner as a client of the servers on the Ubuntu host.

Run it on the Windows VM (Python 3.10+, standard library only; it changes nothing and sends no secrets):

    python scripts\\windows\\diagnostics.py                      # host 192.168.56.1, ports 8190 (sd-server) and 8191 (llama)
    python scripts\\windows\\diagnostics.py --host 192.168.56.1 --sd-port 8190 --llm-port 8191
    python scripts\\windows\\diagnostics.py --repo C:\\path\\to\\prompt-runner    # also try importing prompt-runner's modules
    python scripts\\windows\\diagnostics.py --chat                # also send one tiny chat request (needs a model loaded on the host)

Lines starting with "!!" are the ones to look at.
"""

import argparse
import importlib
import json
import locale
import mimetypes
import os
import platform
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def say(text: str = "") -> None:
    print(text, flush=True)


def section(title: str) -> None:
    say()
    say(f"== {title}")


def problem(text: str) -> None:
    say(f"!! {text}")


def system() -> None:
    section("Python and system")
    say(
        f"python        {sys.version.split()[0]} ({platform.python_implementation()}) at {sys.executable}"
    )
    say(f"platform      {platform.platform()}")
    say(f"cwd           {os.getcwd()}")
    say(
        f"PYTHONUTF8    {os.environ.get('PYTHONUTF8')!r}   (utf8_mode={sys.flags.utf8_mode})"
    )
    say(
        f"locale        {locale.getpreferredencoding(False)} (default for open() without encoding)"
    )
    say(f"stdout        encoding={sys.stdout.encoding} isatty={sys.stdout.isatty()}")
    say(f"stdin         encoding={sys.stdin.encoding if sys.stdin else None}")
    if (
        not sys.flags.utf8_mode
        and locale.getpreferredencoding(False).lower() != "utf-8"
    ):
        problem(
            "not in UTF-8 mode: set PYTHONUTF8=1 in the .cmd (model replies and the YAML configs contain non-ASCII text)"
        )
    try:
        text = "em dash \u2014, accents \u00e9\u00e8, arrow \u2192, emoji \U0001f98a"
        text.encode(sys.stdout.encoding or "ascii")
        say("unicode print ok on this stdout")
    except UnicodeEncodeError as e:
        problem(f"this stdout cannot print common characters ({e}); needs PYTHONUTF8=1")


def modules() -> None:
    section("Modules prompt-runner's client path needs")
    for name in (
        "json",
        "urllib.request",
        "base64",
        "mimetypes",
        "secrets",
        "dataclasses",
        "yaml",
        "PIL",
    ):
        try:
            importlib.import_module(name)
            say(f"ok       {name}")
        except ImportError as e:
            note = (
                "needed (pip install pyyaml)"
                if name == "yaml"
                else "optional (only for contact sheets)" if name == "PIL" else "needed"
            )
            say(f"missing  {name:15} {note}  [{e}]")
    for name in ("fcntl", "readline", "termios"):
        try:
            importlib.import_module(name)
            say(f"present  {name}")
        except ImportError:
            say(
                f"absent   {name:15} (expected on Windows; prompt-runner's keep-alive code uses fcntl, the remote path must not import it)"
            )
    say(
        f"signal.SIGKILL {'present' if hasattr(__import__('signal'), 'SIGKILL') else 'absent (expected on Windows)'}"
    )
    say(
        f"os.killpg      {'present' if hasattr(os, 'killpg') else 'absent (expected on Windows)'}"
    )
    say(
        f"mimetypes      .png={mimetypes.guess_type('a.png')[0]} .jpg={mimetypes.guess_type('a.jpg')[0]} .webp={mimetypes.guess_type('a.webp')[0]}"
    )
    for tool in ("git",):
        try:
            out = subprocess.run(
                [tool, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            say(f"{tool:14} {out.stdout.strip() or out.stderr.strip()}")
        except (OSError, subprocess.SubprocessError) as e:
            say(f"{tool:14} not found ({e.__class__.__name__})")


def proxies() -> None:
    section(
        "Proxy settings (urllib uses these; a proxy in the way would break calls to the host)"
    )
    found = urllib.request.getproxies()
    say(f"getproxies    {found or 'none'}")
    for key in ("HTTP_PROXY", "http_proxy", "NO_PROXY", "no_proxy"):
        if key in os.environ:
            say(f"{key:13} {os.environ[key]}")
    if found:
        problem(
            "a proxy is configured: requests to the host may go through it; set NO_PROXY for the host address"
        )


def tcp(host: str, port: int) -> bool:
    start = time.monotonic()
    try:
        with socket.create_connection((host, port), timeout=3):
            say(f"tcp {host}:{port}  open ({(time.monotonic() - start) * 1000:.0f} ms)")
            return True
    except OSError as e:
        problem(f"tcp {host}:{port}  cannot connect ({e.__class__.__name__}: {e})")
        return False


def get(url: str) -> tuple[int | None, object]:
    start = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read()
            code = response.status
    except urllib.error.HTTPError as e:
        say(f"GET {url}  -> HTTP {e.code} ({(time.monotonic() - start) * 1000:.0f} ms)")
        return e.code, None
    except OSError as e:
        problem(f"GET {url}  failed ({e.__class__.__name__}: {e})")
        return None, None
    try:
        data: object = json.loads(body)
    except ValueError:
        data = body[:80]
    say(
        f"GET {url}  -> HTTP {code}, {len(body)} bytes ({(time.monotonic() - start) * 1000:.0f} ms)"
    )
    return code, data


def llama(host: str, port: int, chat: bool) -> None:
    section(f"llama serve at {host}:{port}")
    if not tcp(host, port):
        say(
            '(nothing listening: on the host, start it with LLM_HOST=<this address> and `llm MODEL "hi"` once the host change exists;'
        )
        say(
            " today the host servers bind to 127.0.0.1 only, so this is expected to fail)"
        )
        return
    base = f"http://{host}:{port}"
    get(base + "/health")
    _code, props = get(base + "/props")
    if isinstance(props, dict):
        say(f"model_path    {props.get('model_path')}")
        say(f"build_info   {props.get('build_info')}")
        say(f"total_slots  {props.get('total_slots')}")
        say(f"modalities   {props.get('modalities')}")
        settings = (props.get("default_generation_settings") or {}).get("params") or {}
        say(
            f"server sampling  temperature={settings.get('temperature')} top_k={settings.get('top_k')} top_p={settings.get('top_p')}"
        )
    if chat:
        body = json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Reply with the single word: ready"}
                ],
                "max_tokens": 16,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode()
        request = urllib.request.Request(
            base + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                reply = json.loads(response.read())
            text = reply["choices"][0]["message"]["content"]
            say(
                f"chat         ok in {time.monotonic() - start:.1f}s: {text!r}; timings={ {k: reply.get('timings', {}).get(k) for k in ('prompt_n', 'predicted_n', 'predicted_per_second')} }"
            )
        except (OSError, ValueError, KeyError) as e:
            problem(f"chat request failed ({e.__class__.__name__}: {e})")


def sd(host: str, port: int) -> None:
    section(f"sd-server at {host}:{port}")
    if not tcp(host, port):
        say("(nothing listening: same note as above)")
        return
    _code, caps = get(f"http://{host}:{port}/sdcpp/v1/capabilities")
    if isinstance(caps, dict):
        say(f"model        {caps.get('model')}")
        say(f"keys         {sorted(caps)[:12]}")


def repo_imports(repo: Path) -> None:
    section(f"prompt-runner at {repo}")
    src = repo / "src"
    if not src.is_dir():
        problem(f"{src} not found")
        return
    sys.path.insert(0, str(src))
    for name in (
        "utils",
        "config",
        "server_process",
        "sdcpp_server",
        "llama_server",
        "image_runner",
        "llm_runner",
        "keepalive",
        "genimg",
        "llm",
    ):
        try:
            importlib.import_module(f"prompt_runner.{name}")
            say(f"ok       prompt_runner.{name}")
        except Exception as e:  # noqa: BLE001 - this is a report
            say(f"FAILS    prompt_runner.{name}: {e.__class__.__name__}: {e}")
    cfg = repo / "config" / "config.yaml"
    try:
        text = (
            cfg.read_text()
        )  # no encoding on purpose: this is what prompt-runner's loader does
        say(f"config.yaml  read ok with the default encoding ({len(text)} characters)")
    except (OSError, UnicodeDecodeError) as e:
        problem(f"config.yaml cannot be read with the default encoding: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--host", default="192.168.56.1")
    ap.add_argument("--sd-port", type=int, default=8190)
    ap.add_argument("--llm-port", type=int, default=8191)
    ap.add_argument(
        "--repo", type=Path, help="path to the prompt-runner checkout on this machine"
    )
    ap.add_argument(
        "--chat", action="store_true", help="also send one tiny chat request"
    )
    args = ap.parse_args()
    system()
    modules()
    proxies()
    section(f"Host {args.host}")
    try:
        say(
            f"this machine's addresses: {sorted({i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None)})}"
        )
    except OSError:
        pass
    llama(args.host, args.llm_port, args.chat)
    sd(args.host, args.sd_port)
    if args.repo:
        repo_imports(args.repo)
    say()
    say("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
