"""A fake OpenAI-compatible chat server on 127.0.0.1 for tests (never the internet).

``FakeOpenAI`` answers ``POST /v1/chat/completions``. What it does is set per model name
in ``behaviour``: ``"ok"`` (default) answers with ``reply``, ``"500"`` answers HTTP 500,
``"slow"`` waits ``slow_s`` before answering. Every request body and its
``Authorization`` header are kept in ``requests`` so a test can check the fields sent.
Run it as a script to serve until stopped (used by the simulator scenarios):

    python fake_openai.py --port 8099 [--slow-model m] [--fail-model m]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

__all__ = ["FakeOpenAI"]


def _answer(model: str, content: str | None, calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls" if calls else "stop",
                "message": {"role": "assistant", "content": content, "tool_calls": calls},
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
    }


class FakeOpenAI:
    def __init__(self, port: int = 0) -> None:
        self.behaviour: dict[str, str] = {}
        self.slow_s = 30.0
        self.reply_content: str | None = "Batteriet är 80 procent."
        self.reply_calls: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.auth: list[str] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_POST(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                fake.requests.append(body)
                fake.auth.append(self.headers.get("authorization", ""))
                if self.path != "/v1/chat/completions":
                    self._send(404, {"error": "not found"})
                    return
                model = str(body.get("model", ""))
                how = fake.behaviour.get(model, "ok")
                if how == "500":
                    self._send(500, {"error": {"message": "provider down"}})
                    return
                if how == "slow":
                    time.sleep(fake.slow_s)
                self._send(200, _answer(model, fake.reply_content, fake.reply_calls))

            def _send(self, status: int, data: dict[str, Any]) -> None:
                raw = json.dumps(data).encode()
                try:
                    self.send_response(status)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    return  # the client gave up (timeout tests)

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def models_asked(self) -> list[str]:
        return [str(r.get("model")) for r in self.requests]

    def __enter__(self) -> FakeOpenAI:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--slow-model", default="")
    p.add_argument("--fail-model", default="")
    args = p.parse_args()
    fake = FakeOpenAI(args.port)
    if args.slow_model:
        fake.behaviour[args.slow_model] = "slow"
    if args.fail_model:
        fake.behaviour[args.fail_model] = "500"
    with fake:
        sys.stdout.write(f"fake OpenAI on {fake.base_url}\n")
        sys.stdout.flush()
        threading.Event().wait()


if __name__ == "__main__":
    main()
