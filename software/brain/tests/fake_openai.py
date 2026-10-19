"""A fake OpenAI-compatible chat server on 127.0.0.1 for tests (never the internet).

``FakeOpenAI`` answers ``POST /v1/chat/completions``. What it does is set per model name
in ``behaviour``: ``"ok"`` (default) answers with ``reply`` (and ``usage.cost`` when
``reply_cost`` is set), ``"500"`` answers HTTP 500, ``"401"`` refuses the key,
``"slow"`` waits ``slow_s`` before answering. Every request body and its
``Authorization`` header are kept in ``requests`` so a test can check the fields sent.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

__all__ = ["FakeOpenAI"]


def _answer(
    model: str, content: str | None, calls: list[dict[str, Any]], cost: float | None = None
) -> dict[str, Any]:
    usage: dict[str, Any] = {"prompt_tokens": 100, "completion_tokens": 10}
    if cost is not None:
        usage["cost"] = cost
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
        "usage": usage,
    }


class FakeOpenAI:
    def __init__(self, port: int = 0) -> None:
        self.behaviour: dict[str, str] = {}
        self.slow_s = 30.0
        self.reply_content: str | None = "Batteriet är 80 procent."
        self.reply_calls: list[dict[str, Any]] = []
        self.reply_cost: float | None = None  # usage.cost in US dollars, as OpenRouter states it
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
                if how == "401":
                    self._send(401, {"error": {"message": "User not found."}})
                    return
                if how == "slow":
                    time.sleep(fake.slow_s)
                answer = _answer(model, fake.reply_content, fake.reply_calls, fake.reply_cost)
                self._send(200, answer)

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
