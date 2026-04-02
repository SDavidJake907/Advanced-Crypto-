"""
Phi-3 compatibility server for KrakenSK.

Provides OpenAI-style chat/completion endpoints backed by an OpenVINO GenAI
pipeline that is loaded on the requested device. This lets the trader use the
same HTTP client path for Phi-3 while keeping device selection explicit.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import openvino_genai as ov_genai


MODEL_DIR = Path(
    os.getenv(
        "PHI3_MODEL_DIR",
        r"C:\Users\kitti\Projects\kraken-hybrad9\openvino_models\phi-3-mini-int4",
    )
)
PORT = int(os.getenv("PHI3_PORT", "8084"))
MAX_BODY = 2_000_000
ALLOWED_HOSTS = {"127.0.0.1", "localhost"}


def _extract_text_part(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        text = part.get("text")
        if isinstance(text, str):
            return text
    return ""


def _extract_message_content(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_extract_text_part(part) for part in content if _extract_text_part(part))
    return ""


def _build_chat_prompt(messages: list[dict[str, Any]]) -> str:
    segments: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip().lower()
        content = _extract_message_content(message).strip()
        if not content:
            continue
        if role == "system":
            tag = "system"
        elif role == "assistant":
            tag = "assistant"
        else:
            tag = "user"
        segments.append(f"<|{tag}|>\n{content}<|end|>")
    segments.append("<|assistant|>\n")
    return "\n".join(segments)


print(f"[PHI3] Loading Phi-3 from {MODEL_DIR} ...")
_load_started = time.time()
requested_device = os.environ.get("PHI3_DEVICE", "CPU").strip().upper()
strict_device = os.environ.get("PHI3_STRICT_DEVICE", "true").strip().lower() in {"1", "true", "yes", "on"}

if requested_device in {"NPU", "CPU", "GPU"}:
    device_order = [requested_device]
else:
    device_order = ["CPU"]

if not strict_device:
    for fallback in ("NPU", "GPU", "CPU"):
        if fallback not in device_order:
            device_order.append(fallback)

pipe = None
device_used = ""
load_error = ""
for device_name in device_order:
    try:
        pipe = ov_genai.LLMPipeline(str(MODEL_DIR), device_name)
        device_used = device_name
        print(f"[PHI3] Pipeline loaded on {device_name} in {time.time() - _load_started:.1f}s")
        break
    except Exception as exc:  # pragma: no cover - hardware dependent
        load_error = str(exc)
        print(f"[PHI3] {device_name} failed: {exc}")

if pipe is None:  # pragma: no cover - hardware dependent
    raise RuntimeError(f"[PHI3] Could not load pipeline. Last error: {load_error}")

print("[PHI3] Warming up...")
_warm_started = time.time()
_ = pipe.generate(
    _build_chat_prompt(
        [
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": "{\"ping\":true}"},
        ]
    ),
    max_new_tokens=24,
    do_sample=False,
)
print(f"[PHI3] Warmup done in {time.time() - _warm_started:.1f}s on {device_used}")

server_metrics: dict[str, Any] = {
    "device": device_used,
    "requested_device": requested_device,
    "strict_device": strict_device,
    "model_dir": str(MODEL_DIR),
    "request_count": 0,
    "last_latency_ms": 0,
    "startup_chat_ok": True,
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _check_host(self) -> bool:
        host = self.headers.get("Host", "").split(":")[0].strip()
        return host in ALLOWED_HOSTS

    def _read_json(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > MAX_BODY:
                return None
            raw = self.rfile.read(length)
            return json.loads(raw)
        except Exception:
            return None

    def _send_json(self, code: int, obj: dict[str, Any]) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if not self._check_host():
            self._send_json(403, {"error": "forbidden"})
            return
        if self.path == "/health":
            self._send_json(200, {"status": "ok", **server_metrics})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._check_host():
            self._send_json(403, {"error": "forbidden"})
            return

        payload = self._read_json()
        if payload is None:
            self._send_json(400, {"error": "invalid json"})
            return

        if self.path == "/v1/chat/completions":
            messages = payload.get("messages")
            if not isinstance(messages, list):
                self._send_json(400, {"error": "messages must be a list"})
                return
            prompt = _build_chat_prompt(messages)
            max_tokens = int(payload.get("max_tokens", 400) or 400)
            try:
                started = time.perf_counter()
                raw_text = str(
                    pipe.generate(prompt, max_new_tokens=max_tokens, do_sample=False)
                ).strip()
                latency_ms = int((time.perf_counter() - started) * 1000)
            except Exception as exc:  # pragma: no cover - hardware dependent
                self._send_json(500, {"error": str(exc)})
                return
            server_metrics["request_count"] += 1
            server_metrics["last_latency_ms"] = latency_ms
            self._send_json(
                200,
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": payload.get("model", "phi3"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": raw_text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                },
            )
            return

        if self.path == "/v1/completions":
            prompt = str(payload.get("prompt", "") or "")
            if not prompt:
                self._send_json(400, {"error": "prompt required"})
                return
            max_tokens = int(payload.get("max_tokens", 400) or 400)
            try:
                started = time.perf_counter()
                raw_text = str(
                    pipe.generate(prompt, max_new_tokens=max_tokens, do_sample=False)
                ).strip()
                latency_ms = int((time.perf_counter() - started) * 1000)
            except Exception as exc:  # pragma: no cover - hardware dependent
                self._send_json(500, {"error": str(exc)})
                return
            server_metrics["request_count"] += 1
            server_metrics["last_latency_ms"] = latency_ms
            self._send_json(
                200,
                {
                    "id": f"cmpl-{uuid.uuid4().hex}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": payload.get("model", "phi3"),
                    "choices": [{"index": 0, "text": raw_text, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                },
            )
            return

        self._send_json(404, {"error": "not found"})


print(f"[PHI3] Listening on http://127.0.0.1:{PORT}")
HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
