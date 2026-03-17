from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

DEFAULT_PROMPT = (
    "Review this Kraken chart or desktop screenshot as an advisory assistant only. "
    "Assume the preferred chart stack is 5m, 15m, and 1h. "
    "Treat 15m as the main decision chart, 5m as fast confirmation and acceleration, and 1h as higher-context structure. "
    "When visible, use MA 7, MA 26, MACD 12/26/9, RSI 14, Bollinger Bands, and volume bars to judge trend vs range, "
    "breakout vs fakeout, and overextension risk. "
    "Return strict JSON with: chart_state, setup_quality, visual_risk, range_warning, "
    "overextension_warning, support_resistance_note, reason. "
    "Do not give trade instructions."
)

MODEL_DIR = Path(os.getenv("VISUAL_PHI3_MODEL_DIR", ""))
PORT = int(os.getenv("VISUAL_PHI3_PORT", "8085"))
DEVICE_PREF = os.getenv("VISUAL_PHI3_DEVICE", "NPU").strip().upper()
MAX_BODY = 65_536

pipe = None
device_used = "uninitialized"
load_error = ""


def _load_pipeline() -> None:
    global pipe, device_used, load_error
    try:
        import openvino as ov  # noqa: F401
        import openvino_genai as ov_genai
        from PIL import Image  # noqa: F401
    except Exception as exc:  # pragma: no cover
        load_error = f"missing_runtime:{exc}"
        return

    if not MODEL_DIR or not MODEL_DIR.exists():
        load_error = f"missing_model_dir:{MODEL_DIR}"
        return

    device_order = [DEVICE_PREF] if DEVICE_PREF in {"NPU", "CPU"} else ["CPU"]
    if "CPU" not in device_order:
        device_order.append("CPU")

    for dev in device_order:
        try:
            pipe = ov_genai.VLMPipeline(str(MODEL_DIR), dev)
            device_used = dev
            load_error = ""
            return
        except Exception as exc:  # pragma: no cover
            load_error = f"{dev}_failed:{exc}"


def _build_response(image_path: str, prompt: str) -> dict[str, Any]:
    import numpy as np
    import openvino as ov
    from PIL import Image

    if pipe is None:
        raise RuntimeError(load_error or "visual_phi3_not_ready")

    image = Image.open(image_path).convert("RGB")
    image_data = np.array(image, dtype=np.uint8)
    image_tensor = ov.Tensor(image_data.reshape(1, image_data.shape[0], image_data.shape[1], 3))

    started = time.perf_counter()
    result = pipe.generate(prompt, image=image_tensor)
    latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
    text = str(result.texts[0] if hasattr(result, "texts") else result).strip()

    payload = {
        "raw": text,
        "latency_ms": latency_ms,
        "device": device_used,
    }
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            payload.update(json.loads(text[start : end + 1]))
    except Exception:
        payload.update(
            {
                "chart_state": "unknown",
                "setup_quality": "unparsed",
                "visual_risk": "medium",
                "range_warning": False,
                "overextension_warning": False,
                "support_resistance_note": "",
                "reason": "unparsed_visual_response",
            }
        )
    return payload


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            if pipe is not None:
                status, code = "ok", 200
            elif load_error:
                status, code = "error", 503
            else:
                status, code = "loading", 200
            self._send_json(code, {"status": status, "device": device_used, "error": load_error})
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/review_image":
            self._send_json(404, {"error": "not_found"})
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0 or length > MAX_BODY:
            self._send_json(400, {"error": "bad_request"})
            return
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw)
        except Exception:
            self._send_json(400, {"error": "invalid_json"})
            return
        image_path = str(data.get("image_path", "")).strip()
        prompt = str(data.get("prompt", DEFAULT_PROMPT)).strip() or DEFAULT_PROMPT
        if not image_path:
            self._send_json(400, {"error": "missing_image_path"})
            return
        if not Path(image_path).exists():
            self._send_json(404, {"error": "image_not_found"})
            return
        try:
            payload = _build_response(image_path, prompt)
            self._send_json(200, payload)
        except Exception as exc:
            self._send_json(500, {"error": str(exc), "device": device_used})


def main() -> None:
    import threading
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(json.dumps({"event": "visual_phi3_start", "port": PORT, "device": device_used, "model_dir": str(MODEL_DIR), "error": load_error}), flush=True)
    loader = threading.Thread(target=_load_pipeline, daemon=True)
    loader.start()
    server.serve_forever()


if __name__ == "__main__":
    main()
