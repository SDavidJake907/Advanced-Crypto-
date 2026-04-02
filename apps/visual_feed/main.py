from __future__ import annotations

import json
import os
import time
import ctypes
import ctypes.wintypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.runtime.log_rotation import prune_directory_files, rotate_jsonl_if_needed

LOG_PATH = Path("logs/visual_phi3_feed.jsonl")
SHOT_DIR = Path("logs/screens")
INTERVAL_SEC = float(os.getenv("VISUAL_PHI3_INTERVAL_SEC", "30"))
REVIEW_URL = os.getenv("VISUAL_PHI3_REVIEW_URL", "http://127.0.0.1:8085/review_image")
VISUAL_OLLAMA_MODEL = os.getenv("VISUAL_OLLAMA_MODEL", "").strip()
VISUAL_OLLAMA_URL = os.getenv("VISUAL_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
CAPTURE_TITLE = os.getenv("VISUAL_PHI3_WINDOW_TITLE", "Kraken")
CAPTURE_PROCESS_NAME = os.getenv("VISUAL_PHI3_PROCESS_NAME", "KrakenDesktop.exe").strip().lower()
CAPTURE_PROCESS_PID = int(os.getenv("VISUAL_PHI3_PROCESS_PID", "0") or "0")
FALLBACK_CAPTURE_MODE = os.getenv("VISUAL_PHI3_FALLBACK_CAPTURE", "foreground").strip().lower()
MIN_CAPTURE_WIDTH = int(os.getenv("VISUAL_PHI3_MIN_CAPTURE_WIDTH", "600"))
MIN_CAPTURE_HEIGHT = int(os.getenv("VISUAL_PHI3_MIN_CAPTURE_HEIGHT", "400"))
SCREENSHOT_KEEP_LATEST = int(os.getenv("VISUAL_SCREENSHOT_KEEP_LATEST", "80"))
SCREENSHOT_MAX_AGE_DAYS = float(os.getenv("VISUAL_SCREENSHOT_MAX_AGE_DAYS", "2"))
EXCLUDED_TITLE_PARTS = ("krakensk", "visual studio code", "visual_feed", "visual_phil", "review_scheduler", "trader")
PREFERRED_PROCESS_NAMES = {"krakendesktop.exe", "krakendesktop"}

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _log(payload: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rotate_jsonl_if_needed(LOG_PATH)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _window_area(window: Any) -> int:
    width = max(0, int(window.right) - int(window.left))
    height = max(0, int(window.bottom) - int(window.top))
    return width * height


def _window_meta(window: Any) -> dict[str, Any]:
    return {
        "title": window.title,
        "left": int(window.left),
        "top": int(window.top),
        "right": int(window.right),
        "bottom": int(window.bottom),
        "width": max(0, int(window.right) - int(window.left)),
        "height": max(0, int(window.bottom) - int(window.top)),
        "is_minimized": bool(window.isMinimized),
        "area": _window_area(window),
    }


def _query_process_info(hwnd: int) -> tuple[int, str]:
    pid = ctypes.c_ulong()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not pid.value:
        return 0, ""

    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not handle:
        return int(pid.value), ""

    try:
        size = ctypes.c_ulong(260)
        buffer = ctypes.create_unicode_buffer(size.value)
        ok = kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size))
        if not ok:
            return int(pid.value), ""
        return int(pid.value), os.path.basename(buffer.value).lower()
    finally:
        kernel32.CloseHandle(handle)


def _enum_process_windows() -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _callback(hwnd: int, lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        if user32.IsIconic(hwnd):
            return True

        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value.strip()

        rect = ctypes.wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True

        width = max(0, rect.right - rect.left)
        height = max(0, rect.bottom - rect.top)
        if width < MIN_CAPTURE_WIDTH or height < MIN_CAPTURE_HEIGHT:
            return True

        process_pid, process_name = _query_process_info(hwnd)
        windows.append(
            {
                "hwnd": int(hwnd),
                "title": title,
                "left": int(rect.left),
                "top": int(rect.top),
                "right": int(rect.right),
                "bottom": int(rect.bottom),
                "width": width,
                "height": height,
                "area": width * height,
                "process_pid": process_pid,
                "process_name": process_name,
            }
        )
        return True

    user32.EnumWindows(_callback, 0)
    return windows


def _foreground_window_meta() -> dict[str, Any] | None:
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    rect = ctypes.wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    width = max(0, rect.right - rect.left)
    height = max(0, rect.bottom - rect.top)
    if width < MIN_CAPTURE_WIDTH or height < MIN_CAPTURE_HEIGHT:
        return None
    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    process_pid, process_name = _query_process_info(hwnd)
    return {
        "hwnd": int(hwnd),
        "title": buffer.value.strip(),
        "left": int(rect.left),
        "top": int(rect.top),
        "right": int(rect.right),
        "bottom": int(rect.bottom),
        "width": width,
        "height": height,
        "area": width * height,
        "process_pid": process_pid,
        "process_name": process_name,
    }


def _desktop_meta() -> dict[str, Any]:
    width = int(user32.GetSystemMetrics(78))
    height = int(user32.GetSystemMetrics(79))
    left = int(user32.GetSystemMetrics(76))
    top = int(user32.GetSystemMetrics(77))
    return {
        "title": "desktop_fallback",
        "left": left,
        "top": top,
        "right": left + width,
        "bottom": top + height,
        "width": width,
        "height": height,
        "area": width * height,
        "process_pid": 0,
        "process_name": "desktop",
    }


def _select_window() -> tuple[Any | None, dict[str, Any]]:
    candidates = []
    near_matches = []
    for meta in _enum_process_windows():
        title_lower = (meta["title"] or "").lower()
        process_name = meta.get("process_name", "")
        if any(excluded in title_lower for excluded in EXCLUDED_TITLE_PARTS):
            continue

        process_pid = int(meta.get("process_pid", 0) or 0)
        pid_score = 1 if CAPTURE_PROCESS_PID > 0 and process_pid == CAPTURE_PROCESS_PID else 0
        process_score = 1 if process_name in PREFERRED_PROCESS_NAMES or process_name == CAPTURE_PROCESS_NAME else 0
        exact_title_score = 1 if title_lower in ("kraken", "kraken desktop", "kraken pro") else 0
        partial_title_score = 1 if CAPTURE_TITLE.lower() in title_lower else 0
        if pid_score == 0 and process_score == 0 and partial_title_score == 0:
            near_matches.append(meta)
            continue

        meta["pid_score"] = pid_score
        meta["process_score"] = process_score
        meta["exact_title_score"] = exact_title_score
        meta["partial_title_score"] = partial_title_score
        candidates.append(meta)

    if not candidates:
        fallback_meta = None
        fallback_reason = "window_not_found"
        if FALLBACK_CAPTURE_MODE == "foreground":
            fallback_meta = _foreground_window_meta()
            if fallback_meta is not None:
                fallback_reason = "foreground_fallback"
        elif FALLBACK_CAPTURE_MODE == "desktop":
            fallback_meta = _desktop_meta()
            fallback_reason = "desktop_fallback"

        if fallback_meta is not None:
            return fallback_meta, {
                "match_title": CAPTURE_TITLE,
                "match_process_name": CAPTURE_PROCESS_NAME,
                "match_process_pid": CAPTURE_PROCESS_PID,
                "candidate_count": 0,
                "fallback_mode": FALLBACK_CAPTURE_MODE,
                "fallback_reason": fallback_reason,
                "chosen": fallback_meta,
                "min_width": MIN_CAPTURE_WIDTH,
                "min_height": MIN_CAPTURE_HEIGHT,
                "near_matches": near_matches[:10],
            }
        return None, {
            "match_title": CAPTURE_TITLE,
            "match_process_name": CAPTURE_PROCESS_NAME,
            "match_process_pid": CAPTURE_PROCESS_PID,
            "candidate_count": 0,
            "fallback_mode": FALLBACK_CAPTURE_MODE,
            "min_width": MIN_CAPTURE_WIDTH,
            "min_height": MIN_CAPTURE_HEIGHT,
            "near_matches": near_matches[:10],
        }

    candidates.sort(
        key=lambda item: (
            item.get("pid_score", 0),
            item.get("process_score", 0),
            item.get("exact_title_score", 0),
            item.get("partial_title_score", 0),
            item["area"],
            item["width"],
            item["height"],
        ),
        reverse=True,
    )
    chosen_meta = candidates[0]
    return chosen_meta, {
        "match_title": CAPTURE_TITLE,
        "match_process_name": CAPTURE_PROCESS_NAME,
        "match_process_pid": CAPTURE_PROCESS_PID,
        "candidate_count": len(candidates),
        "chosen": chosen_meta,
        "candidates": candidates[:5],
    }


def _capture_window_image(out_path: Path) -> tuple[bool, str, dict[str, Any]]:
    try:
        from PIL import ImageGrab
    except Exception as exc:  # pragma: no cover
        return False, f"missing_capture_runtime:{exc}", {}

    window_meta, debug = _select_window()
    if window_meta is None:
        return False, "window_not_found", debug

    image = ImageGrab.grab(
        bbox=(
            int(window_meta["left"]),
            int(window_meta["top"]),
            int(window_meta["right"]),
            int(window_meta["bottom"]),
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    prune_directory_files(
        SHOT_DIR,
        pattern="*.png",
        keep_latest=SCREENSHOT_KEEP_LATEST,
        max_age_days=SCREENSHOT_MAX_AGE_DAYS,
    )
    return True, "captured", debug


_VISUAL_PROMPT = (
    "Review this Kraken chart or desktop screenshot as an advisory assistant only. "
    "Assume the preferred chart stack is 5m, 15m, and 1h. "
    "Treat 15m as the main decision chart, 5m as fast confirmation and acceleration, and 1h as higher-context structure. "
    "When visible, use MA 7, MA 26, MACD 12/26/9, RSI 14, Bollinger Bands, and volume bars to judge trend vs range, "
    "breakout vs fakeout, and overextension risk. "
    "Return strict JSON with keys: chart_state, setup_quality, visual_risk, range_warning, "
    "overextension_warning, support_resistance_note, reason. "
    "Do not give trade instructions. JSON only."
)


def _send_for_review_ollama(image_path: Path) -> tuple[bool, dict]:
    import base64
    try:
        import requests
    except Exception as exc:  # pragma: no cover
        return False, {"error": f"missing_requests:{exc}"}
    try:
        img_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        payload = {
            "model": VISUAL_OLLAMA_MODEL,
            "messages": [{"role": "user", "content": _VISUAL_PROMPT, "images": [img_b64]}],
            "stream": False,
        }
        resp = requests.post(f"{VISUAL_OLLAMA_URL}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "")
        result: dict[str, Any] = {"raw": raw, "device": "ollama", "model": VISUAL_OLLAMA_MODEL}
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                result.update(json.loads(raw[start:end + 1]))
            except Exception:
                pass
        return True, result
    except Exception as exc:
        return False, {"error": str(exc)}


def _send_for_review(image_path: Path) -> tuple[bool, dict]:
    if VISUAL_OLLAMA_MODEL:
        return _send_for_review_ollama(image_path)
    try:
        import requests
    except Exception as exc:  # pragma: no cover
        return False, {"error": f"missing_requests:{exc}"}
    try:
        response = requests.post(REVIEW_URL, json={"image_path": str(image_path)}, timeout=120)
        return response.ok, response.json()
    except Exception as exc:
        return False, {"error": str(exc)}


def main() -> None:
    while True:
        ts = datetime.now(timezone.utc).isoformat()
        shot_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".png"
        shot_path = SHOT_DIR / shot_name
        ok, status, capture_debug = _capture_window_image(shot_path)
        if not ok:
            _log(
                {
                    "ts": ts,
                    "event": "capture_failed",
                    "status": status,
                    "window_title": CAPTURE_TITLE,
                    "capture_debug": capture_debug,
                }
            )
            time.sleep(INTERVAL_SEC)
            continue

        review_ok, review_payload = _send_for_review(shot_path)
        _log(
            {
                "ts": ts,
                "event": "visual_review",
                "image_path": str(shot_path),
                "window_title": CAPTURE_TITLE,
                "capture_debug": capture_debug,
                "review_ok": review_ok,
                "review": review_payload,
            }
        )
        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
