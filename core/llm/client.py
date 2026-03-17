from __future__ import annotations

from contextlib import nullcontext
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import msvcrt

from core.runtime.log_rotation import rotate_jsonl_if_needed

NEMOTRON_DEBUG_LOG = Path("logs/nemotron_debug.jsonl")
NEMOTRON_DEBUG_VERBOSE = os.getenv("NEMOTRON_DEBUG_VERBOSE", "false").lower() == "true"
_LOCAL_NEMOTRON_LOCK = Path("logs/nemotron_local.lock")


def _append_nemotron_debug(event: dict[str, Any]) -> None:
    NEMOTRON_DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    rotate_jsonl_if_needed(NEMOTRON_DEBUG_LOG)
    compact_event = dict(event)
    if not NEMOTRON_DEBUG_VERBOSE:
        compact_event.pop("payload", None)
    payload = sanitize_for_json({"ts": datetime.now(timezone.utc).isoformat(), **compact_event})
    with NEMOTRON_DEBUG_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return sanitize_for_json(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return sanitize_for_json(value.item())
        except Exception:
            pass
    return value


def _chat_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _completion_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/completions"
    return f"{base}/v1/completions"


def _ollama_generate_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return f"{base}/api/generate"


def _completion_prompt(*, system: str, user_payload: dict[str, Any]) -> str:
    return (
        f"{system.strip()}\n\n"
        "Input JSON:\n"
        f"{json.dumps(sanitize_for_json(user_payload), separators=(',', ':'))}\n\n"
        "Return only the final JSON object. No markdown. No prose. No <think> tags."
    )


def _chat_completion(
    *,
    base_url: str,
    model: str,
    system: str,
    user_payload: dict[str, Any],
    temperature: float = 0.1,
    max_tokens: int = 256,
    headers: dict[str, str] | None = None,
    timeout_s: float = 30.0,
    extra_body: dict[str, Any] | None = None,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(sanitize_for_json(user_payload))},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        payload.update(extra_body)
    response = httpx.post(
        _chat_endpoint(base_url),
        json=payload,
        headers=headers,
        timeout=timeout_s,
    )
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0]["message"]["content"])


def _completion_completion(
    *,
    base_url: str,
    model: str,
    system: str,
    user_payload: dict[str, Any],
    temperature: float = 0.0,
    max_tokens: int = 256,
    headers: dict[str, str] | None = None,
    timeout_s: float = 30.0,
    extra_body: dict[str, Any] | None = None,
) -> str:
    payload = {
        "model": model,
        "prompt": _completion_prompt(system=system, user_payload=user_payload),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        payload.update(extra_body)
    response = httpx.post(
        _completion_endpoint(base_url),
        json=payload,
        headers=headers,
        timeout=timeout_s,
    )
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0].get("text", ""))


def _ollama_generate_completion(
    *,
    base_url: str,
    model: str,
    system: str,
    user_payload: dict[str, Any],
    temperature: float = 0.0,
    max_tokens: int = 256,
    headers: dict[str, str] | None = None,
    timeout_s: float = 30.0,
    extra_body: dict[str, Any] | None = None,
) -> str:
    payload = {
        "model": model,
        "system": system,
        "prompt": json.dumps(sanitize_for_json(user_payload)),
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if extra_body:
        options = payload.get("options", {})
        extra_options = extra_body.get("options", {}) if isinstance(extra_body, dict) else {}
        payload.update({k: v for k, v in extra_body.items() if k != "options"})
        payload["options"] = {**options, **extra_options}
    response = httpx.post(
        _ollama_generate_endpoint(base_url),
        json=payload,
        headers=headers,
        timeout=timeout_s,
    )
    response.raise_for_status()
    data = response.json()
    return str(data.get("response", ""))


def phi3_chat(user_payload: dict[str, Any], *, system: str, max_tokens: int = 200) -> str:
    base_url = os.getenv("PHI3_BASE_URL", "http://127.0.0.1:8084")
    model = os.getenv("PHI3_MODEL", "phi3")
    return _chat_completion(
        base_url=base_url,
        model=model,
        system=system,
        user_payload=user_payload,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout_s=30.0,
    )


def _nemotron_provider() -> str:
    return os.getenv("NEMOTRON_PROVIDER", "local").strip().lower()


def _nemotron_cloud_headers() -> dict[str, str]:
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set for nvidia Nemotron provider")
    return {"Authorization": f"Bearer {api_key}"}


def _nemotron_cloud_base_url() -> str:
    return os.getenv("NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1").strip()


def _nemotron_cloud_model() -> str:
    return os.getenv("NVIDIA_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5").strip()


def _strip_reasoning_blocks(raw: str) -> str:
    text = raw.strip()
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start)
        if end == -1:
            break
        text = (text[:start] + text[end + len("</think>") :]).strip()
    return text.strip()


def _extract_first_object(candidate: str) -> str:
    start = candidate.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", candidate, 0)
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(candidate)):
        ch = candidate[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return candidate[start : idx + 1]
    raise json.JSONDecodeError("Unterminated JSON object", candidate, start)


def _local_nemotron_extra_body() -> dict[str, Any]:
    return {
        "stream": False,
        "response_format": {"type": "json_object"},
        "keep_alive": "30m",
        "options": {
            "temperature": 0,
            "top_p": 0.9,
            "num_predict": 900,
        },
    }


def _local_nemotron_completion_extra_body() -> dict[str, Any]:
    return {
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0,
            "top_p": 0.9,
            "num_predict": 900,
        },
    }


def _repair_json_response(
    *,
    base_url: str,
    model: str,
    raw: str,
    headers: dict[str, str] | None,
    timeout_s: float,
    use_completion_api: bool = False,
) -> str:
    stripped = _strip_reasoning_blocks(raw)
    try:
        return _extract_first_object(stripped)
    except json.JSONDecodeError:
        repair_payload = {
            "bad_response": raw[-6000:],
            "instruction": "Return only the final JSON object. No prose. No markdown. No <think> tags.",
        }
        if use_completion_api:
            repaired = _completion_completion(
                base_url=base_url,
                model=model,
                system="You repair malformed model outputs into a single valid JSON object. Return JSON only.",
                user_payload=repair_payload,
                temperature=0.0,
                max_tokens=400,
                headers=headers,
                timeout_s=timeout_s,
                extra_body=_local_nemotron_completion_extra_body(),
            )
        else:
            repaired = _chat_completion(
                base_url=base_url,
                model=model,
                system="You repair malformed model outputs into a single valid JSON object. Return JSON only.",
                user_payload=repair_payload,
                temperature=0.0,
                max_tokens=400,
                headers=headers,
                timeout_s=timeout_s,
                extra_body=_local_nemotron_extra_body(),
            )
        return _extract_first_object(_strip_reasoning_blocks(repaired))


class _LocalNemotronLock:
    def __init__(self, path: Path, timeout_s: float = 120.0) -> None:
        self.path = path
        self.timeout_s = timeout_s
        self._fh: Any | None = None

    def __enter__(self) -> "_LocalNemotronLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self._fh.write(b"0")
            self._fh.flush()
        deadline = time.time() + self.timeout_s
        while True:
            try:
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                return self
            except OSError:
                if time.time() >= deadline:
                    raise TimeoutError("Timed out waiting for local Nemotron lock")
                time.sleep(0.25)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._fh is None:
            return
        try:
            self._fh.seek(0)
            msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self._fh.close()
            self._fh = None


def nvidia_nemotron_chat(
    user_payload: dict[str, Any],
    *,
    system: str,
    max_tokens: int = 1200,
    model: str | None = None,
) -> str:
    return _chat_completion(
        base_url=_nemotron_cloud_base_url(),
        model=model or _nemotron_cloud_model(),
        system=system,
        user_payload=user_payload,
        temperature=0.1,
        max_tokens=max_tokens,
        headers=_nemotron_cloud_headers(),
        timeout_s=60.0,
    )


def nemotron_chat(user_payload: dict[str, Any], *, system: str, max_tokens: int = 800) -> str:
    provider = _nemotron_provider()
    if provider == "nvidia":
        base_url = _nemotron_cloud_base_url()
        model = _nemotron_cloud_model()
        headers = _nemotron_cloud_headers()
        timeout_s = 60.0
        retry_token_budgets = [max_tokens, max(400, max_tokens // 2), 300]
        extra_body = None
    else:
        base_url = os.getenv("NEMOTRON_BASE_URL", "http://127.0.0.1:8081")
        model = os.getenv("NEMOTRON_MODEL", "nemotron-9b")
        headers = None
        timeout_s = 90.0
        local_max_tokens = max(max_tokens, 80)
        retry_token_budgets = [
            local_max_tokens,
            max(80, local_max_tokens // 2),
            max(60, local_max_tokens // 3),
        ]
        extra_body = _local_nemotron_extra_body()
        completion_extra_body = _local_nemotron_completion_extra_body()
    last_raw = ""
    lock_ctx: Any
    lock_ctx = _LocalNemotronLock(_LOCAL_NEMOTRON_LOCK, timeout_s=max(timeout_s, 120.0)) if provider != "nvidia" else nullcontext()
    with lock_ctx:
        for token_budget in retry_token_budgets:
            try:
                if provider != "nvidia":
                    try:
                        last_raw = _chat_completion(
                            base_url=base_url,
                            model=model,
                            system=system,
                            user_payload=user_payload,
                            temperature=0.0,
                            max_tokens=token_budget,
                            headers=headers,
                            timeout_s=timeout_s,
                            extra_body=extra_body,
                        )
                        if last_raw.strip():
                            return _repair_json_response(
                                base_url=base_url,
                                model=model,
                                raw=last_raw,
                                headers=headers,
                                timeout_s=timeout_s,
                            )
                    except httpx.HTTPStatusError as exc:
                        if exc.response is None or exc.response.status_code != 404:
                            raise
                        try:
                            last_raw = _completion_completion(
                                base_url=base_url,
                                model=model,
                                system=system,
                                user_payload=user_payload,
                                temperature=0.0,
                                max_tokens=token_budget,
                                headers=headers,
                                timeout_s=timeout_s,
                                extra_body=completion_extra_body,
                            )
                        except httpx.HTTPStatusError as completion_exc:
                            if completion_exc.response is None or completion_exc.response.status_code != 404:
                                raise
                            last_raw = _ollama_generate_completion(
                                base_url=base_url,
                                model=model,
                                system=system,
                                user_payload=user_payload,
                                temperature=0.0,
                                max_tokens=token_budget,
                                headers=headers,
                                timeout_s=timeout_s,
                                extra_body=completion_extra_body,
                            )
                        if last_raw.strip():
                            return _repair_json_response(
                                base_url=base_url,
                                model=model,
                                raw=last_raw,
                                headers=headers,
                                timeout_s=timeout_s,
                                use_completion_api=True,
                            )
                        continue
                else:
                    last_raw = _chat_completion(
                        base_url=base_url,
                        model=model,
                        system=system,
                        user_payload=user_payload,
                        temperature=0.1,
                        max_tokens=token_budget,
                        headers=headers,
                        timeout_s=timeout_s,
                        extra_body=extra_body,
                    )
                if last_raw.strip():
                    return last_raw
            except httpx.HTTPStatusError as exc:
                if exc.response is None or exc.response.status_code < 500:
                    raise
            except (httpx.RequestError, TimeoutError, json.JSONDecodeError):
                pass
            time.sleep(1.0)
    return last_raw


def parse_json_response(raw: str) -> dict[str, Any]:
    text = _strip_reasoning_blocks(raw)
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = json.loads(_extract_first_object(text))
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed


def _coerce_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(_extract_first_object(_strip_reasoning_blocks(text)))
            except json.JSONDecodeError:
                return None
        return parsed if isinstance(parsed, dict) else None
    return None


def run_nemotron_tool_loop(
    initial_payload: dict[str, Any],
    *,
    system: str,
    tool_registry: dict[str, Any],
    max_round_trips: int = 2,
) -> dict[str, Any]:
    payload = sanitize_for_json(initial_payload)
    for round_trip in range(max_round_trips):
        raw = nemotron_chat(payload, system=system)
        _append_nemotron_debug(
            {
                "event": "nemotron_raw_response",
                "round_trip": round_trip,
                "payload": payload,
                "raw_response": raw,
            }
        )
        response = parse_json_response(raw)
        final_decision = _coerce_object(response.get("final_decision"))
        if final_decision is not None:
            return {"final_decision": final_decision}
        tool_name = response.get("tool")
        if isinstance(tool_name, dict):
            tool_name = tool_name.get("name")
        args = _coerce_object(response.get("args")) or {}
        if tool_name is None:
            return response
        if tool_name not in tool_registry:
            raise ValueError(f"Unknown tool requested: {tool_name}")
        if round_trip >= max_round_trips - 1:
            raise RuntimeError("Nemotron requested too many tool round trips")
        if not args:
            raise ValueError("Tool args must be a JSON object")
        result = sanitize_for_json(tool_registry[tool_name](**args))
        _append_nemotron_debug(
            {
                "event": "nemotron_tool_result",
                "round_trip": round_trip,
                "tool": tool_name,
                "args": args,
                "result": result,
            }
        )
        payload = {"tool_result": {"tool": tool_name, "result": result}}
    raise RuntimeError("Nemotron exceeded max tool-call round trips")
