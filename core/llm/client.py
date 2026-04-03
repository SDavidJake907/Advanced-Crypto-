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


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _nemotron_temperature() -> float:
    return _env_float("NEMOTRON_TEMPERATURE", 0.1)


def _local_nemotron_temperature() -> float:
    return _env_float("NEMOTRON_LOCAL_TEMPERATURE", 0.0)


def _cloud_nemotron_temperature() -> float:
    return _env_float("NEMOTRON_CLOUD_TEMPERATURE", _nemotron_temperature())


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
    return _extract_chat_content(data)


def _extract_text_from_content_part(part: Any) -> str:
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        return ""
    for key in ("text", "content", "value"):
        value = part.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            text_parts = [_extract_text_from_content_part(item) for item in value]
            joined = "".join(chunk for chunk in text_parts if chunk)
            if joined.strip():
                return joined
    if isinstance(part.get("text"), str):
        return str(part["text"])
    if part.get("type") == "text":
        inner = part.get("text")
        if isinstance(inner, str):
            return inner
        if isinstance(inner, dict):
            for key in ("value", "content"):
                value = inner.get(key)
                if isinstance(value, str):
                    return value
                if isinstance(value, list):
                    text_parts = [_extract_text_from_content_part(item) for item in value]
                    joined = "".join(chunk for chunk in text_parts if chunk)
                    if joined.strip():
                        return joined
    return ""


def _extract_chat_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        choice0 = choices[0] if isinstance(choices[0], dict) else {}
        message = choice0.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, dict):
                text = _extract_text_from_content_part(content)
                if text.strip():
                    return text
            if isinstance(content, list):
                text_parts = [_extract_text_from_content_part(part) for part in content]
                joined = "".join(part for part in text_parts if part)
                if joined.strip():
                    return joined
        for key in ("text", "output_text"):
            value = choice0.get(key)
            if isinstance(value, str) and value.strip():
                return value
        delta = choice0.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, dict):
                text = _extract_text_from_content_part(content)
                if text.strip():
                    return text
            if isinstance(content, list):
                text_parts = [_extract_text_from_content_part(part) for part in content]
                joined = "".join(part for part in text_parts if part)
                if joined.strip():
                    return joined
    output = data.get("output")
    if isinstance(output, list):
        text_parts = [_extract_text_from_content_part(part) for part in output]
        joined = "".join(part for part in text_parts if part)
        if joined.strip():
            return joined
    for key in ("response", "content", "output_text", "text"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            text = _extract_text_from_content_part(value)
            if text.strip():
                return text
        if isinstance(value, list):
            text_parts = [_extract_text_from_content_part(part) for part in value]
            joined = "".join(part for part in text_parts if part)
            if joined.strip():
                return joined
    return ""


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


def phi3_chat(user_payload: dict[str, Any], *, system: str, max_tokens: int = 400) -> str:
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


def phi3_advisory_chat(user_payload: dict[str, Any], *, system: str, max_tokens: int = 400) -> str:
    return phi3_chat(user_payload, system=system, max_tokens=max_tokens)


def _normalize_nemotron_provider(raw: str) -> str:
    raw = str(raw or "").strip().lower()
    if raw in {"cloud", "nvidia_cloud"}:
        return "nvidia"
    return raw


def _nemotron_provider() -> str:
    raw = os.getenv("NEMOTRON_STRATEGIST_PROVIDER", "").strip()
    if not raw:
        raw = os.getenv("NEMOTRON_PROVIDER", "local")
    return _normalize_nemotron_provider(raw)


def _advisory_provider() -> str:
    raw = str(os.getenv("ADVISORY_MODEL_PROVIDER", "phi3") or "").strip().lower()
    if raw in {"local_nemo", "local_nemotron", "nemo9b"}:
        return "local_nemo"
    return "phi3"


def _nemotron_cloud_headers() -> dict[str, str]:
    api_key = os.getenv("NEMOTRON_CLOUD_API_KEY", os.getenv("NVIDIA_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError("NEMOTRON_CLOUD_API_KEY or NVIDIA_API_KEY not set for nvidia Nemotron provider")
    return {"Authorization": f"Bearer {api_key}"}


def _nemotron_cloud_base_url() -> str:
    return os.getenv(
        "NEMOTRON_CLOUD_URL",
        os.getenv("NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1"),
    ).strip()


def _nemotron_cloud_model() -> str:
    return os.getenv(
        "NEMOTRON_CLOUD_MODEL",
        os.getenv("NVIDIA_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5"),
    ).strip()


def nemotron_provider_name() -> str:
    return _nemotron_provider()


def nemotron_provider_model() -> str:
    provider = _nemotron_provider()
    if provider == "nvidia":
        return _nemotron_cloud_model()
    if provider == "openai":
        return _openai_model()
    return os.getenv("NEMOTRON_MODEL", os.getenv("LOCAL_STRATEGIST_MODEL", "nemotron-9b")).strip()


def nemotron_provider_api_url() -> str:
    provider = _nemotron_provider()
    if provider == "nvidia":
        return _nemotron_cloud_base_url()
    if provider == "openai":
        return _openai_base_url()
    return os.getenv("NEMOTRON_BASE_URL", "http://127.0.0.1:8081").strip()


def advisory_provider_name() -> str:
    return _advisory_provider()


def advisory_provider_model() -> str:
    if _advisory_provider() == "local_nemo":
        return os.getenv(
            "ADVISORY_LOCAL_MODEL",
            os.getenv("NEMOTRON_MODEL", os.getenv("LOCAL_STRATEGIST_MODEL", "nemotron-9b")),
        ).strip()
    return os.getenv("PHI3_MODEL", "phi3").strip()


def advisory_provider_api_url() -> str:
    if _advisory_provider() == "local_nemo":
        return os.getenv("ADVISORY_LOCAL_BASE_URL", os.getenv("NEMOTRON_BASE_URL", "http://127.0.0.1:11434")).strip()
    return os.getenv("PHI3_BASE_URL", "http://127.0.0.1:8084").strip()


def _openai_headers() -> dict[str, str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set for openai Nemotron provider")
    return {"Authorization": f"Bearer {api_key}"}


def _openai_base_url() -> str:
    return os.getenv("OPENAI_API_URL", "https://api.openai.com/v1").strip()


def _openai_model() -> str:
    return os.getenv("OPENAI_MODEL", os.getenv("NEMOTRON_MODEL", "gpt-4.1-mini")).strip()


def _local_llm_backend() -> str:
    return str(os.getenv("LOCAL_LLM_BACKEND", "ollama") or "ollama").strip().lower()


def _strip_reasoning_blocks(raw: str) -> str:
    text = raw.strip()
    while "<think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start)
        if end == -1:
            # Truncated think block (token limit hit) — discard everything from <think> onward
            text = text[:start].strip()
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


def _local_nemotron_extra_body(max_tokens: int = 2048) -> dict[str, Any]:
    return {
        "stream": False,
        "keep_alive": "1440h",
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "options": {
            "temperature": 0,
            "top_p": 1.0,
            "num_predict": max(max_tokens, 6000),  # thinking model: ~4000 think + ~600 json
            "num_ctx": 24576,  # thinking models need more ctx: input(~2000) + think(~6000) + json(~600)
        },
    }


def _local_nemotron_completion_extra_body(max_tokens: int = 2048) -> dict[str, Any]:
    return {
        "stream": False,
        "keep_alive": "1440h",
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "options": {
            "temperature": 0,
            "top_p": 1.0,
            "num_predict": max_tokens,
            "num_ctx": 12288,
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
                extra_body=_local_nemotron_completion_extra_body(max_tokens=400),
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
                extra_body=_local_nemotron_extra_body(max_tokens=400),
            )
        return _extract_first_object(_strip_reasoning_blocks(repaired))


def cleanup_nemotron_lock() -> None:
    """Delete a stale lock file left by a crashed or killed process.

    Call this at startup before acquiring the lock. Safe to call even if
    another process legitimately holds the lock — the delete will fail
    silently in that case and the normal timeout logic handles it.
    """
    try:
        if _LOCAL_NEMOTRON_LOCK.exists():
            _LOCAL_NEMOTRON_LOCK.unlink(missing_ok=True)
    except OSError:
        pass  # File held by another live process — timeout logic will handle it


class _LocalNemotronLock:
    def __init__(self, path: Path, timeout_s: float = 30.0) -> None:
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
    max_tokens: int = 400,
    model: str | None = None,
) -> str:
    return _chat_completion(
        base_url=_nemotron_cloud_base_url(),
        model=model or _nemotron_cloud_model(),
        system=system,
        user_payload=user_payload,
        temperature=_cloud_nemotron_temperature(),
        max_tokens=max_tokens,
        headers=_nemotron_cloud_headers(),
        timeout_s=60.0,
    )


def warm_nemotron_model() -> bool:
    """Pre-load Nemotron into GPU and run 3 inference passes to fully warm CUDA kernels.

    The model requires 3-5 real inference calls before reaching full speed.
    First call alone can take 3-5 minutes (model load + kernel compilation).
    Returns True if warmup succeeded, False otherwise.
    """
    provider = _nemotron_provider()
    if provider != "local":
        return True  # Cloud providers manage their own warm state
    if _local_llm_backend() == "lmstudio":
        base_url = nemotron_provider_api_url()
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            models_url = f"{base}/models"
        else:
            models_url = f"{base}/v1/models"
        try:
            resp = httpx.get(models_url, timeout=15.0)
            resp.raise_for_status()
            return True
        except Exception as exc:
            print(f"[warm_nemotron] LM Studio readiness check failed: {exc}")
            return False
    model = os.getenv("NEMOTRON_MODEL", "nemotron-9b")
    ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    endpoint = f"{ollama_host}/api/generate"
    warmup_payload = {
        "model": model,
        "prompt": '{"action":"HOLD"}',
        "stream": False,
        "keep_alive": "1440h",
        "options": {"num_predict": 30, "temperature": 0, "num_ctx": 8192},
    }
    print("[warm_nemotron] Loading model and warming CUDA kernels (3 passes, may take 5+ min)...")
    success = False
    for i in range(3):
        try:
            resp = httpx.post(endpoint, json=warmup_payload, timeout=360.0)
            resp.raise_for_status()
            print(f"[warm_nemotron] Pass {i + 1}/3 done.")
            success = True
        except Exception as exc:
            print(f"[warm_nemotron] Pass {i + 1}/3 failed: {exc}")
    return success


def unload_nemotron_model() -> None:
    """Tell Ollama to unload Nemotron from VRAM immediately.

    Call this at graceful shutdown so the GPU is freed and the lock file
    can be deleted cleanly on the next startup. Using keep_alive=0 instructs
    Ollama to evict the model right away.
    """
    provider = _nemotron_provider()
    if provider != "local":
        return
    if _local_llm_backend() == "lmstudio":
        return
    model = os.getenv("NEMOTRON_MODEL", "nemotron-9b")
    ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    try:
        httpx.post(
            f"{ollama_host}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=10.0,
        )
    except Exception:
        pass


def _local_model_chat(
    *,
    base_url: str,
    model: str,
    system: str,
    user_payload: dict[str, Any],
    max_tokens: int,
    timeout_s: float,
) -> str:
    local_max_tokens = max(max_tokens, 80)
    retry_token_budgets = [
        local_max_tokens,
        max(200, local_max_tokens // 2),
        max(150, local_max_tokens // 3),
    ]
    extra_body = _local_nemotron_extra_body(max_tokens=local_max_tokens)
    completion_extra_body = _local_nemotron_completion_extra_body(max_tokens=local_max_tokens)
    use_ollama_generate = os.getenv("NEMOTRON_USE_OLLAMA_GENERATE", "false").lower() == "true"
    last_raw = ""
    with _LocalNemotronLock(_LOCAL_NEMOTRON_LOCK, timeout_s=max(timeout_s, 300.0)):
        for token_budget in retry_token_budgets:
            try:
                if use_ollama_generate:
                    last_raw = _ollama_generate_completion(
                        base_url=base_url,
                        model=model,
                        system=system,
                        user_payload=user_payload,
                        temperature=0.0,
                        max_tokens=token_budget,
                        headers=None,
                        timeout_s=timeout_s,
                        extra_body=completion_extra_body,
                    )
                    if last_raw.strip():
                        return _repair_json_response(
                            base_url=base_url,
                            model=model,
                            raw=last_raw,
                            headers=None,
                            timeout_s=timeout_s,
                            use_completion_api=True,
                        )
                    continue
                try:
                    last_raw = _chat_completion(
                        base_url=base_url,
                        model=model,
                        system=system,
                        user_payload=user_payload,
                        temperature=0.0,
                        max_tokens=token_budget,
                        headers=None,
                        timeout_s=timeout_s,
                        extra_body=extra_body,
                    )
                    if last_raw.strip():
                        return _repair_json_response(
                            base_url=base_url,
                            model=model,
                            raw=last_raw,
                            headers=None,
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
                            headers=None,
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
                            headers=None,
                            timeout_s=timeout_s,
                            extra_body=completion_extra_body,
                        )
                    if last_raw.strip():
                        return _repair_json_response(
                            base_url=base_url,
                            model=model,
                            raw=last_raw,
                            headers=None,
                            timeout_s=timeout_s,
                            use_completion_api=True,
                        )
                    continue
            except httpx.HTTPStatusError as exc:
                if exc.response is None or exc.response.status_code < 500:
                    raise
            except httpx.ConnectError:
                break
            except (httpx.RequestError, TimeoutError, json.JSONDecodeError):
                pass
            time.sleep(1.0)
    return last_raw


def advisory_chat(user_payload: dict[str, Any], *, system: str, max_tokens: int = 400) -> str:
    if _advisory_provider() == "local_nemo":
        return _local_model_chat(
            base_url=advisory_provider_api_url(),
            model=advisory_provider_model(),
            system=system,
            user_payload=user_payload,
            max_tokens=max_tokens,
            timeout_s=float(os.getenv("ADVISORY_LOCAL_TIMEOUT_SEC", os.getenv("NEMOTRON_LOCAL_TIMEOUT_SEC", "120"))),
        )
    return phi3_chat(user_payload, system=system, max_tokens=max_tokens)


def nemotron_chat(user_payload: dict[str, Any], *, system: str, max_tokens: int = 400) -> str:
    provider = _nemotron_provider()
    if provider == "nvidia":
        base_url = _nemotron_cloud_base_url()
        model = _nemotron_cloud_model()
        headers = _nemotron_cloud_headers()
        timeout_s = float(os.getenv("NEMOTRON_CLOUD_TIMEOUT_SEC", "60"))
        retry_token_budgets = [max_tokens, max(400, max_tokens // 2), 300]
        extra_body = None
    elif provider == "openai":
        base_url = _openai_base_url()
        model = _openai_model()
        headers = _openai_headers()
        timeout_s = float(os.getenv("NEMOTRON_OPENAI_TIMEOUT_SEC", "60"))
        retry_token_budgets = [max_tokens, max(400, max_tokens // 2), 300]
        extra_body = None
    else:
        base_url = os.getenv("NEMOTRON_BASE_URL", "http://127.0.0.1:8081")
        model = os.getenv("NEMOTRON_MODEL", "nemotron-9b")
        headers = None
        timeout_s = float(os.getenv("NEMOTRON_LOCAL_TIMEOUT_SEC", "120"))
        local_max_tokens = max(max_tokens, 80)
        retry_token_budgets = [
            local_max_tokens,
            max(200, local_max_tokens // 2),
            max(150, local_max_tokens // 3),
        ]
        extra_body = _local_nemotron_extra_body(max_tokens=local_max_tokens)
        completion_extra_body = _local_nemotron_completion_extra_body(max_tokens=local_max_tokens)
        # Skip OpenAI-compatible endpoints for completion-only models — go straight to Ollama native
        use_ollama_generate = os.getenv("NEMOTRON_USE_OLLAMA_GENERATE", "false").lower() == "true"
    last_raw = ""
    lock_ctx: Any
    lock_ctx = _LocalNemotronLock(_LOCAL_NEMOTRON_LOCK, timeout_s=max(timeout_s, 300.0)) if provider == "local" else nullcontext()
    with lock_ctx:
        for token_budget in retry_token_budgets:
            try:
                if provider == "local":
                    if use_ollama_generate:
                        last_raw = _ollama_generate_completion(
                            base_url=base_url,
                            model=model,
                            system=system,
                            user_payload=user_payload,
                            temperature=_local_nemotron_temperature(),
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
                    try:
                        last_raw = _chat_completion(
                            base_url=base_url,
                            model=model,
                            system=system,
                            user_payload=user_payload,
                            temperature=_local_nemotron_temperature(),
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
                                temperature=_local_nemotron_temperature(),
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
                                temperature=_local_nemotron_temperature(),
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
                        temperature=_cloud_nemotron_temperature(),
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
                if exc.response is None or exc.response.status_code < 500:
                    raise
            except httpx.ConnectError:
                break  # server is down — retrying with smaller token budgets won't help
            except (httpx.RequestError, TimeoutError, json.JSONDecodeError):
                pass
            time.sleep(1.0)
    if last_raw.strip():
        if provider == "local":
            return _repair_json_response(
                base_url=base_url,
                model=model,
                raw=last_raw,
                headers=headers,
                timeout_s=timeout_s,
                use_completion_api=use_ollama_generate,
            )
        return _repair_json_response(
            base_url=base_url,
            model=model,
            raw=last_raw,
            headers=headers,
            timeout_s=timeout_s,
        )
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


def _expected_symbol_from_payload(payload: dict[str, Any]) -> str | None:
    raw_symbol = payload.get("symbol")
    if isinstance(raw_symbol, str) and raw_symbol.strip():
        return raw_symbol.strip()
    features = payload.get("features")
    if isinstance(features, dict):
        feature_symbol = features.get("symbol")
        if isinstance(feature_symbol, str) and feature_symbol.strip():
            return feature_symbol.strip()
    return None


def _sanitize_final_decision_symbol(final_decision: dict[str, Any], *, expected_symbol: str | None) -> dict[str, Any]:
    if not expected_symbol:
        return final_decision
    cleaned = dict(final_decision)
    raw_symbol = str(cleaned.get("symbol", "") or "").strip()
    if raw_symbol in {"", "SYMBOL", "<current_symbol>", "CURRENT_SYMBOL", "LINK/USD", "BTC/USD", "ETH/USD"}:
        cleaned["symbol"] = expected_symbol
    return cleaned


def run_nemotron_tool_loop(
    initial_payload: dict[str, Any],
    *,
    system: str,
    tool_registry: dict[str, Any],
    max_round_trips: int = 2,
) -> dict[str, Any]:
    payload = sanitize_for_json(initial_payload)
    expected_symbol = _expected_symbol_from_payload(payload)
    for round_trip in range(max_round_trips):
        raw = nemotron_chat(payload, system=system, max_tokens=400)
        _append_nemotron_debug(
            {
                "event": "nemotron_raw_response",
                "round_trip": round_trip,
                "payload": payload,
                "raw_response": raw,
            }
        )
        response = parse_json_response(raw)
        tool_name = response.get("tool")
        if isinstance(tool_name, dict):
            tool_name = tool_name.get("name")
        final_decision = _coerce_object(response.get("final_decision"))
        if final_decision is not None:
            final_decision = _sanitize_final_decision_symbol(final_decision, expected_symbol=expected_symbol)
        # Model always echoes tool call even after receiving a result.
        # Round 0: run the tool (ignore embedded final_decision).
        # Round 1+: model still includes tool but we already ran it — trust final_decision.
        if tool_name is not None and final_decision is not None and round_trip > 0:
            return {"final_decision": final_decision}
        if tool_name is None:
            if final_decision is not None:
                return {"final_decision": final_decision}
            return response
        args = _coerce_object(response.get("args")) or {}
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
