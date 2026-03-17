from __future__ import annotations

import json
import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256


class GatekeepRequest(BaseModel):
    trade_snapshot: dict
    max_new_tokens: int = 256


class GatekeepResponse(BaseModel):
    approve: bool
    reason: str
    final_size_usd: float
    tif: str
    allow_partial: bool
    rejection_code: str | None = None


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    base_url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8002")
    model = os.getenv("VLLM_MODEL", "Qwen/Qwen3-4B-Instruct")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": req.max_new_tokens,
        "temperature": 0.2,
    }
    try:
        r = httpx.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120.0)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return {"text": text}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/gatekeep_trade")
def gatekeep_trade(req: GatekeepRequest) -> dict:
    base_url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8002")
    model = os.getenv("VLLM_MODEL", "Qwen/Qwen3-4B-Instruct")

    system = (
        "You are a trading risk gatekeeper. "
        "Return ONLY valid JSON with keys: approve (bool), reason (string), "
        "final_size_usd (number), tif (string), allow_partial (bool), "
        "rejection_code (string or null)."
    )
    user = json.dumps(req.trade_snapshot)
    prompt = f"{system}\n\nTRADE_SNAPSHOT_JSON:\n{user}\n\nJSON:"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": req.max_new_tokens,
        "temperature": 0.1,
    }
    try:
        r = httpx.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120.0)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(text)
            validated = GatekeepResponse(**parsed)
            return validated.model_dump()
        except Exception:
            # strict schema: reject by default
            return {
                "approve": False,
                "reason": "schema_invalid",
                "final_size_usd": 0.0,
                "tif": "IOC",
                "allow_partial": False,
                "rejection_code": "schema_invalid",
            }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
