import json
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import httpx

from core.llm.client import nemotron_chat, parse_json_response


class NemotronClientTests(unittest.TestCase):
    def test_local_provider_falls_back_to_completion_endpoint_on_chat_404(self) -> None:
        def fake_post(url: str, **kwargs: object) -> httpx.Response:
            request = httpx.Request("POST", url)
            if url.endswith("/v1/chat/completions"):
                response = httpx.Response(404, request=request)
                raise httpx.HTTPStatusError("not found", request=request, response=response)
            if url.endswith("/v1/completions"):
                payload = {
                    "choices": [
                        {
                            "text": (
                                "<think>\ninternal reasoning\n</think>\n"
                                '{"final_decision":{"symbol":"TEST/USD","action":"HOLD",'
                                '"side":null,"size":0,"reason":"ok","debug":{}}}'
                            )
                        }
                    ]
                }
                return httpx.Response(200, request=request, json=payload)
            self.fail(f"unexpected url {url}")

        with patch.dict(
            "os.environ",
            {
                "NEMOTRON_PROVIDER": "local",
                "NEMOTRON_BASE_URL": "http://127.0.0.1:11434",
                "NEMOTRON_MODEL": "mirage335/NVIDIA-Nemotron-Nano-9B-v2-virtuoso:latest",
            },
            clear=False,
        ):
            with patch("core.llm.client._LocalNemotronLock", return_value=nullcontext()):
                with patch("core.llm.client.httpx.post", side_effect=fake_post):
                    raw = nemotron_chat(
                        {"symbol": "TEST/USD"},
                        system="Return final_decision JSON only.",
                        max_tokens=64,
                    )

        parsed = parse_json_response(raw)
        self.assertEqual(parsed["final_decision"]["symbol"], "TEST/USD")
        self.assertEqual(parsed["final_decision"]["action"], "HOLD")
        self.assertEqual(parsed["final_decision"]["reason"], "ok")

    def test_local_provider_falls_back_to_ollama_generate_when_completion_404s(self) -> None:
        def fake_post(url: str, **kwargs: object) -> httpx.Response:
            request = httpx.Request("POST", url)
            if url.endswith("/v1/chat/completions"):
                response = httpx.Response(404, request=request)
                raise httpx.HTTPStatusError("not found", request=request, response=response)
            if url.endswith("/v1/completions"):
                response = httpx.Response(404, request=request)
                raise httpx.HTTPStatusError("not found", request=request, response=response)
            if url.endswith("/api/generate"):
                payload = {
                    "response": (
                        "<think>\ninternal reasoning\n</think>\n"
                        '{"final_decision":{"symbol":"TEST/USD","action":"HOLD",'
                        '"side":null,"size":0,"reason":"ok_generate","debug":{}}}'
                    )
                }
                return httpx.Response(200, request=request, json=payload)
            self.fail(f"unexpected url {url}")

        with patch.dict(
            "os.environ",
            {
                "NEMOTRON_PROVIDER": "local",
                "NEMOTRON_BASE_URL": "http://127.0.0.1:11434",
                "NEMOTRON_MODEL": "mirage335/NVIDIA-Nemotron-Nano-9B-v2-virtuoso:latest",
            },
            clear=False,
        ):
            with patch("core.llm.client._LocalNemotronLock", return_value=nullcontext()):
                with patch("core.llm.client.httpx.post", side_effect=fake_post):
                    raw = nemotron_chat(
                        {"symbol": "TEST/USD"},
                        system="Return final_decision JSON only.",
                        max_tokens=64,
                    )

        parsed = parse_json_response(raw)
        self.assertEqual(parsed["final_decision"]["symbol"], "TEST/USD")
        self.assertEqual(parsed["final_decision"]["action"], "HOLD")
        self.assertEqual(parsed["final_decision"]["reason"], "ok_generate")


if __name__ == "__main__":
    unittest.main()
