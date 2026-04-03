import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import httpx

from core.execution.cpp_exec import CppExecutor
from core.llm.nemotron import NemotronDecision, NemotronStrategist
from core.llm.client import (
    advisory_chat,
    advisory_provider_name,
    nemotron_chat,
    nemotron_provider_name,
    parse_json_response,
    run_nemotron_tool_loop,
    unload_nemotron_model,
    warm_nemotron_model,
)
from core.risk.basic_risk import BasicRiskEngine
from core.risk.portfolio import PortfolioConfig, PositionState
from core.state.portfolio import PortfolioState
from core.strategy.simple_momo import SimpleMomentumStrategy
from core.llm.prompts import get_nemotron_batch_strategist_prompt, get_nemotron_strategist_system_prompt


class NemotronClientTests(unittest.TestCase):
    def test_cloud_provider_parses_list_style_message_content(self) -> None:
        def fake_post(url: str, **kwargs: object) -> httpx.Response:
            request = httpx.Request("POST", url)
            payload = {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": '{"final_decision":{"symbol":"TEST/USD","action":"HOLD","side":null,"size":0,"reason":"list_content_ok","debug":{}}}',
                                }
                            ]
                        }
                    }
                ]
            }
            return httpx.Response(200, request=request, json=payload)

        with patch.dict(
            "os.environ",
            {
                "NEMOTRON_PROVIDER": "cloud",
                "NEMOTRON_CLOUD_API_KEY": "test-cloud-key",
                "NEMOTRON_CLOUD_URL": "https://integrate.api.nvidia.com/v1",
                "NEMOTRON_CLOUD_MODEL": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
            },
            clear=False,
        ):
            with patch("core.llm.client.httpx.post", side_effect=fake_post):
                raw = nemotron_chat(
                    {"symbol": "TEST/USD"},
                    system="Return final_decision JSON only.",
                    max_tokens=64,
                )

        parsed = parse_json_response(raw)
        self.assertEqual(parsed["final_decision"]["reason"], "list_content_ok")

    def test_cloud_provider_parses_choice_text_fallback(self) -> None:
        def fake_post(url: str, **kwargs: object) -> httpx.Response:
            request = httpx.Request("POST", url)
            payload = {
                "choices": [
                    {
                        "text": '{"final_decision":{"symbol":"TEST/USD","action":"HOLD","side":null,"size":0,"reason":"choice_text_ok","debug":{}}}'
                    }
                ]
            }
            return httpx.Response(200, request=request, json=payload)

        with patch.dict(
            "os.environ",
            {
                "NEMOTRON_PROVIDER": "cloud",
                "NEMOTRON_CLOUD_API_KEY": "test-cloud-key",
                "NEMOTRON_CLOUD_URL": "https://integrate.api.nvidia.com/v1",
                "NEMOTRON_CLOUD_MODEL": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
            },
            clear=False,
        ):
            with patch("core.llm.client.httpx.post", side_effect=fake_post):
                raw = nemotron_chat(
                    {"symbol": "TEST/USD"},
                    system="Return final_decision JSON only.",
                    max_tokens=64,
                )

        parsed = parse_json_response(raw)
        self.assertEqual(parsed["final_decision"]["reason"], "choice_text_ok")

    def test_cloud_provider_parses_output_array_shape(self) -> None:
        def fake_post(url: str, **kwargs: object) -> httpx.Response:
            request = httpx.Request("POST", url)
            payload = {
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"final_decision":{"symbol":"TEST/USD","action":"HOLD","side":null,"size":0,"reason":"output_array_ok","debug":{}}}',
                            }
                        ]
                    }
                ]
            }
            return httpx.Response(200, request=request, json=payload)

        with patch.dict(
            "os.environ",
            {
                "NEMOTRON_PROVIDER": "cloud",
                "NEMOTRON_CLOUD_API_KEY": "test-cloud-key",
                "NEMOTRON_CLOUD_URL": "https://integrate.api.nvidia.com/v1",
                "NEMOTRON_CLOUD_MODEL": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
            },
            clear=False,
        ):
            with patch("core.llm.client.httpx.post", side_effect=fake_post):
                raw = nemotron_chat(
                    {"symbol": "TEST/USD"},
                    system="Return final_decision JSON only.",
                    max_tokens=64,
                )

        parsed = parse_json_response(raw)
        self.assertEqual(parsed["final_decision"]["reason"], "output_array_ok")

    def test_cloud_strategist_prompt_selector_uses_cloud_prompt(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "NEMOTRON_STRATEGIST_PROVIDER": "cloud",
                "NEMOTRON_CLOUD_API_KEY": "test-cloud-key",
            },
            clear=False,
        ):
            prompt = get_nemotron_strategist_system_prompt()
            batch_prompt = get_nemotron_batch_strategist_prompt()

        self.assertIn("cloud runtime", prompt)
        self.assertIn("Do not output tool calls.", batch_prompt)

    def test_local_strategist_prompt_selector_uses_local_prompt(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "NEMOTRON_STRATEGIST_PROVIDER": "local",
            },
            clear=False,
        ):
            prompt = get_nemotron_strategist_system_prompt()
            batch_prompt = get_nemotron_batch_strategist_prompt()

        self.assertIn("local runtime", prompt)
        self.assertIn("Return compact JSON immediately.", batch_prompt)

    def test_advisory_local_nemo_uses_local_model_even_when_strategist_is_cloud(self) -> None:
        urls: list[str] = []

        def fake_post(url: str, **kwargs: object) -> httpx.Response:
            urls.append(url)
            request = httpx.Request("POST", url)
            if url.endswith("/v1/chat/completions"):
                response = httpx.Response(404, request=request)
                raise httpx.HTTPStatusError("not found", request=request, response=response)
            if url.endswith("/v1/completions"):
                payload = {"choices": [{"text": '{"reflex":"allow","micro_state":"stable","reason":"local_nemo_ok"}'}]}
                return httpx.Response(200, request=request, json=payload)
            self.fail(f"unexpected url {url}")

        with patch.dict(
            "os.environ",
            {
                "ADVISORY_MODEL_PROVIDER": "local_nemo",
                "ADVISORY_LOCAL_BASE_URL": "http://127.0.0.1:11434",
                "ADVISORY_LOCAL_MODEL": "mirage335/NVIDIA-Nemotron-Nano-9B-v2-virtuoso:latest",
                "NEMOTRON_STRATEGIST_PROVIDER": "cloud",
                "NEMOTRON_CLOUD_API_KEY": "test-cloud-key",
            },
            clear=False,
        ):
            with patch("core.llm.client._LocalNemotronLock", return_value=nullcontext()):
                with patch("core.llm.client.httpx.post", side_effect=fake_post):
                    raw = advisory_chat({"symbol": "TEST/USD"}, system="Return JSON only.", max_tokens=64)
                    advisory_provider = advisory_provider_name()
                    strategist_provider = nemotron_provider_name()

        self.assertEqual(json.loads(raw)["reason"], "local_nemo_ok")
        self.assertEqual(advisory_provider, "local_nemo")
        self.assertEqual(strategist_provider, "nvidia")
        self.assertTrue(any(url.endswith("/v1/completions") for url in urls))

    def test_local_provider_is_respected_even_when_nvidia_key_exists(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "ADVISORY_MODEL_PROVIDER": "local_nemo",
                "NEMOTRON_PROVIDER": "local",
                "NVIDIA_API_KEY": "test-cloud-key",
            },
            clear=False,
        ):
            strategist_provider = nemotron_provider_name()

        self.assertEqual(strategist_provider, "local")

    def test_tool_loop_rewrites_current_symbol_placeholder(self) -> None:
        responses = iter(
            [
                '{"tool":"strategy_decision","args":{"symbol":"TEST/USD","features":{"symbol":"TEST/USD"}}}',
                '{"final_decision":{"symbol":"CURRENT_SYMBOL","action":"OPEN","side":"LONG","size":10,"reason":"strong_entry","debug":{}},"tool":"strategy_decision","args":{"symbol":"CURRENT_SYMBOL"}}',
            ]
        )

        def fake_chat(payload: dict[str, object], *, system: str, max_tokens: int) -> str:
            self.assertEqual(system, "Return final_decision JSON only.")
            self.assertEqual(max_tokens, 400)
            return next(responses)

        with patch("core.llm.client.nemotron_chat", side_effect=fake_chat):
            parsed = run_nemotron_tool_loop(
                {"symbol": "TEST/USD", "features": {"symbol": "TEST/USD"}},
                system="Return final_decision JSON only.",
                tool_registry={"strategy_decision": lambda **kwargs: {"ok": True, "symbol": kwargs.get("symbol")}},
            )

        self.assertEqual(parsed["final_decision"]["symbol"], "TEST/USD")
        self.assertEqual(parsed["final_decision"]["action"], "OPEN")

    def test_openai_provider_uses_chat_completion(self) -> None:
        def fake_post(url: str, **kwargs: object) -> httpx.Response:
            request = httpx.Request("POST", url)
            self.assertEqual(url, "https://api.openai.com/v1/chat/completions")
            self.assertEqual(kwargs.get("headers"), {"Authorization": "Bearer test-openai-key"})
            payload = {
                "choices": [
                    {
                        "message": {
                            "content": '{"final_decision":{"symbol":"TEST/USD","action":"HOLD","side":null,"size":0,"reason":"openai_ok","debug":{}}}'
                        }
                    }
                ]
            }
            return httpx.Response(200, request=request, json=payload)

        with patch.dict(
            "os.environ",
            {
                "NEMOTRON_PROVIDER": "openai",
                "NEMOTRON_STRATEGIST_PROVIDER": "openai",
                "OPENAI_API_KEY": "test-openai-key",
                "OPENAI_API_URL": "https://api.openai.com/v1",
                "OPENAI_MODEL": "gpt-4.1-mini",
            },
            clear=False,
        ):
            with patch("core.llm.client.httpx.post", side_effect=fake_post):
                raw = nemotron_chat(
                    {"symbol": "TEST/USD"},
                    system="Return final_decision JSON only.",
                    max_tokens=64,
                )

        parsed = parse_json_response(raw)
        self.assertEqual(parsed["final_decision"]["symbol"], "TEST/USD")
        self.assertEqual(parsed["final_decision"]["action"], "HOLD")
        self.assertEqual(parsed["final_decision"]["reason"], "openai_ok")

    def test_openai_provider_repairs_non_json_prefix(self) -> None:
        calls = {"count": 0}

        def fake_post(url: str, **kwargs: object) -> httpx.Response:
            request = httpx.Request("POST", url)
            calls["count"] += 1
            if calls["count"] == 1:
                payload = {
                    "choices": [
                        {
                            "message": {
                                "content": 'analysis first\n{"final_decision":{"symbol":"TEST/USD","action":"HOLD","side":null,"size":0,"reason":"repair_ok","debug":{}}}'
                            }
                        }
                    ]
                }
                return httpx.Response(200, request=request, json=payload)
            self.fail("repair path should not require a second request for prefixed JSON")

        with patch.dict(
            "os.environ",
            {
                "NEMOTRON_PROVIDER": "openai",
                "NEMOTRON_STRATEGIST_PROVIDER": "openai",
                "OPENAI_API_KEY": "test-openai-key",
                "OPENAI_API_URL": "https://api.openai.com/v1",
                "OPENAI_MODEL": "gpt-4.1-mini",
            },
            clear=False,
        ):
            with patch("core.llm.client.httpx.post", side_effect=fake_post):
                raw = nemotron_chat(
                    {"symbol": "TEST/USD"},
                    system="Return final_decision JSON only.",
                    max_tokens=64,
                )

        parsed = parse_json_response(raw)
        self.assertEqual(parsed["final_decision"]["reason"], "repair_ok")

    def test_cloud_provider_alias_uses_nvidia_route_and_cloud_env_names(self) -> None:
        def fake_post(url: str, **kwargs: object) -> httpx.Response:
            request = httpx.Request("POST", url)
            self.assertEqual(url, "https://integrate.api.nvidia.com/v1/chat/completions")
            self.assertEqual(kwargs.get("headers"), {"Authorization": "Bearer test-cloud-key"})
            payload = {
                "choices": [
                    {
                        "message": {
                            "content": '{"final_decision":{"symbol":"TEST/USD","action":"HOLD","side":null,"size":0,"reason":"cloud_ok","debug":{}}}'
                        }
                    }
                ]
            }
            return httpx.Response(200, request=request, json=payload)

        with patch.dict(
            "os.environ",
            {
                "NEMOTRON_PROVIDER": "cloud",
                "NEMOTRON_STRATEGIST_PROVIDER": "cloud",
                "NEMOTRON_CLOUD_API_KEY": "test-cloud-key",
                "NEMOTRON_CLOUD_URL": "https://integrate.api.nvidia.com/v1",
                "NEMOTRON_CLOUD_MODEL": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
            },
            clear=False,
        ):
            with patch("core.llm.client.httpx.post", side_effect=fake_post):
                raw = nemotron_chat(
                    {"symbol": "TEST/USD"},
                    system="Return final_decision JSON only.",
                    max_tokens=64,
                )

        parsed = parse_json_response(raw)
        self.assertEqual(parsed["final_decision"]["reason"], "cloud_ok")

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

    def test_warm_nemotron_model_uses_lmstudio_readiness_when_configured(self) -> None:
        request_urls: list[str] = []

        def fake_get(url: str, **kwargs: object) -> httpx.Response:
            request_urls.append(url)
            return httpx.Response(200, request=httpx.Request("GET", url), json={"data": [{"id": "gemma4-e4b-it"}]})

        with patch.dict(
            "os.environ",
            {
                "NEMOTRON_PROVIDER": "local",
                "NEMOTRON_STRATEGIST_PROVIDER": "local",
                "LOCAL_LLM_BACKEND": "lmstudio",
                "NEMOTRON_BASE_URL": "http://127.0.0.1:1234",
                "NEMOTRON_MODEL": "gemma4-e4b-it",
            },
            clear=False,
        ):
            with patch("core.llm.client.httpx.get", side_effect=fake_get):
                self.assertTrue(warm_nemotron_model())

        self.assertEqual(request_urls, ["http://127.0.0.1:1234/v1/models"])

    def test_unload_nemotron_model_skips_http_call_for_lmstudio(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "NEMOTRON_PROVIDER": "local",
                "NEMOTRON_STRATEGIST_PROVIDER": "local",
                "LOCAL_LLM_BACKEND": "lmstudio",
                "NEMOTRON_BASE_URL": "http://127.0.0.1:1234",
                "NEMOTRON_MODEL": "gemma4-e4b-it",
            },
            clear=False,
        ):
            with patch("core.llm.client.httpx.post") as post_mock:
                unload_nemotron_model()
        post_mock.assert_not_called()


class NemotronBatchParsingTests(unittest.TestCase):
    def _make_strategist(self) -> NemotronStrategist:
        strategist = NemotronStrategist(
            strategy=SimpleMomentumStrategy(),
            risk_engine=BasicRiskEngine(max_position_notional=1_000_000.0, max_leverage=10.0, cooldown_bars=0),
            portfolio_config=PortfolioConfig(max_open_positions=4),
            executor=CppExecutor(),
        )
        strategist._memory_store.build_lessons_block = lambda max_lessons=8: ""
        strategist._memory_store.build_behavior_score_block = lambda lookback=50: None
        return strategist

    def _candidate(self, symbol: str = "BTC/USD") -> dict[str, object]:
        return {
            "symbol": symbol,
            "features": {
                "symbol": symbol,
                "lane": "L2",
                "entry_score": 48.0,
                "entry_recommendation": "WATCH",
                "reversal_risk": "MEDIUM",
                "rotation_score": 0.1,
                "momentum_5": 0.01,
                "momentum_14": 0.02,
                "rsi": 54.0,
                "volume_ratio": 1.2,
                "macd_hist": 0.1,
                "adx": 21.0,
                "trend_confirmed": True,
                "ranging_market": False,
                "structure_quality": 52.0,
                "trade_quality": 90.0,
                "risk_quality": 80.0,
                "ema9_above_ema20": True,
                "range_breakout_1h": False,
                "pullback_hold": True,
                "higher_low_count": 3,
                "point_breakdown": {"net_edge_pct": 0.4, "cost_penalty_pts": 0.0},
                "bar_ts": "2026-03-28T03:00:00+00:00",
                "bar_idx": 1,
            },
            "proposed_weight": 0.1,
            "reflex": {"reflex": "allow", "micro_state": "stable", "reason": "test"},
            "phi3_ms": 0.0,
        }

    def _strong_buy_candidate(self, symbol: str = "BTC/USD") -> dict[str, object]:
        candidate = self._candidate(symbol)
        features = dict(candidate["features"])
        features.update(
            {
                "entry_score": 72.0,
                "entry_recommendation": "BUY",
                "reversal_risk": "MEDIUM",
                "volume_ratio": 1.4,
                "trade_quality": 92.0,
                "structure_quality": 70.0,
                "continuation_quality": 72.0,
                "short_tf_ready_15m": True,
                "tp_after_cost_valid": True,
                "net_edge_pct": 0.4,
                "trend_1h": 1,
                "indicators_ready": True,
            }
        )
        candidate["features"] = features
        return candidate

    def test_batch_decide_parses_tool_result_wrapped_decisions(self) -> None:
        strategist = self._make_strategist()
        portfolio_state = PortfolioState(cash=100.0)
        positions_state = PositionState()

        raw = (
            '{"tool_result":{"result":{"reasoning":"wrapped","decisions":'
            '[{"symbol":"BTC/USD","action":"HOLD","reason":"wrapped_hold"}]}}}'
        )

        with patch("core.llm.nemotron.nemotron_chat", return_value=raw):
            with patch("core.llm.nemotron.nemotron_provider_name", return_value="nvidia"):
                with patch("core.llm.nemotron.strategy_decision", return_value="FLAT"):
                    with patch("core.llm.nemotron.risk_adjust", return_value=["no_action"]):
                        with patch(
                            "core.llm.nemotron.portfolio_evaluate",
                            return_value={"decision": "allow", "size_factor": 1.0, "reasons": []},
                        ):
                            results = strategist.batch_decide(
                                candidates=[self._candidate()],
                                portfolio_state=portfolio_state,
                                positions_state=positions_state,
                                symbols=["BTC/USD"],
                            )

        self.assertEqual(results["BTC/USD"].execution["nemotron"]["reason"], "wrapped_hold")
        self.assertEqual(results["BTC/USD"].signal, "FLAT")

    def test_batch_decide_falls_back_when_decisions_missing(self) -> None:
        strategist = self._make_strategist()
        portfolio_state = PortfolioState(cash=100.0)
        positions_state = PositionState()

        raw = '{"tool":"strategy_decision","args":{"symbol":"BTC/USD"}}'

        with patch("core.llm.nemotron.nemotron_chat", return_value=raw):
            with patch("core.llm.nemotron.nemotron_provider_name", return_value="nvidia"):
                with patch("core.llm.nemotron.strategy_decision", return_value="FLAT"):
                    with patch("core.llm.nemotron.risk_adjust", return_value=["no_action"]):
                        with patch(
                            "core.llm.nemotron.portfolio_evaluate",
                            return_value={"decision": "allow", "size_factor": 1.0, "reasons": []},
                        ):
                            results = strategist.batch_decide(
                                candidates=[self._candidate()],
                                portfolio_state=portfolio_state,
                                positions_state=positions_state,
                                symbols=["BTC/USD"],
                            )

        self.assertEqual(results["BTC/USD"].execution["nemotron"]["reason"], "batch_parse_fallback_hold")

    def test_local_batch_failure_falls_back_to_single_decide(self) -> None:
        strategist = self._make_strategist()
        portfolio_state = PortfolioState(cash=100.0)
        positions_state = PositionState()
        single_decision = NemotronDecision(
            signal="FLAT",
            risk_checks=["no_action"],
            portfolio_decision={"decision": "allow", "size_factor": 1.0, "reasons": []},
            execution={"status": "no_trade", "nemotron": {"reason": "single_fallback"}},
            reflex={},
            timings={},
        )

        with patch("core.llm.nemotron.nemotron_chat", return_value=""):
            with patch("core.llm.nemotron.nemotron_provider_name", return_value="local"):
                with patch.dict("os.environ", {"NEMOTRON_LOCAL_BATCH_SINGLE_FALLBACK": "true"}, clear=False):
                    with patch.object(strategist, "decide", return_value=single_decision) as decide_mock:
                        results = strategist.batch_decide(
                            candidates=[self._candidate()],
                            portfolio_state=portfolio_state,
                            positions_state=positions_state,
                            symbols=["BTC/USD"],
                        )

        self.assertIs(results["BTC/USD"], single_decision)
        decide_mock.assert_called_once()

    def test_batch_parse_failure_uses_deterministic_open_for_strong_buy_candidate(self) -> None:
        strategist = self._make_strategist()
        portfolio_state = PortfolioState(cash=100.0)
        positions_state = PositionState()

        with patch("core.llm.nemotron.nemotron_chat", return_value=""):
            with patch("core.llm.nemotron.nemotron_provider_name", return_value="nvidia"):
                with patch("core.llm.nemotron.strategy_decision", return_value="LONG"):
                    with patch("core.llm.nemotron.risk_adjust", return_value=[]):
                        with patch(
                            "core.llm.nemotron.portfolio_evaluate",
                            return_value={"decision": "allow", "size_factor": 1.0, "reasons": []},
                        ):
                            results = strategist.batch_decide(
                                candidates=[self._strong_buy_candidate()],
                                portfolio_state=portfolio_state,
                                positions_state=positions_state,
                                symbols=["BTC/USD"],
                            )

        self.assertEqual(results["BTC/USD"].execution["nemotron"]["reason"], "model_parse_failed_deterministic_open")
        self.assertEqual(results["BTC/USD"].signal, "LONG")

    def test_single_parse_failure_uses_deterministic_open_for_strong_buy_candidate(self) -> None:
        strategist = self._make_strategist()
        portfolio_state = PortfolioState(cash=100.0)
        positions_state = PositionState()
        candidate = self._strong_buy_candidate()

        with patch("core.llm.nemotron.nemotron_chat", return_value=""):
            with patch("core.llm.nemotron.build_advisory_bundle") as advisory_mock:
                advisory_mock.return_value = type(
                    "Bundle",
                    (),
                    {
                        "reflex": {"reflex": "allow", "micro_state": "stable", "reason": "test"},
                        "market_state_review": {},
                        "visual_review": {},
                        "timings": {"phi3_ms": 0.0, "advisory_ms": 0.0},
                    },
                )()
                with patch("core.llm.nemotron.strategy_decision", return_value="LONG"):
                    with patch("core.llm.nemotron.risk_adjust", return_value=[]):
                        with patch(
                            "core.llm.nemotron.portfolio_evaluate",
                            return_value={"decision": "allow", "size_factor": 1.0, "reasons": []},
                        ):
                            result = strategist.decide(
                                symbol="BTC/USD",
                                features=candidate["features"],
                                portfolio_state=portfolio_state,
                                positions_state=positions_state,
                                symbols=["BTC/USD"],
                                proposed_weight=0.1,
                            )

        self.assertEqual(result.execution["nemotron"]["reason"], "model_parse_failed_deterministic_open")
        self.assertEqual(result.signal, "LONG")

    def test_single_fallback_logging_dedupes_repeated_identical_errors(self) -> None:
        strategist = self._make_strategist()
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch("core.llm.nemotron.rotate_jsonl_if_needed"):
                    strategist._log_fallback({"symbol": "BTC/USD"}, ValueError("No JSON object found"))
                    strategist._log_fallback({"symbol": "BTC/USD"}, ValueError("No JSON object found"))
                log_path = os.path.join(tmpdir, "logs", "nemotron_debug.jsonl")
                with open(log_path, "r", encoding="utf-8") as handle:
                    lines = [line for line in handle.read().splitlines() if line.strip()]
            finally:
                os.chdir(cwd)
        self.assertEqual(len(lines), 1)


if __name__ == "__main__":
    unittest.main()
