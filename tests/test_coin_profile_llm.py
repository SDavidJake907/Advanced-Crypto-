import unittest

from core.llm.micro_prompts import _build_nemo_observation_buckets


class CoinProfileLLMTests(unittest.TestCase):
    def test_nemo_observation_includes_coin_profile_dimensions(self) -> None:
        observation = _build_nemo_observation_buckets(
            {
                "symbol": "APT/USD",
                "lane": "L2",
                "entry_score": 71.0,
                "momentum_5": 0.01,
                "rsi": 58.0,
                "ema_9": 10.5,
                "ema_20": 10.1,
                "ema_26": 9.9,
                "ema9_above_ema20": True,
                "ema9_above_ema26": True,
                "ema_cross_distance_pct": 0.018,
                "coin_profile": {
                    "structure_quality": 84.0,
                    "momentum_quality": 77.0,
                    "volume_quality": 63.0,
                    "trade_quality": 88.0,
                    "market_support": 58.0,
                    "continuation_quality": 72.0,
                    "risk_quality": 66.0,
                },
            }
        )
        self.assertIn("coin_profile", observation)
        profile = observation["coin_profile"]
        self.assertEqual(profile["structure_quality"], 84.0)
        self.assertEqual(profile["trade_quality"], 88.0)
        self.assertEqual(profile["risk_quality"], 66.0)
        self.assertIn("ema_cross", observation)
        self.assertTrue(observation["setup_state"]["ema_cross_bullish"])
        self.assertEqual(observation["ema_cross"]["ema_26"], 9.9)
        self.assertEqual(observation["ema_cross"]["ema_cross_distance_pct"], 0.018)


if __name__ == "__main__":
    unittest.main()
