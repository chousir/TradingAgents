import unittest
import sys
import types
import json
from unittest.mock import patch

if "yfinance" not in sys.modules:
    fake_yfinance = types.ModuleType("yfinance")

    class _FakeTicker:
        def __init__(self, *args, **kwargs):
            pass

        def history(self, *args, **kwargs):
            raise RuntimeError("yfinance should not be called in this test")

    fake_yfinance.Ticker = _FakeTicker
    sys.modules["yfinance"] = fake_yfinance

if "langchain_core" not in sys.modules:
    fake_langchain_core = types.ModuleType("langchain_core")
    fake_tools = types.ModuleType("langchain_core.tools")

    class _FakeTool:
        def __init__(self, func):
            self._func = func
            self.name = func.__name__

        def invoke(self, arguments):
            if isinstance(arguments, dict):
                return self._func(**arguments)
            return self._func(arguments)

        def __call__(self, *args, **kwargs):
            return self._func(*args, **kwargs)

    def fake_tool(func):
        return _FakeTool(func)

    fake_tools.tool = fake_tool
    fake_langchain_core.tools = fake_tools
    sys.modules["langchain_core"] = fake_langchain_core
    sys.modules["langchain_core.tools"] = fake_tools

from tradingagents.dataflows import market_regime


class MarketRegimeToolTests(unittest.TestCase):
    def test_market_regime_summary_combines_inputs_into_bias(self):
        def fake_schema(curr_date: str):
            if curr_date == "2026-04-04":
                return {
                    "date": curr_date,
                    "signals": {
                        "taiex_futures_foreign_oi_net": 25_000,
                        "spot_net_buy_sell": {
                            "foreign": 10_000_000,
                            "investment_trust": 2_000_000,
                            "dealer": -1_000_000,
                        },
                        "vix_close": 16.20,
                        "usd_twd_close": 31.2500,
                    },
                    "sources": {},
                    "errors": [],
                }
            return {
                "date": curr_date,
                "signals": {
                    "taiex_futures_foreign_oi_net": 20_000,
                    "spot_net_buy_sell": {
                        "foreign": 5_000_000,
                        "investment_trust": 1_000_000,
                        "dealer": -500_000,
                    },
                    "vix_close": 17.0,
                    "usd_twd_close": 31.1,
                },
                "sources": {},
                "errors": [],
            }

        with patch.object(market_regime, "collect_analyst_market_regime_schema", side_effect=fake_schema):
            summary = market_regime.get_market_regime_summary.invoke({"curr_date": "2026-04-04"})

        payload = json.loads(summary)
        self.assertEqual(payload["as_of"], "2026-04-04")
        self.assertEqual(payload["window_days"], 7)
        self.assertEqual(len(payload["signal_trend"]), 7)
        self.assertIn("market_session", payload["signal_trend"][0])
        self.assertEqual(payload["signal_trend"][0]["market_session"]["status"], "closed_weekend")
        self.assertEqual(payload["signal_trend"][0]["foreign_taiex_oi_net"], 25_000)
        self.assertEqual(payload["signal_trend"][0]["spot_net_buy_sell"]["foreign"], 10_000_000)


if __name__ == "__main__":
    unittest.main()