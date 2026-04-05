import os
import json
import unittest

from tradingagents.dataflows import market_regime


class MarketRegimeIntegrationTests(unittest.TestCase):
    def test_fetch_real_market_regime_data(self):
        if os.environ.get("RUN_INTEGRATION_TESTS") != "1":
            self.skipTest("Set RUN_INTEGRATION_TESTS=1 to run real market data integration tests.")

        trade_date = os.environ.get("MARKET_REGIME_TEST_DATE", "2026-04-02")

        summary = market_regime.get_market_regime_summary.invoke({"curr_date": trade_date})

        print("\n=== Market Regime Summary ===")
        print(summary)

        payload = json.loads(summary)
        self.assertEqual(payload.get("window_days"), 7)
        self.assertEqual(len(payload.get("signal_trend", [])), 7)


if __name__ == "__main__":
    unittest.main()