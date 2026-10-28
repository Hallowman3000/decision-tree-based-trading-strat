"""Deterministic checks of causality, purging and executable trade accounting."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from forex_strategy import Config, FEATURES, backtest, features, labels, metrics, walk_forward


def quotes(n=150, start="2024-01-02 07:00", spread=0.0002):
    index = pd.date_range(start, periods=n, freq="5min")
    mid = 1.1 + np.sin(np.arange(n)/7)*0.001 + np.arange(n)*0.000001
    q = pd.DataFrame(index=index)
    for side, offset in [("bid", -spread/2), ("ask", spread/2)]:
        for col, shift in [("open", 0), ("close", 0), ("high", 0.0003), ("low", -0.0003)]:
            q[f"{side}_{col}"] = mid+offset+shift
    for col in ["open", "close", "high", "low"]:
        q[col] = (q[f"bid_{col}"]+q[f"ask_{col}"])/2
    q["volume"] = 100.0
    return q


class StrategyTests(unittest.TestCase):
    def test_features_do_not_see_future(self):
        q = quotes()
        before = features(q).iloc[:100]
        q.loc[q.index[100]:, ["open", "high", "low", "close"]] *= 2
        pd.testing.assert_frame_equal(before, features(q).iloc[:100])
        self.assertEqual(list(before.columns), FEATURES)
        self.assertTrue(before.iloc[-1].notna().all())

    def test_flat_market_is_neutral_after_costs(self):
        q = quotes()
        for col in ["bid_open", "bid_close"]:
            q[col] = 1.0
        for col in ["ask_open", "ask_close"]:
            q[col] = 1.0002
        y = labels(q, "EURUSD", Config())
        self.assertTrue(y.loc[y.eligible, "target"].eq(0).all())
        self.assertFalse(y.complete_target.iloc[-13:].any())

    def test_next_open_entry_and_exact_long_short_costs(self):
        q = quotes(n=30)
        q[["bid_open", "bid_close"]] = 1.0
        q[["ask_open", "ask_close"]] = 1.0002
        c = Config()
        fee, slip, mid = c.commission_bps/10000, 0.00001, 1.0001
        expected = (-0.00022-fee*(1.00021+0.99999))/mid
        for side in [-1, 1]:
            with self.subTest(side=side):
                signals = pd.Series(0, index=q.index)
                signals.iloc[0] = side
                equity, trades = backtest(q, signals, "EURUSD", c, q.index[0], q.index[-1])
                self.assertEqual(len(trades), 1)
                self.assertEqual(trades.entry_time.iloc[0], q.index[1])
                self.assertEqual(trades.exit_time.iloc[0], q.index[13])
                self.assertAlmostEqual(trades["return"].iloc[0], expected, places=12)
                self.assertAlmostEqual(equity.iloc[-1], 1+expected, places=12)
                self.assertLess(equity.iloc[1], 1)  # spread immediately recognized

    def test_missing_bar_and_overnight_are_ineligible(self):
        q = quotes().drop(quotes().index[5])
        self.assertFalse(labels(q, "EURUSD", Config()).complete_target.iloc[0])
        self.assertTrue(labels(q, "EURUSD", Config()).eligible.iloc[0])
        evening = quotes(start="2024-01-02 23:30")
        self.assertFalse(labels(evening, "EURUSD", Config(session_start=0, session_end=24)).eligible.iloc[0])

    def test_missing_future_exit_does_not_cancel_entry(self):
        original = quotes(n=30)
        q = original.drop(original.index[13])
        signals = pd.Series(0, index=q.index)
        signals.iloc[0] = 1
        _, trades = backtest(q, signals, "EURUSD", Config(), q.index[0], q.index[-1])
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.exit_time.iloc[0], original.index[14])
        self.assertTrue(trades.delayed_exit.iloc[0])

    def test_training_labels_purged_at_test_boundary(self):
        q = quotes(n=700, start="2024-01-01 00:00")
        x, y = features(q), labels(q, "EURUSD", Config())
        start, end = pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")
        c = Config(min_train_rows=20, min_leaf=5)
        with tempfile.TemporaryDirectory() as tmp:
            predictions, folds = walk_forward(q, x, y, c, start, end, Path(tmp))
            self.assertLess(pd.Timestamp(folds[0]["last_training_label_exit"]), start)
            self.assertGreater(folds[0]["test_rows"], 0)
            # Alter all post-cutoff outcomes: training tree must be identical.
            original = (Path(tmp)/"tree_2024-01-02.txt").read_text()
            changed = y.copy()
            changed.loc[changed.label_end >= start, "target"] = 1
            walk_forward(q, x, changed, c, start, end, Path(tmp))
            self.assertEqual(original, (Path(tmp)/"tree_2024-01-02.txt").read_text())
            self.assertTrue(predictions.loc[predictions.index < start].eq(0).all())

    def test_no_overlap_and_higher_costs_reduce_equity(self):
        q = quotes(n=100)
        signals = pd.Series(1, index=q.index)
        _, trades = backtest(q, signals, "EURUSD", Config(), q.index[0], q.index[-1])
        self.assertTrue((trades.entry_time.iloc[1:].to_numpy() >= trades.exit_time.iloc[:-1].to_numpy()).all())
        cheap, _ = backtest(q, signals, "EURUSD", Config(slippage_pips=0), q.index[0], q.index[-1])
        costly, _ = backtest(q, signals, "EURUSD", Config(slippage_pips=1), q.index[0], q.index[-1])
        self.assertLess(costly.iloc[-1], cheap.iloc[-1])

    def test_no_trade_baseline_and_invalid_config(self):
        q = quotes(n=30)
        equity, trades = backtest(q, pd.Series(0, index=q.index), "EURUSD", Config(), q.index[0], q.index[-1])
        self.assertEqual(metrics(equity, trades)["trades"], 0)
        self.assertTrue(equity.eq(1).all())
        with self.assertRaises(ValueError):
            Config(slippage_pips=-1)


if __name__ == "__main__":
    unittest.main()
