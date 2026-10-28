"""Tests for forex data causality and TiRex-2 wiring, not pretrained accuracy."""
import unittest

import numpy as np
import pandas as pd

from forex_strategy import Config
from tirex2_forex import (calendar_covariates, choose_origins, forecast_batch,
                          forecast_metrics, hourly_panel, make_inputs, trade_signal)


def panel():
    index = pd.date_range("2026-09-25 00:55", periods=200, freq="h")
    price = pd.DataFrame({"EURUSD": 1.1+np.arange(200)*0.00001,
                          "USDJPY": 150+np.arange(200)*0.001}, index=index)
    return price, price*0+1


class ForecastExperimentTests(unittest.TestCase):
    def test_hourly_completion_and_missing_bars(self):
        index = pd.date_range("2026-10-01", periods=36, freq="5min")
        q = pd.DataFrame({"close": 1.0+np.arange(36)*0.0001}, index=index)
        q["bid_close"], q["ask_close"] = q.close-0.00005, q.close+0.00005
        price, _ = hourly_panel({"EURUSD": q})
        self.assertEqual(price.index[0], pd.Timestamp("2026-10-01 00:55"))
        self.assertEqual(price.iloc[0,0], q.close.iloc[11])
        damaged, _ = hourly_panel({"EURUSD": q.drop(q.index[13])})
        self.assertTrue(pd.isna(damaged.iloc[1,0]))
        changed = q.copy()
        changed.loc[q.index[12]:, "close"] *= 2
        updated, _ = hourly_panel({"EURUSD": changed})
        self.assertEqual(price.iloc[0,0], updated.iloc[0,0])

    def test_future_prices_and_spreads_never_enter_inputs(self):
        price, spread = panel()
        origin = price.index[150]
        original = make_inputs(price, spread, origin, 128)
        price.loc[price.index > origin] *= 10
        spread.loc[spread.index > origin] *= 100
        changed = make_inputs(price, spread, origin, 128)
        np.testing.assert_array_equal(original.target, changed.target)
        np.testing.assert_array_equal(original.past_covariates, changed.past_covariates)
        np.testing.assert_array_equal(original.future_covariates, changed.future_covariates)
        self.assertEqual(original.target.shape, (2,128))
        self.assertEqual(original.past_covariates.shape, (2,128))
        self.assertEqual(original.future_covariates.shape, (4,129))
        self.assertEqual(original.timestamps[-1], origin+pd.Timedelta(hours=1))

    def test_origin_eligibility_does_not_use_future_outcome(self):
        price, _ = panel()
        origin = pd.Timestamp("2026-10-01 08:55")
        before, _ = choose_origins(price, origin, origin+pd.Timedelta(hours=3), 128)
        price.loc[origin+pd.Timedelta(hours=1)] = np.nan
        after, _ = choose_origins(price, origin, origin+pd.Timedelta(hours=3), 128)
        self.assertIn(origin, before)
        self.assertIn(origin, after)

    def test_calendar_is_known_and_bounded(self):
        values = calendar_covariates(pd.date_range("2026-10-01", periods=48, freq="h"))
        self.assertEqual(values.shape, (4,48))
        self.assertTrue((np.abs(values) <= 1).all())
        np.testing.assert_allclose(values[:2,0], values[:2,24], atol=1e-6)

    def test_official_adapter_shapes_and_quantile_selection(self):
        # This checks the installed TimeseriesType adapter with a test double;
        # it deliberately does not claim to exercise pretrained model weights.
        try:
            import torch
            import tirex2  # noqa: F401
        except ImportError:
            self.skipTest("Install requirements-tirex2.txt to test the official adapter")
        price, spread = panel()
        item = make_inputs(price, spread, price.index[150], 128)

        class ModelDouble:
            quantiles = torch.tensor([i/10 for i in range(1,10)])

            def forecast(self, timeseries, prediction_length, output_type, batch_size):
                assert prediction_length == 1 and output_type == "numpy"
                assert batch_size == len(timeseries)
                outputs = []
                for series in timeseries:
                    assert series.future_covariates.shape[-1] == series.target.shape[-1]+1
                    outputs.append(np.repeat(series.target[:, -1:].numpy()[:,None,:], 9, axis=1))
                return outputs

        result = forecast_batch(ModelDouble(), [item])[0]
        self.assertEqual(result.shape, (2,3))
        np.testing.assert_allclose(result[:,1], price.iloc[150].to_numpy(), rtol=1e-6)

    def test_forecast_scoring_against_persistence(self):
        rows = pd.DataFrame({"actual": [1.0002, np.nan], "current": [1.0,1.0],
                             "q10": [1.0,1.0], "q50": [1.0001,1.0001], "q90": [1.0003,1.0003]})
        score = forecast_metrics(rows, "EURUSD")
        self.assertEqual(score["forecasts"], 1)
        self.assertEqual(score["unscored_missing_outcomes"], 1)
        self.assertAlmostEqual(score["median_mae_pips"], 1)
        self.assertAlmostEqual(score["mae_ratio_to_persistence"], 0.5)
        self.assertEqual(score["central_80pct_coverage"], 1)

    def test_cost_filter_and_jpy_pip_scale(self):
        c = Config()
        self.assertEqual(trade_signal(1.1,1.1,0.0001,"EURUSD",c), 0)
        self.assertEqual(trade_signal(1.1,1.101,0.0001,"EURUSD",c), 1)
        self.assertEqual(trade_signal(1.1,1.099,0.0001,"EURUSD",c), -1)
        self.assertEqual(trade_signal(150,150.001,0.01,"USDJPY",c), 0)
        self.assertEqual(trade_signal(1.1,1.101,0.01,"EURUSD",c), 0)

    def test_insufficient_context_and_zero_scored_forecasts_fail(self):
        price, spread = panel()
        with self.assertRaises(ValueError):
            make_inputs(price, spread, price.index[10], 128)
        with self.assertRaises(ValueError):
            make_inputs(price, spread, price.index[150], 16)
        rows = pd.DataFrame(columns=["actual", "current", "q10", "q50", "q90"])
        with self.assertRaises(ValueError):
            forecast_metrics(rows, "EURUSD")


if __name__ == "__main__":
    unittest.main()
