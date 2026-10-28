"""TiRex-2 zero-shot hourly forex forecast experiment (arXiv:2607.01204v1).

The public pretrained checkpoint is required for real forecasts. --prepare-only
validates local data and inputs, without substituting a different forecasting model.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from forex_strategy import PAIRS, Config, backtest, load_quotes, metrics, pip_size


@dataclass
class Inputs:
    target: np.ndarray
    past_covariates: np.ndarray
    future_covariates: np.ndarray
    timestamps: pd.DatetimeIndex


def hourly_panel(quotes: dict[str, pd.DataFrame]):
    """Regular wall-clock grid. Incomplete hours are missing, never interpolated.

    Input M5 timestamps are provisionally interpreted as bar starts. An H1 row
    labelled 06:55 includes the final M5 close of 06:00--06:59, available at 07:00.
    """
    prices, spreads = {}, {}
    for pair, q in quotes.items():
        if (q.index.minute % 5 != 0).any() or (q.index.second != 0).any():
            raise ValueError("Expected five-minute grid timestamps")
        complete = q.close.resample("h").count().eq(12)
        prices[pair] = q.close.resample("h").last().where(complete)
        spreads[pair] = ((q.ask_close-q.bid_close)/q.close*10000).resample("h").mean().where(complete)
    price = pd.DataFrame(prices)
    spread = pd.DataFrame(spreads).reindex(price.index)
    price.index = price.index+pd.Timedelta(minutes=55)
    spread.index = price.index
    if not np.isfinite(price.to_numpy()).any():
        raise ValueError("No complete hourly bars")
    return price, spread


def calendar_covariates(index: pd.DatetimeIndex) -> np.ndarray:
    """Only deterministic clock values may be supplied beyond the origin."""
    hour = 2*np.pi*(index.hour+index.minute/60)/24
    weekday = 2*np.pi*index.dayofweek/7
    return np.asarray([np.sin(hour), np.cos(hour), np.sin(weekday), np.cos(weekday)], dtype=np.float32)


def make_inputs(price, spread, origin, context):
    if context < 32:
        raise ValueError("Use at least 32 hourly context positions")
    loc = price.index.get_loc(origin)
    if loc < context-1:
        raise ValueError("Insufficient history")
    history = price.iloc[loc-context+1:loc+1]
    if not history.notna().sum().ge(32).all() or not history.iloc[-1].notna().all():
        raise ValueError("Insufficient observed context or missing current price")
    stamps = history.index.append(pd.DatetimeIndex([origin+pd.Timedelta(hours=1)]))
    return Inputs(np.log(history.to_numpy(dtype=float)).T.astype(np.float32),
                  spread.iloc[loc-context+1:loc+1].to_numpy(dtype=np.float32).T,
                  calendar_covariates(stamps), stamps)


def choose_origins(price, start, end, context):
    """Choose origins using the current clock and history only, not future outcomes."""
    candidates = price.index[(price.index >= start) & (price.index < end)]
    origins, excluded = [], 0
    for origin in candidates:
        entry, exit_time = origin+pd.Timedelta(minutes=5), origin+pd.Timedelta(minutes=65)
        if not (entry.dayofweek < 5 and entry.hour >= 7 and exit_time.hour < 17 and
                entry.normalize() == exit_time.normalize() and exit_time < pd.Timestamp(end)):
            continue
        loc = price.index.get_loc(origin)
        history = price.iloc[max(0, loc-context+1):loc+1]
        if loc < context-1 or not history.notna().sum().ge(32).all() or not history.iloc[-1].notna().all():
            excluded += 1
            continue
        origins.append(origin)
    return origins, excluded


def forecast_batch(model, inputs: list[Inputs]):
    """Thin adapter to the official public API; no custom architecture replacement."""
    import torch
    from tirex2 import TimeseriesType
    series = [TimeseriesType(target=torch.from_numpy(item.target),
                             past_covariates=torch.from_numpy(item.past_covariates),
                             future_covariates=torch.from_numpy(item.future_covariates))
              for item in inputs]
    with torch.inference_mode():
        result = model.forecast(series, prediction_length=1, output_type="numpy", batch_size=len(series))
    levels = [round(float(q), 6) for q in model.quantiles]
    selected = [levels.index(level) for level in (0.1, 0.5, 0.9)]
    outputs = []
    if len(result) != len(inputs):
        raise ValueError("Unexpected number of forecasts")
    for item, forecast in zip(inputs, result):
        forecast = np.asarray(forecast)
        if forecast.shape != (item.target.shape[0], len(levels), 1):
            raise ValueError(f"Unexpected model output shape: {forecast.shape}")
        values = np.exp(forecast[:, selected, 0].astype(np.float64))
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Invalid price forecast")
        outputs.append(values)
    return outputs


def forecast_metrics(rows: pd.DataFrame, pair: str):
    valid = rows.dropna(subset=["actual"])
    if valid.empty:
        raise ValueError(f"No scored forecasts for {pair}")
    pip = pip_size(pair)
    error = (valid.q50-valid.actual)/pip
    naive_error = (valid.current-valid.actual)/pip
    mae, baseline = float(error.abs().mean()), float(naive_error.abs().mean())
    actual_move, predicted_move = valid.actual-valid.current, valid.q50-valid.current
    moving = actual_move.ne(0)
    lower, upper = valid.q10, valid.q90
    pinball = []
    for level, column in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
        residual = (valid.actual-valid[column])/pip
        pinball.append(np.maximum(level*residual, (level-1)*residual).mean())
    return {"forecasts": len(valid), "unscored_missing_outcomes": len(rows)-len(valid),
            "median_mae_pips": mae, "persistence_mae_pips": baseline,
            "mae_ratio_to_persistence": mae/baseline if baseline > 0 else None,
            "median_rmse_pips": float(np.sqrt((error**2).mean())),
            "direction_accuracy_excluding_actual_ties": float((np.sign(predicted_move[moving]) == np.sign(actual_move[moving])).mean()) if moving.any() else None,
            "direction_observations": int(moving.sum()),
            "central_80pct_coverage": float(((valid.actual >= lower) & (valid.actual <= upper)).mean()),
            "mean_80pct_width_pips": float(((upper-lower)/pip).mean()),
            "quantile_crossing_fraction": float(((lower > valid.q50) | (valid.q50 > upper)).mean()),
            "mean_pinball_loss_pips": float(np.mean(pinball))}


def trade_signal(current, median, known_spread, pair, c):
    """Estimate costs with the last known spread; actual fills use future bid/ask."""
    cost = known_spread + 2*c.slippage_pips*pip_size(pair) + 2*current*c.commission_bps/10000
    threshold = cost+c.min_edge_pips*pip_size(pair)
    move = median-current
    return 1 if move > threshold else (-1 if move < -threshold else 0)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def load_pretrained(model_path, revision, cache_dir):
    """Resolve an immutable snapshot and use safetensors only."""
    # Supported HTTPS download path; avoids requiring the separate Xet CAS API.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, snapshot_download
    from tirex2 import load_model
    local = Path(model_path)
    if local.is_dir():
        resolved, sha = local.resolve(), None
    else:
        sha = HfApi().model_info(model_path, revision=revision).sha
        resolved = Path(snapshot_download(model_path, revision=sha, cache_dir=cache_dir,
                                          allow_patterns=["config.json", "model.safetensors"]))
    for name in ("config.json", "model.safetensors"):
        if not (resolved/name).is_file():
            raise ValueError(f"Missing required checkpoint file: {name}")
    hashes = {}
    for name in ("config.json", "model.safetensors"):
        with (resolved/name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return load_model(resolved, device="cpu", use_flex_attention=False), {
        "source": model_path, "resolved_revision": sha, "sha256": hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/tirex2"))
    parser.add_argument("--start", default="2026-10-01")
    parser.add_argument("--end", default="2026-10-23", help="Exclusive end")
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--pairs", nargs="+", choices=PAIRS, default=list(PAIRS))
    parser.add_argument("--model", default="NX-AI/TiRex-2", help="Official HF ID or local checkpoint directory")
    parser.add_argument("--revision", default="main", help="Resolved to a full HF commit SHA before download")
    parser.add_argument("--cache-dir", type=Path, default=Path(".venv/hf-cache"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if pd.Timestamp(args.start) >= pd.Timestamp(args.end) or min(args.batch_size, args.threads) < 1:
        parser.error("Invalid evaluation dates, batch size or thread count")
    if any((args.output/name).exists() for name in ("report.json", "forecasts.csv")):
        parser.error("This output directory already contains results; use a fresh --output directory")
    pairs = list(dict.fromkeys(args.pairs))
    quotes = {pair: load_quotes(args.data_dir, pair) for pair in pairs}
    if any(pd.Timestamp(args.end) > q.index[-1]+pd.Timedelta(minutes=5) for q in quotes.values()):
        parser.error("Evaluation end exceeds available quotes")
    price, spread = hourly_panel(quotes)
    origins, excluded = choose_origins(price, args.start, args.end, args.context)
    if not origins:
        parser.error("No valid forecast origins")
    sample = make_inputs(price, spread, origins[0], args.context)
    args.output.mkdir(parents=True, exist_ok=True)
    preparation = {"status": "inputs_prepared_not_forecast", "pairs": pairs,
                   "start": args.start, "end_exclusive": args.end,
                   "forecast_origins": len(origins), "historical_context_exclusions": excluded,
                   "first_origin_bar": str(origins[0]), "last_origin_bar": str(origins[-1]),
                   "target_shape": list(sample.target.shape),
                   "past_covariates_shape": list(sample.past_covariates.shape),
                   "future_covariates_shape": list(sample.future_covariates.shape)}
    write_json(args.output/"preparation.json", preparation)
    print(json.dumps(preparation, indent=2), flush=True)
    if args.prepare_only:
        return
    import torch
    torch.set_num_threads(args.threads)
    try:
        model, checkpoint = load_pretrained(args.model, args.revision, args.cache_dir)
    except Exception as exc:
        write_json(args.output/"run_status.json", {"status": "checkpoint_load_failed",
                   "exception_type": type(exc).__name__,
                   "message": "Could not load the pretrained model. Check network access or provide --model /local/checkpoint. No forecast or trading results were generated by this attempt."})
        raise RuntimeError("Pretrained checkpoint unavailable; no model results. See run_status.json.") from exc
    started = time.perf_counter()
    rows = []
    for offset in range(0, len(origins), args.batch_size):
        batch_origins = origins[offset:offset+args.batch_size]
        inputs = [make_inputs(price, spread, origin, args.context) for origin in batch_origins]
        predictions = forecast_batch(model, inputs)
        for origin, prediction in zip(batch_origins, predictions):
            target_time = origin+pd.Timedelta(hours=1)
            for j, pair in enumerate(pairs):
                rows.append({"origin_bar": origin, "forecast_available_at": origin+pd.Timedelta(minutes=5),
                             "target_bar": target_time, "pair": pair, "current": price.at[origin, pair],
                             "actual": price.at[target_time, pair] if target_time in price.index else np.nan,
                             "q10": prediction[j,0], "q50": prediction[j,1], "q90": prediction[j,2]})
        print(f"Forecasted {min(offset+args.batch_size, len(origins))}/{len(origins)} origins", flush=True)
    forecasts = pd.DataFrame(rows)
    forecasts.to_csv(args.output/"forecasts.csv", index=False)
    c = Config()
    reports = {}
    for pair in pairs:
        subset = forecasts.loc[forecasts.pair == pair]
        q = quotes[pair]
        signals = pd.Series(0, index=q.index, dtype=int)
        for row in subset.itertuples():
            origin = row.origin_bar
            # Known quotes only: no realised future price or spread enters decisions.
            signals.at[origin] = trade_signal(row.current, row.q50,
                                              q.at[origin,"ask_close"]-q.at[origin,"bid_close"], pair, c)
        equity, trades = backtest(q, signals, pair, c, args.start, args.end)
        equity.to_csv(args.output/f"{pair}_equity.csv")
        trades.to_csv(args.output/f"{pair}_trades.csv", index=False)
        reports[pair] = {"forecast": forecast_metrics(subset, pair),
                         "provisional_trading_rule": metrics(equity, trades), "equity_currency": pair[3:]}
    write_json(args.output/"report.json", {
        "status": "pretrained_forecasts_evaluated", "paper": "arXiv:2607.01204v1",
        "checkpoint": checkpoint, "preparation": preparation, "pairs": reports,
        "inference_seconds": time.perf_counter()-started,
        "versions": {name: importlib.metadata.version(name) for name in ["torch", "tirex-2", "numpy", "pandas"]},
        "limitations": ["Short initial October 2026 experiment, not a profitability claim",
                        "Pretraining data overlap with forex history cannot be ruled out",
                        "Assumed UTC M5 bar-start timestamps; provenance unverified",
                        "Public release recomputes rolling context; PRO streaming is not used",
                        "Separate quote-currency accounts; no consolidated USD portfolio",
                        "Forecast inference latency is not charged to next-open fills"]})
    write_json(args.output/"run_status.json", {"status": "completed"})
    print(f"Completed; results in {args.output}", flush=True)


if __name__ == "__main__":
    main()
