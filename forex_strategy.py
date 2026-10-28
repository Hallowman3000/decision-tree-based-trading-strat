"""Paper-inspired, cost-aware forex decision trees; research/backtesting only."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

PAIRS = ("EURUSD", "GBPUSD", "USDCAD", "USDCHF", "USDJPY")
FEATURES = ["return_1", "return_15", "rsi_14", "adx_14", "sma_close",
            "sma_correlation", "volatility_14", "volatility_15_14", "vwap_close"]


@dataclass(frozen=True)
class Config:
    horizon: int = 12
    depth: int = 4
    min_leaf: int = 500
    confidence: float = 0.55
    min_edge_pips: float = 1.0
    slippage_pips: float = 0.1
    commission_bps: float = 0.35  # per side; 0.7 bps round trip
    session_start: int = 7
    session_end: int = 17
    train_days: int = 730
    min_train_rows: int = 5000
    notional_fraction: float = 1.0

    def __post_init__(self):
        if self.horizon < 1 or self.depth < 1 or self.min_leaf < 1:
            raise ValueError("Horizon, depth and min_leaf must be positive")
        if not 0.5 <= self.confidence <= 1 or not 0 < self.notional_fraction <= 1:
            raise ValueError("Confidence must be in [0.5, 1], notional fraction in (0, 1]")
        if not 0 <= self.session_start < self.session_end <= 24:
            raise ValueError("Invalid UTC session")
        if min(self.min_edge_pips, self.slippage_pips, self.commission_bps) < 0:
            raise ValueError("Trading costs and minimum edge must be nonnegative")
        if self.train_days < 1 or self.min_train_rows < 1:
            raise ValueError("Training window and minimum rows must be positive")


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


def load_quotes(data_dir: Path, pair: str) -> pd.DataFrame:
    if pair not in PAIRS:
        raise ValueError(f"Unsupported forex pair: {pair}")
    sides = []
    for side in ("BID", "ASK"):
        frame = pd.read_csv(data_dir / f"{pair}_M5_{side}_5years.csv", parse_dates=["time"])
        if list(frame.columns) != ["time", "open", "high", "low", "close", "volume"]:
            raise ValueError("Unexpected CSV schema")
        frame = frame.set_index("time")
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError("Quotes must have unique, increasing timestamps")
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (frame.drop(columns="volume") <= 0).any().any():
            raise ValueError("Nonfinite values or nonpositive prices")
        if ((frame.low > frame[["open", "close"]].min(axis=1)) |
                (frame.high < frame[["open", "close"]].max(axis=1)) |
                (frame.volume < 0)).any():
            raise ValueError("Invalid OHLC or volume")
        sides.append(frame)
    bid, ask = sides
    if not bid.index.equals(ask.index):
        raise ValueError("Bid/ask timestamps differ; repair the source rather than interpolate")
    if ((ask.open < bid.open) | (ask.close < bid.close)).any():
        raise ValueError("Crossed bid/ask quotes")
    quotes = bid.add_prefix("bid_").join(ask.add_prefix("ask_"))
    for col in ("open", "high", "low", "close", "volume"):
        quotes[col] = (bid[col] + ask[col]) / 2
    return quotes


def features(q: pd.DataFrame) -> pd.DataFrame:
    """All features at t use bars <= t. No filling across gaps or future interpolation."""
    close = q.close
    ret = close.pct_change(fill_method=None)
    ret15 = close.pct_change(15, fill_method=None)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100).mask((gain == 0) & (loss == 0), 50)
    tr = pd.concat([q.high - q.low, (q.high-close.shift()).abs(),
                    (q.low-close.shift()).abs()], axis=1).max(axis=1)
    up, down = q.high.diff(), -q.low.diff()
    plus = up.where((up > down) & (up > 0), 0.0)
    minus = down.where((down > up) & (down > 0), 0.0)
    smooth = lambda s: s.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    atr = smooth(tr)
    pdi, mdi = 100*smooth(plus)/atr, 100*smooth(minus)/atr
    denominator = pdi + mdi
    dx = (100*(pdi-mdi).abs()/denominator.replace(0, np.nan)).mask(denominator == 0, 0)
    sma = close.rolling(14).mean()
    typical = (q.high + q.low + close) / 3
    vwap = (typical*q.volume).rolling(14).sum()/q.volume.rolling(14).sum().replace(0, np.nan)
    return pd.DataFrame({
        "return_1": ret, "return_15": ret15, "rsi_14": rsi, "adx_14": smooth(dx),
        "sma_close": sma/close, "sma_correlation": sma.rolling(14).corr(close),
        "volatility_14": ret.rolling(14).std(),
        "volatility_15_14": ret15.rolling(14).std(), "vwap_close": vwap/close,
    }).replace([np.inf, -np.inf], np.nan)


def labels(q: pd.DataFrame, pair: str, c: Config) -> pd.DataFrame:
    """Signal at close t, entry open t+1, exit open t+1+horizon."""
    pip = pip_size(pair)
    slip = pip*c.slippage_pips
    fee = c.commission_bps/10000
    entry_bid, entry_ask = q.bid_open.shift(-1), q.ask_open.shift(-1)
    exit_bid, exit_ask = q.bid_open.shift(-c.horizon-1), q.ask_open.shift(-c.horizon-1)
    mid = (entry_bid + entry_ask)/2
    long_net = (exit_bid-slip - (entry_ask+slip)
                - fee*(entry_ask+slip+exit_bid-slip))/mid
    short_net = (entry_bid-slip - (exit_ask+slip)
                 - fee*(entry_bid-slip+exit_ask+slip))/mid
    target = pd.Series(0, index=q.index, dtype=int)
    target = target.mask(long_net > c.min_edge_pips*pip/mid, 1)
    target = target.mask(short_net > c.min_edge_pips*pip/mid, -1)
    times = pd.Series(q.index, index=q.index)
    entry_time, end_time = times.shift(-1), times.shift(-c.horizon-1)
    complete = end_time-times == pd.Timedelta(minutes=5*(c.horizon+1))
    # Entry eligibility uses the clock only, never the availability of future quotes.
    planned_entry = times + pd.Timedelta(minutes=5)
    planned_exit = planned_entry + pd.Timedelta(minutes=5*c.horizon)
    eligible = ((planned_entry.dt.normalize() == planned_exit.dt.normalize()) &
                (planned_entry.dt.hour >= c.session_start) &
                (planned_exit.dt.hour < c.session_end) & (planned_entry.dt.dayofweek < 5))
    return pd.DataFrame({"target": target, "eligible": eligible,
                         "complete_target": complete, "label_end": planned_exit})


def walk_forward(q, x, y, c, start, end, output_dir):
    """Quarterly refits; purge any label whose exit reaches the test boundary."""
    predictions = pd.Series(0, index=q.index, dtype=int)
    folds = []
    cursor = pd.Timestamp(start)
    end = pd.Timestamp(end)
    while cursor < end:
        stop = min(cursor + pd.offsets.QuarterBegin(startingMonth=1), end)
        train = ((x.index >= cursor-pd.Timedelta(days=c.train_days)) &
                 (x.index < cursor) & (y.label_end < cursor) & y.eligible &
                 y.complete_target & x.notna().all(axis=1))
        test = ((x.index >= cursor) & (x.index < stop) & y.eligible &
                (y.label_end < stop) & x.notna().all(axis=1))
        if train.sum() < c.min_train_rows:
            raise ValueError(f"Insufficient training rows before {cursor}: {train.sum()}")
        model = DecisionTreeClassifier(criterion="gini", max_depth=c.depth,
                                       min_samples_leaf=c.min_leaf, random_state=42)
        model.fit(x.loc[train, FEATURES], y.loc[train, "target"])
        if test.any():
            proba = model.predict_proba(x.loc[test, FEATURES])
            best = proba.argmax(axis=1)
            signals = np.where(proba.max(axis=1) >= c.confidence, model.classes_[best], 0)
            predictions.loc[test] = signals
        fold = {"test_start": str(cursor), "test_end_exclusive": str(stop),
                "train_rows": int(train.sum()), "test_rows": int(test.sum()),
                "last_training_label_exit": str(y.loc[train, "label_end"].max()),
                "class_counts": {str(k): int(v) for k,v in y.loc[train,"target"].value_counts().items()}}
        folds.append(fold)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir/f"tree_{cursor.date()}.txt").write_text(export_text(model, feature_names=FEATURES, decimals=6))
        cursor = stop
    return predictions, folds


def backtest(q, signals, pair, c, start, end):
    """One trade at a time, mark to liquidation value each bar, quote-currency equity."""
    slip = pip_size(pair)*c.slippage_pips
    fee = c.commission_bps/10000
    eligible = labels(q, pair, c).eligible.to_numpy()
    sig = signals.reindex(q.index).fillna(0).to_numpy()
    bo, ao = q.bid_open.to_numpy(), q.ask_open.to_numpy()
    bc, ac = q.bid_close.to_numpy(), q.ask_close.to_numpy()
    times = q.index
    cash, units, exit_i = 1.0, 0.0, -1
    entry_equity = entry_price = 0.0
    entry_time = planned_exit = None
    trades, curve = [], []
    for i in np.flatnonzero((times >= pd.Timestamp(start)) & (times < pd.Timestamp(end))):
        if units and i == exit_i:
            price = bo[i]-slip if units > 0 else ao[i]+slip
            cash += units*price - abs(units)*price*fee
            trades.append({"entry_time": entry_time, "exit_time": times[i],
                           "side": "long" if units > 0 else "short", "entry_price": entry_price,
                           "exit_price": price, "return": cash/entry_equity-1,
                           "delayed_exit": bool(times[i] > planned_exit)})
            units = 0.0
        # A signal from the just-completed previous bar can enter at this open.
        if (not units and i > 0 and sig[i-1] and eligible[i-1] and
                times[i]-times[i-1] == pd.Timedelta(minutes=5) and
                times[i-1] >= pd.Timestamp(start) and
                times[i]+pd.Timedelta(minutes=5*c.horizon) < pd.Timestamp(end)):
            entry_equity = cash
            units = int(sig[i-1])*cash*c.notional_fraction/((bo[i]+ao[i])/2)
            entry_price = ao[i]+slip if units > 0 else bo[i]-slip
            cash -= units*entry_price + abs(units)*entry_price*fee
            entry_time = times[i]
            planned_exit = entry_time+pd.Timedelta(minutes=5*c.horizon)
            # A missing scheduled exit quote delays execution; never cancel an
            # entry using hindsight about whether its future bars are present.
            exit_i = int(times.searchsorted(planned_exit))
        liquidation = bc[i]-slip if units > 0 else ac[i]+slip
        equity = cash + units*liquidation - abs(units)*liquidation*fee
        if equity <= 0:
            raise ValueError("Account depleted; stop the simulation")
        curve.append((times[i], equity))
    if units:
        raise AssertionError("Unclosed trade at backtest end")
    return pd.Series(dict(curve), name="equity").rename_axis("time"), pd.DataFrame(trades)


def metrics(equity, trades=None):
    if equity.empty:
        raise ValueError("Empty evaluation period")
    daily = equity.resample("B").last().ffill().dropna()
    daily_returns = daily.pct_change().dropna()
    if len(daily):
        daily_returns = pd.concat([pd.Series([daily.iloc[0]-1], index=daily.index[:1]), daily_returns])
    years = (equity.index[-1]-equity.index[0]).total_seconds()/(365.25*86400)
    result = {"total_return": float(equity.iloc[-1]-1),
              "cagr": float(equity.iloc[-1]**(1/years)-1) if years > 0 else None,
              "max_drawdown": float((equity/equity.cummax().clip(lower=1)-1).min()),
              "daily_sharpe_zero_rf": float(daily_returns.mean()/daily_returns.std()*np.sqrt(252))
              if daily_returns.std() > 0 else None,
              "start": str(equity.index[0]), "end": str(equity.index[-1])}
    if trades is not None:
        r = trades["return"] if len(trades) else pd.Series(dtype=float)
        losses = -r[r < 0].sum()
        result.update(trades=len(r), win_rate=float((r > 0).mean()) if len(r) else None,
                      delayed_exits=int(trades.delayed_exit.sum()) if len(r) else 0,
                      profit_factor=float(r[r > 0].sum()/losses) if losses > 0 else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/forex"))
    parser.add_argument("--pairs", nargs="+", choices=PAIRS, default=list(PAIRS))
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-10-23", help="Exclusive evaluation end")
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--slippage-pips", type=float, default=0.1)
    parser.add_argument("--commission-bps", type=float, default=0.35)
    args = parser.parse_args()
    if pd.Timestamp(args.start) >= pd.Timestamp(args.end):
        parser.error("--start must precede --end")
    c = Config(confidence=args.confidence, slippage_pips=args.slippage_pips,
               commission_bps=args.commission_bps)
    args.output.mkdir(parents=True, exist_ok=True)
    reports, curves = {}, {}
    for pair in dict.fromkeys(args.pairs):
        print(f"Backtesting {pair}...", flush=True)
        q = load_quotes(args.data_dir, pair)
        if pd.Timestamp(args.end) > q.index[-1]+pd.Timedelta(minutes=5):
            raise ValueError("Requested end exceeds available data")
        x = features(q)
        y = labels(q, pair, c)
        signals, folds = walk_forward(q, x, y, c, args.start, args.end, args.output/pair)
        equity, trades = backtest(q, signals, pair, c, args.start, args.end)
        equity.to_csv(args.output/pair/"equity.csv")
        trades.to_csv(args.output/pair/"trades.csv", index=False)
        signals.loc[(signals.index >= args.start) & (signals.index < args.end)].to_csv(args.output/pair/"signals.csv", header=["signal"])
        reports[pair] = {"metrics": metrics(equity, trades), "folds": folds,
                         "equity_currency": pair[3:]}
        curves[pair] = equity
        print(json.dumps(reports[pair]["metrics"], indent=2), flush=True)
    # Separate local quote-currency accounts normalized to 1; this is an index,
    # not a USD-funded portfolio (conversion/funding data are not supplied).
    combined = pd.DataFrame(curves).ffill().fillna(1).mean(axis=1)
    combined.name = "equal_weight_normalized_equity"
    combined.to_csv(args.output/"normalized_equity.csv")
    report = {"source": "arXiv:2405.13959v1, adapted rather than reproduced",
              "config": asdict(c), "pairs": reports,
              "normalized_account_index": metrics(combined),
              "limitations": ["CSV provenance and timezone unverified; UTC assumed",
                              "Volume treated as broker activity, not consolidated FX volume",
                              "Quote-currency accounts, no USD conversion or funding model",
                              "No live orders, stop loss, latency model or liquidity model",
                              "Historical results do not establish future profitability"]}
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"Saved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
