# Decision-tree forex strategy

A research implementation inspired by *Decision Trees for Intuitive Intraday
Trading Strategies*, Naga Prajwal et al., [arXiv:2405.13959v1](https://arxiv.org/abs/2405.13959).
The paper supplies the modeling idea, not evidence that this forex adaptation is profitable.
This project generates interpretable rules and backtests; it does not place live orders.

## Run

Python 3.12 was used for validation. From the existing repository checkout:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest -v
.venv/bin/python forex_strategy.py
```

For a single pair or a cost stress test:

```bash
.venv/bin/python forex_strategy.py --pairs EURUSD --output artifacts/eurusd
.venv/bin/python forex_strategy.py --slippage-pips 0.5 --commission-bps 0.7 --output artifacts/stress
```

Defaults evaluate 2023-01-01 through 2026-10-22 inclusive on the supplied files.
`--start` and `--end` change this interval; end is exclusive. Output includes
`report.json`, per-pair trades, bar-level equity, signals, and readable tree rules
for every quarterly refit. Outputs are ignored by Git. No external API or secret is needed.

## Strategy rules

1. Use the five actual currency pairs: EURUSD, GBPUSD, USDCAD, USDCHF and USDJPY.
   Metals and oil are excluded. Join bid and ask on identical timestamps; reject
   misalignment, crossed execution quotes, invalid prices and duplicate timestamps.
   Treat the timestamp as **UTC provisionally**; confirm this with the data vendor
   before interpreting session results. No missing price interpolation is performed.
2. Calculate nine inputs on completed five-minute midpoint bars: one-bar and
   15-bar returns, Wilder-style exponentially smoothed RSI(14) and ADX(14),
   SMA(14)/close, rolling 14-bar SMA/close correlation, 14-bar return standard
   deviation, rolling 14-observation standard deviation of 15-bar returns,
   and rolling 14-bar typical-price VWAP/close. EWM initialization is pandas'
   recursive convention, not a seeded arithmetic Wilder average. The paper's
   “210-period volatility” description is ambiguous; this implements its stated
   14-period volatility of overlapping 15-period returns, not 210 raw bars.
   Bid/ask midpoint highs and lows are feature proxies, not executable quotes.
3. Fit a separate Gini decision tree per pair, maximum depth four, minimum
   500 observations per leaf, fixed random seed. No feature scaling is needed.
   Every calendar quarter, fit only on the preceding 730 calendar days;
   exclude training observations whose target exit reaches the refit boundary.
4. Training targets use entry at the next bar's open and exit 12 bars later
   (60 minutes). Label **long** if that trade's after-cost profit exceeds one pip;
   label **short** if the short trade does; otherwise label **flat**. Costs include
   observed bid/ask spreads, 0.1 pip adverse slippage per fill and 0.35 basis
   points commission per side (0.7 bps round trip). These are adjustable assumptions,
   not verified broker fees. JPY pip size is 0.01; other pairs use 0.0001.
5. At a completed bar, act only if the most probable class has probability
   at least 55% and is long or short. Tree probabilities are uncalibrated leaf
   frequencies, not promised win rates. Buy at next ask open plus slippage;
   sell short at next bid open minus slippage. Exit at the opposite side of the
   spread with adverse slippage. Ignore further signals while a trade is open.
6. Admit only Monday–Friday entries at or after 07:00 UTC whose scheduled exit
   is before 17:00 UTC on the same date. Omit incomplete training targets.
   Entry orders expire if their next five-minute quote is missing. A missing exit
   quote delays liquidation until the next available open; it does not cause
   a past entry to be canceled. Such delays are counted and can breach the
   intended session or overnight policy. Scheduled exits precede quarterly/evaluation boundaries;
   a data outage preventing liquidation before evaluation end fails the run.
   The fixed UTC session is not adjusted for London/New York daylight saving.
7. Size notional at 1× current equity **per independent pair account**, in that
   pair's quote currency. Do not pyramid or average down. There is no stop loss;
   the scheduled exit bounds holding time, not maximum loss. Equity is marked
   each bar to executable close-side liquidation value including estimated exit
   costs. Intrabar excursions are not measured. Do not sum these accounts as
   five full-size positions in one account: real portfolio deployment requires
   currency conversion, aggregate exposure limits and broker margin controls.

## What changes from the paper

The paper studies one-minute Indian equities, next-return binary labels, and
cost-free backtests. Here, the available M5 forex quotes motivate a fixed one-hour
holding horizon, short selling, a neutral class and explicit trading costs. Depth
four and the nine feature families are retained; the confidence, leaf-size,
session, horizon, costs and edge thresholds are fixed research defaults, not
parameters optimized for these results. Volume is an unverified broker-activity
proxy: spot forex has no consolidated exchange volume. A VWAP ablation should
be tested on separately reserved data if volume provenance cannot be established.

All reported evaluation predictions are chronological walk-forward predictions.
Earlier evaluation periods can enter later training windows after their outcomes
are known, as in scheduled live retraining. There is no untouched final holdout
reserved by this script. Do not repeatedly adjust defaults on these results and
continue calling the same period independent evidence. Freeze changes and obtain
new unseen data for final evaluation and paper trading.

## Reading results and limitations

Returns are after modeled spread, commission and slippage. The zero-return cash
baseline is implicit. Sharpe uses business-day equity returns, 252 days/year,
and a zero risk-free rate; it is not the paper's Indian intraday annualization.
Maximum drawdown uses bar-close liquidation equity. Profit factor sums
per-trade account return fractions, not dollar P&L. No-trade runs explicitly
report zero trades, with undefined win rate/profit factor as null.

`normalized_account_index` equally weights independently compounded quote-currency
accounts at inception; it is **not a USD portfolio return**. No FX conversion,
cash interest, financing, latency, variable liquidity, economic-news calendar,
holiday calendar or broker-specific execution constraints are modeled. The
intraday policy avoids scheduled overnight holdings but is not a financing model.
CSV provenance, timestamp convention and whether quotes represent tradable prices
remain unverified. Results cannot establish live executability or profitability.

The included unit tests check feature causality, label purging, next-open fills,
exact long/short transaction costs, session/gap exclusions, nonoverlap and cost
sensitivity. Run the full historical backtest as a separate functional check.
