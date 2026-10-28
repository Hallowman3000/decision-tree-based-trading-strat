# Initial forex evaluation

Source: *Decision Trees for Intuitive Intraday Trading Strategies*, arXiv:2405.13959v1.
Research adaptation, not a reproduction of its equity results.

Evaluation: January 1, 2023 through October 22, 2026. Sixteen quarterly refits per
pair, rolling 730-day training window, purged training labels, and fixed parameters
described in [README.md](README.md). No pair was selected or discarded after seeing
these results. The supplied data's vendor, timezone and executable-price status
have not been independently verified.

| Pair | Trades | Net return | Max drawdown | Win rate | Profit factor | Higher-cost return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EURUSD | 195 | -2.25% | -4.86% | 41.54% | 0.84 | -4.94% |
| GBPUSD | 57 | -0.70% | -1.99% | 50.88% | 0.89 | -1.45% |
| USDCAD | 106 | +1.32% | -2.14% | 54.72% | 1.23 | -0.05% |
| USDCHF | 24 | -0.09% | -0.70% | 37.50% | 0.93 | -0.48% |
| USDJPY | 165 | +0.88% | -2.41% | 52.73% | 1.08 | -1.21% |

These are cumulative returns across the full evaluation, not annual returns.
Each pair uses an independent account in its quote currency and 1× equity
notional per trade. Baseline costs: recorded bid/ask spread, 0.1 pip adverse
slippage per fill and 0.35 bps commission per side. The higher-cost column uses
the **same learned signals**, with 0.5 pip slippage per fill and 0.7 bps commission
per side; models were not retrained for this comparison. No observed trade had
a delayed exit caused by a missing quote.

The equal-weight index of normalized quote-currency accounts returned **-0.17%**,
with -1.27% maximum bar-close drawdown. This is not a USD portfolio: account-currency
conversion, interest and funding are absent. Cash's assumed zero return beats this
index. All five pairs lose under the higher-cost assumptions. This implementation
does **not** establish a robust profitable forex edge and should remain a research
baseline, rather than being connected to a live trading account.

## Validation and reproduction

Eight unit tests pass, covering feature causality, train/test target purging,
flat targets after costs, exact long/short costs, next-open fills, missing-bar
handling, nonoverlapping positions and cost sensitivity. The full functional
backtest completed on all five pairs. All pinned direct dependencies were present.

```bash
python -m unittest -v
python forex_strategy.py --output artifacts/baseline
```

The full machine-readable report, 80 readable trees, trade logs, signals and equity
curves are in `artifacts/baseline/` (ignored by Git). To reproduce the fixed-signal
execution-cost stress after the baseline run:

```python
from dataclasses import replace
from pathlib import Path
import json
import pandas as pd
from forex_strategy import Config, backtest, load_quotes, metrics

base = Path("artifacts/baseline")
report = json.loads((base / "report.json").read_text())
costs = replace(Config(**report["config"]), slippage_pips=0.5, commission_bps=0.7)
for pair in report["pairs"]:
    quotes = load_quotes(Path("data"), pair)
    signals = pd.read_csv(base / pair / "signals.csv", index_col=0,
                          parse_dates=True).iloc[:, 0]
    equity, trades = backtest(quotes, signals, pair, costs,
                              "2023-01-01", "2026-10-23")
    print(pair, metrics(equity, trades))
```

Before further evaluation, confirm data provenance, timezone, volume semantics
and broker fees. Freeze any revised strategy before testing newly reserved data;
these results must not be treated as an untouched holdout after further tuning.
