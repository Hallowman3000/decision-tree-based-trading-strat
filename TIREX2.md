# TiRex-2 forex forecasting experiment

Paper: Patrick Podest et al., *TiRex-2: Generalizing TiRex to Multivariate Data
and Streaming*, [arXiv:2607.01204v1](https://arxiv.org/abs/2607.01204).
Official [implementation](https://github.com/NX-AI/tirex-2) and
[checkpoint](https://huggingface.co/NX-AI/TiRex-2).

This is a separate experiment on branch `tirex-2-forex`. The previous depth-four
decision-tree strategy remains available on `depth-4-decision-tree`. This branch
starts from main after that strategy's merge and reuses its quote loader and
execution simulator; the forecasting model itself is the official TiRex-2.

## Current validation status

- CPU PyTorch 2.8.0 and official `tirex-2==0.4.0` installed successfully.
- All 16 unit tests passed, including eight existing execution tests and eight
  new forecast-pipeline tests. The API adapter test uses a test double and the
  installed official `TimeseriesType`; it is not a pretrained-model inference test.
- `pip check` found no broken requirements.
- Data preparation completed on all five forex pairs: 144 hourly forecast origins
  over October 1–22, 2026, representing up to 720 pair-level predictions once
  inference is available. Inputs: 5 targets × 128 hourly positions, 5 past-spread
  covariates × 128 positions, 4 calendar covariates × 129 positions.
- **Pretrained inference is blocked:** both the public Hugging Face metadata
  request and the real runner fail at checkpoint loading with proxy `403 Forbidden`.
  No TiRex-2 forecast-accuracy, trading-return or comparison results exist yet.
  Local artifacts contain `preparation.json` and `run_status.json` with the failure
  status; they are ignored by Git.

The environment network draft adds `huggingface.co` and
`cas-bridge.xethub.hf.co` to the existing package-manager network preset.
Saving a draft does not activate the network change. Review and save it in
environment settings, then publish the environment. Recheck the download after
the change is active. Public checkpoint access does not require inventing a token.
Additional download hosts, if requested by an observed redirect, may need to be
allowed separately. The runner defaults to the Hub's supported HTTPS download
path with `HF_HUB_DISABLE_XET=1`, avoiding the separate Xet CAS client API.
An alternative is an authorized local checkpoint directory
containing the official `config.json` and `model.safetensors`.

## Install and run

Use the existing checkout; no Git worktree is needed. The CPU configuration is
appropriate for this small initial experiment; foundation-model pretraining is
not attempted. Run from the repository root with Python 3.12:

```bash
python -m venv --system-site-packages .venv/tirex2
.venv/tirex2/bin/python -m pip install --cache-dir /tmp/tirex-pip-cache \
    'torch==2.8.0+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv/tirex2/bin/python -m pip install --cache-dir /tmp/tirex-pip-cache \
    -r requirements-tirex2.txt
.venv/tirex2/bin/python -m unittest -v
.venv/tirex2/bin/python tirex2_forex.py --prepare-only
```

After model access is available:

```bash
.venv/tirex2/bin/python tirex2_forex.py
```

For an already downloaded official checkpoint:

```bash
.venv/tirex2/bin/python tirex2_forex.py --model /path/to/TiRex-2
```

The default output is `artifacts/tirex2/`. Once results exist, use a new `--output`
directory for a rerun to avoid confusing old and new results. `--pairs EURUSD`
provides a smaller univariate smoke experiment, not an equivalent multivariate
comparison. `--context`, `--batch-size`, `--threads`, `--start` and `--end` are
configurable; defaults are fixed before seeing any model results. The default
checkpoint reference is resolved to a full Hugging Face commit SHA and recorded,
along with SHA-256 hashes of the configuration and safetensors weights. Use
`--revision <recorded-sha>` to reproduce that snapshot. Checkpoint/cache files
remain under ignored `.venv/`; weights are not committed to Git.

## Method and forex adaptation

The paper's method is a pretrained recurrent xLSTM forecasting model with variate
mixing and probabilistic quantile outputs, not a technical-indicator classifier.
This experiment uses its public multivariate API without training on forex data.
The package handles model-specific normalization and inference. The open-source
release does not expose the incremental streaming feature advertised for TiRex-2
PRO, so rolling history is recomputed here; no constant-time streaming claim is made.

1. Read paired bid/ask M5 files for EURUSD, GBPUSD, USDCAD, USDCHF and USDJPY.
   Preserve the existing loader's integrity checks. Provisionally assume timestamps
   are UTC bar-start labels, pending vendor confirmation.
2. Aggregate completed M5 bars into hourly midpoint closes. Require all 12 bars
   for an observed hour; otherwise represent the hour as missing. Keep a regular
   hourly calendar, including missing weekend hours. Do not interpolate or fill
   prices. The model receives missing observations as NaNs.
3. Supply log-price histories for all five targets jointly, with 128 wall-clock
   hourly positions and at least 32 observed points per target. Supply historical
   mean relative spreads as five past-only covariates. Only hour-of-day and
   day-of-week sine/cosine values are provided beyond the forecast origin.
   Future prices, volumes and realized spreads never enter model inputs.
4. Forecast the next hourly close at fixed hourly origins. Origins are admitted
   from known history and clocks only, not the availability of future outcomes.
   A row labelled 06:55 includes the last M5 close of the 06:00 hour and becomes
   available at 07:00. The next hourly target is labelled 07:55. Calendar filters
   match the existing 07:00–17:00 UTC weekday execution policy. The last admitted
   origin is 14:55, since exits must be strictly before 17:00.
5. Compare the forecast median to persistence: the next price equals the current
   completed hourly close. Report MAE and RMSE in pips, MAE relative to persistence
   (less than one is better), direction accuracy excluding realized zero moves,
   nominal 80% interval coverage and width, quantile crossings, and pinball loss
   averaged over the 10th, 50th and 90th percentiles. Missing outcomes are excluded
   from scoring and counted, not silently filled.
6. Separately evaluate a provisional trading rule: forecast median movement must
   exceed estimated round-trip costs plus one pip. The estimate uses only the
   last known spread, 0.1 pip adverse slippage per fill and 0.35 bps commission per
   side. Actual fills use subsequent bid/ask prices through the original backtest
   engine, with one-hour holds and 1× equity notional in independent quote-currency
   accounts. Missing exit quotes delay liquidation rather than canceling entries
   with hindsight. No stop loss or live orders are added.

## Results to inspect after inference

`report.json` will contain separate forecast and provisional trading metrics per
pair. `forecasts.csv` will contain origins, availability timestamps, realized
outcomes and the three forecast quantiles. Per-pair equity and trade CSVs will
show actual modeled fills. A successful report is generated only after checkpoint
inference, scoring and execution evaluation complete. Read `run_status.json`;
an input-preparation success is not a model result.

First establish whether the method beats persistence on price error and whether
its uncertainty intervals are calibrated. Neither condition guarantees profitable
trading. Keep the trading rule fixed for this first run rather than tuning it on
the same output. Longer comparisons and an independent evaluation period should
follow only after a working small experiment.

## Limits of this experiment

- October 2026 is a short, convenient first evaluation window, not a sufficient
  trading track record. Data overlap with the foundation model's pretraining
  cannot be ruled out; “zero-shot” does not establish a contamination-free test.
- Vendor provenance, timezone and tradable-price status remain unverified.
- Next-open fills assume zero inference latency. An hourly close becomes known
  at the next M5 open; real inference and submission take time. Live execution
  needs delayed-fill tests before the trading results can be interpreted as deployable.
- The forecast target is the next hourly close; the trade exits at the subsequent
  M5 open, so their prices can differ. They are measured separately.
- There is no unified USD funding/conversion model, consolidated currency-risk
  control, financing model, broker-lot handling, news-calendar filter or stop loss.
- The paper's general benchmark results do not establish an edge on forex.
