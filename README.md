# ai-quant-backtests

Each episode of the channel takes one popular trading idea and does three things:
- writes it as precise rules (`specs/`);
- lets an AI coding assistant build the backtest;
- runs fixed checks before any number is shown.

**Teaching examples only, not investment advice. Historical backtests do not predict the future.**

## Run it yourself
This repo has no market data, so bring your own daily data from a source you are allowed to use.

Put CSV files in one folder:

| File | Columns | Notes |
|---|---|---|
| `SPY.csv` | `date,open,high,low,close,volume` | Prices adjusted for splits and dividends. An optional `unadjusted_close` column adds the price-only comparison in episode 1. |
| `rate.csv` | `date,rate_pct` | Optional. 3-month Treasury yield, where 4.17 means 4.17% a year. Without it, cash earns 0% and a warning is shown. |

Then run an episode:

```bash
pip install pandas numpy matplotlib pytest
DATA_DIR=/path/to/your/csv_folder python run_ep01.py
```

Your numbers will be close to the video's but not identical, because data sources adjust prices differently and may cover different dates. Running a script overwrites `results/epNN/`, so `git diff` shows exactly what changed.

Change a parameter in `run_epNN.py` or `strategies/` and run it again to see whether the conclusion holds.

## Published results
`results/epNN/` holds the summaries behind each episode: `results.json` has every number quoted in the video, and episode 1 adds yearly returns, trades, the cost sweep and the parameter grid. Daily equity curves are not published, because they are close to the price series itself.

## Layout
- `backtest/data.py`: loading and data-quality checks.
- `backtest/engine.py`: two independent engines (vectorized, and a day-by-day event loop) plus the deliberate "no shift" bug used for demonstration.
- `backtest/metrics.py`: CAGR, volatility, Sharpe, drawdown, longest underwater period, trades.
- `backtest/checks.py`: the two mandatory checks, look-ahead and reconcile.
- `backtest/research.py`: the three-part health check: costs, parameters, before/after split.
- `strategies/`: one file per episode.
- `run_epNN.py`: writes every number in the episode to `results/epNN/results.json`.

## Timing convention
A signal is computed after the close of day t and executed at the open of day t+1. The overnight move belongs to the old position. Cost is |weight change| × equity × bp. Cash earns the 3-month Treasury rate.

## Tests
```bash
python -m pytest -q
```
