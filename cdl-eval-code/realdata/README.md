# Benchmark of the foundation models

Seven pretrained foundation models applied without fine-tuning (Chronos, Chronos-Bolt, Chronos-2, TimesFM 2.5, Toto,
Sundial, Aurora) and a probabilistic seasonal naive baseline, on three held-out datasets and on 2026 AEMO data.

## Data

`python download_data.py` writes the data to `data/`:

| File | Source |
| --- | --- |
| `data/ercot.parquet`, `data/monash_traffic.parquet`, `data/monash_australian_electricity.parquet` | Hugging Face Hub, collection `autogluon/chronos_datasets` |
| `data/aemo_2026/PRICE_AND_DEMAND_<YYYYMM>_<REGION>.csv` | AEMO price-and-demand files, regions NSW1, QLD1, SA1, TAS1, VIC1, December 2025 to September 2026 |

Test windows are drawn with fixed random seeds in `run.prepare` (held-out data) and `bench.prepare` (AEMO data), so
every model is scored on the same windows: 300 per dataset at 24 steps and 150 at 192 and 720 steps.

## Protocol (`bench.py`)

Every model receives the last 512 context steps and is used with its default decoding settings. Sample-based models
draw 100 paths per window (three sampling seeds at 24 steps, one at longer horizons); quantile models are scored at
their native levels. Each window is scaled by the mean and standard deviation of its last week of context. For every
window the script stores the quantile loss, the coverage of the central 80% and 90% intervals, the CRPS, the energy
and variogram scores, the variogram score after shuffling the paths, the model's rank correlation of adjacent steps
and the PIT values, and it records the number of sequential network calls.

```bash
python bench.py --data heldout --horizon 24 --models naive bolt bolt_base chronos2 timesfm sundial aurora chronos
python bench.py --data aemo    --horizon 24 --models naive bolt bolt_base chronos2 timesfm sundial aurora chronos
python bench.py --data heldout --horizon 720 --n_test 150 --seeds 0 --models naive bolt chronos2 timesfm sundial aurora
python bench.py --data heldout --horizon 24 --models toto --tag toto      # in the Toto environment
```

Results are merged into `results/bench_<tag>.json` (one row per dataset, horizon, model and seed) and
`results/bench_<tag>_perwindow.npz` (per-window scores). `bench_report.py` reads all result files in name order; a
later file overrides rows of the same dataset, horizon, model and seed (for example `bench_main_fix` after
`bench_main`). `run_benchmark.sh` and `run_benchmark.ps1` run all settings of the article.

## Further checks

| Script | Check | Output |
| --- | --- | --- |
| `shuffle_check.py` | Shuffle test and rank measure on synthetic Gaussian forecasts with known dependence and spread | `results/shuffle_check.json`, Table S5 |
| `flow_determinism.py` | Sundial and Aurora return identical forecasts for the same random seed | printed |
| `sundial_official.py` | Sundial through its released `generate()` routine (`transformers` 4.40.1) | `results/sundial_official.json`, Table S6 |
| `aurora_context.py` | Aurora with 96 to 2,880 context steps | `results/aurora_context.json`, Table S7 |
| `run.py`, `run_fm.py`, `robust.py` | First-round runs, small forecasters trained on each dataset, Chronos without top-k truncation, other context lengths, unbiased estimators | `results/real*.json`, `results/robust*.json`, Table S12 and the trained rows of Table IV |

## Reports

```bash
python bench_report.py      # Table IV, Figs. 3-6, Tables S2-S4, S6, S8, S11
python rank_stats.py        # Table V, Tables S9-S10
python robust_report.py     # Tables S7, S12
```

Block-bootstrap intervals resample blocks of windows from the same series and week (5,000 resamples).
