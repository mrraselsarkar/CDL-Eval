# CDL-Eval

**A zero-shot framework for evaluating the calibration, dependence, and leakage of time series foundation models**

Code and results for the article


Pretrained time series foundation models forecast new series without training. Decisions such as unit commitment
and reserve sizing need more than accurate point forecasts: they need intervals that contain the stated share of
outcomes and sample paths whose steps move together as the data do. CDL-Eval checks these properties separately:

- **Accuracy and calibration**: quantile loss and CRPS at the levels that every model provides, and the coverage of
  the central 80% interval (from the quantiles, or from the randomized PIT for sample-based models).
- **Dependence**: a rank-based measure that compares the rank correlation of adjacent steps within the forecast paths
  (rho_model) with that of the observed PIT values (rho_obs). Unlike the shuffle test, it does not depend on the
  spread of the forecast.
- **Leakage**: the change in each model's standing, relative to a reference model, between an older public series
  and data from the same source published after every model.

The benchmark applies CDL-Eval to seven public foundation models, Chronos, Chronos-Bolt, Chronos-2, TimesFM 2.5, Toto,
Sundial and Aurora, and a probabilistic seasonal naive baseline. The data are electricity load in Texas (ERCOT), road
occupancy in San Francisco (Traffic), and electricity demand in five Australian states in 2002-2015 and in 2026 (AEMO).
Forecast horizons are 24, 192 and 720 steps. A controlled simulation with small trained forecasters isolates the
mechanisms.

## Repository structure

```
cdl-eval/
|-- figstyle.py                 shared style of the figures
|-- requirements.txt            main environment (all models except Toto, simulation, reports)
|-- requirements-toto.txt       environment for Toto
|-- requirements-sundial.txt    environment for Sundial's released generation routine
|-- realdata/                   benchmark of the foundation models
|   |-- download_data.py        downloads the four datasets into realdata/data/
|   |-- run.py                  data preparation and the first round of runs (imported by the other scripts)
|   |-- bench.py                the benchmark protocol (all models, horizons, scores, per-window results)
|   |-- run_benchmark.sh/.ps1   all benchmark runs in order
|   |-- shuffle_check.py        the shuffle test and the rank measure on synthetic forecasts
|   |-- flow_determinism.py     Sundial and Aurora are fixed by their initial noise
|   |-- sundial_official.py     Sundial through its released generation routine
|   |-- aurora_context.py       Aurora with context lengths from 96 to 2,880 steps
|   |-- robust.py, run_fm.py    robustness runs (decoding, context length, estimators)
|   |-- bench_report.py         tables and figures of the benchmark
|   |-- rank_stats.py           mean ranks, Friedman, Iman-Davenport, Nemenyi and Wilcoxon tests
|   |-- robust_report.py        tables of the robustness runs
|   `-- results/                stored results of every run (JSON summaries and per-window scores)
`-- simulation/                 controlled simulation
    |-- sim.py                  data, six forecasters, training and evaluation
    |-- calib.py                PIT histograms for the calibration figure
    |-- sim_robust.py, budget.py   training-budget runs
    |-- run_cpu_jobs.sh         detached CPU jobs for budget.py
    |-- plots.py                tables and figures of the simulation
    `-- results*/               stored results (model weights are not included)
```

## Reproduce the tables and figures from the stored results

The stored results in `realdata/results` and `simulation/results*` are enough to rebuild every table and figure of
the article in a few minutes, without a GPU and without the data. The report scripts need only `numpy`, `scipy`
and `matplotlib` (`shuffle_check.py` also needs `torch`).

```bash
pip install numpy scipy matplotlib torch
cd realdata
python bench_report.py      # Table IV, Figs. 3-6, Tables S2-S4, S6, S8, S11
python rank_stats.py        # Table V, Tables S9-S10
python robust_report.py     # Tables S7, S12
python shuffle_check.py     # Table S5
cd ../simulation
python plots.py             # Tables S13, S15, S16, Figs. S1-S2
```

Outputs are written to `outputs/tables` (LaTeX tables) and `outputs/figures` (PDF figures):

| Output | Script | In the article |
| --- | --- | --- |
| `tables/real_table.tex` | `bench_report.py` | Table IV |
| `tables/rank_table.tex` | `rank_stats.py` | Table V |
| `figures/accuracy.pdf`, `dependence.pdf`, `horizon.pdf`, `leakage.pdf` | `bench_report.py` | Figs. 3, 4, 5, 6 |
| `tables/bench_all_table.tex`, `bench_long_table.tex`, `bench_ci_table.tex`, `bench_dep_table.tex` | `bench_report.py` | Tables S11, S2, S3, S4 |
| `tables/shuffle_check_table.tex` | `shuffle_check.py` | Table S5 |
| `tables/sundial_check_table.tex` | `bench_report.py` | Table S6 |
| `tables/real_aurora_table.tex`, `real_robust_table.tex` | `robust_report.py` | Tables S7, S12 |
| `tables/fresh_table.tex` | `bench_report.py` | Table S8 |
| `tables/stats_table.tex`, `wilcoxon_table.tex` | `rank_stats.py` | Tables S9, S10 |
| `tables/sim_table.tex`, `sim_steps_table.tex`, `sim_budget_table.tex` | `simulation/plots.py` | Tables S13, S15, S16 |
| `figures/sim_calibration.pdf`, `sim_cost_data.pdf` | `simulation/plots.py` | Figs. S1, S2 |

Tables S1 (model versions and decoding settings) and S14 (configuration of the simulation forecasters) are
descriptive and not generated by code; Figs. 1 and 2 are diagrams.

## Run everything from scratch

**1. Environments.** The main environment runs every model except Toto, the simulation and the reports. Toto pins
older library versions and needs its own environment, and Sundial's released generation routine needs a third one.
Tested with Python 3.12 on an NVIDIA GPU with CUDA 12.8 (16 GB).

```bash
python -m venv .venv && .venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
python -m venv .venv-toto && .venv-toto/bin/pip install -r requirements-toto.txt
python -m venv .venv-sundial && .venv-sundial/bin/pip install -r requirements-sundial.txt
```

Installing `aurora-model` may replace a CUDA build of torch with a CPU build; reinstall torch from the PyTorch index
afterwards if this happens. Model weights are downloaded from the Hugging Face Hub on first use.

**2. Data.**

```bash
cd realdata
python download_data.py
```

This downloads the `ercot`, `monash_traffic` and `monash_australian_electricity` sets of the collection
`autogluon/chronos_datasets` from the Hugging Face Hub, and the AEMO price-and-demand files of the regions NSW1, QLD1,
SA1, TAS1 and VIC1 for December 2025 to September 2026. The data are not redistributed here and remain subject to
the terms of their sources.

**3. Benchmark.** `run_benchmark.sh` (or `run_benchmark.ps1` on Windows) runs every setting of `bench.py`; set `PY`
and `PY_TOTO` to the interpreters of the two environments. A single run looks like

```bash
python bench.py --data heldout --horizon 24 --models naive bolt chronos2 timesfm sundial aurora
```

Each environment writes its own result files (`--tag`), which `bench_report.py` merges. The further checks are

```bash
python shuffle_check.py                  # synthetic forecasts with known dependence (Table S5)
python flow_determinism.py               # Sundial and Aurora are fixed by their initial noise
../.venv-sundial/bin/python sundial_official.py   # Sundial through its released routine (Table S6)
python aurora_context.py                 # Aurora with other context lengths (Table S7)
python run.py; python run_fm.py; python robust.py # first-round runs and robustness checks (Table S12)
```

Chronos samples one value per network call and is the slowest model: on one RTX 5060 Ti, about 5 minutes per
dataset and sampling seed for the small version and about 50 minutes for the base version at 24 steps. The other
models take seconds to minutes per dataset.

**4. Simulation.** See `simulation/README.md`. The training-budget runs take a few hours on a CPU with four workers.

**5. Reports.** Run the commands of the previous section.



## License

The code is released under the MIT License (see `LICENSE`). The datasets and the pretrained models are subject to
the licenses and terms of their providers.
