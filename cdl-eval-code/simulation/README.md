# Controlled simulation

Six small forecasters, one for each type of generation, trained on synthetic data whose true behaviour is known,
with shared data, backbone family and training budget. The simulation defines the small forecasters used as trained
references in the benchmark, and shows how the number of sampling steps and the training budget of flow matching
change its calibration far more than its accuracy (Section S2 of the Supplementary Material).

| Key           | Forecaster                          | Generation                                      |
|---------------|-------------------------------------|-------------------------------------------------|
| `ar`          | Autoregressive, Gaussian head       | one value at a time, recursive sampling         |
| `direct`      | Direct multi-step                   | whole horizon in one pass, independent steps    |
| `ardiff`      | Autoregressive + diffusion head     | recursive; a diffusion model samples each step  |
| `fulldiff`    | Full-window diffusion               | whole horizon denoised jointly                  |
| `flow`        | Full-window flow matching           | whole horizon transported jointly (Euler steps) |
| `masked`      | Masked diffusion over value tokens  | tokens unmasked by confidence over rounds       |
| `masked_rand` | Masked diffusion, random order      | same model, tokens unmasked in random order     |

Datasets are synthetic (seasonal, regime switch with a bimodal future, noisy Lorenz-63) and generated with fixed seeds
inside `sim.py`; no download is needed.

## Runs

```bash
python sim.py --exp main                 # 3 datasets x 6 forecasters x 3 seeds (Table S13)
python sim.py --exp steps                # sampling-step sweep, reuses the trained models (Table S15, Fig. S2a,b)
python sim.py --exp data                 # training-set-size sweep on the Lorenz data (Fig. S2c)
python calib.py                          # PIT histograms for the calibration figure (Fig. S1)
python sim_robust.py                     # longer training runs, reused by the budget study
bash run_cpu_jobs.sh budget_bimodal      # every forecaster at 1k, 9k and 30k training steps (Table S16, Fig. S2d-f)
bash run_cpu_jobs.sh budget_lorenz
python plots.py                          # Tables S13, S15, S16 and Figs. S1, S2
```

`python sim.py --exp smoke` runs the whole pipeline with tiny settings as a check. Trained weights are saved to
`models/` inside each results folder; they are not included in the repository, and re-running an experiment retrains
missing weights and reuses existing ones. `budget.py` skips runs that are already finished. Run one GPU job at a time.

Interval coverage in Table S13 uses sample quantiles, which are biased low with 50 samples (about 86.5% expected for
a calibrated forecaster at the nominal 90% level); Fig. S1 uses the randomized probability integral transform, which
is unbiased.
