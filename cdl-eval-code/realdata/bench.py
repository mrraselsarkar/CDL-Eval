"""Benchmark of foundation models with per-window scores (revised protocol).

Differences from run.py and run_fm.py:
  - any horizon (24, 192, 720) and two data sources: the three held-out datasets of run.py, and
    Australian state demand from AEMO for January to September 2026, which post-dates every model;
  - quantile models are scored at their native levels; every model is scored by the quantile loss and
    the coverage of the central 80% interval at the levels 0.1 to 0.9, which all models provide;
  - per-window scores (CRPS, energy and variogram score, shuffled variogram score, quantile loss,
    coverage) are stored for block-bootstrap intervals;
  - sample-based models are run with several sampling seeds;
  - a probabilistic seasonal naive forecaster is included;
  - the number of sequential network calls of every model is recorded.

Usage:
  python bench.py --data heldout --horizon 24 --models chronos bolt chronos2 timesfm sundial aurora naive
  python bench.py --data heldout --horizon 24 --models toto             (in the Toto environment)
Results are merged into results/bench_<tag>.json and results/bench_<tag>_perwindow.npz (tag: --tag).
"""

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch

import run

sim, DEV, S = run.sim, run.DEV, run.S
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
LF = 512                                   # context of the foundation models
LONG = 2880                                # context kept for the naive forecaster
Q9 = np.round(np.linspace(0.1, 0.9, 9), 2).tolist()


# =====================================================================
# Data
# =====================================================================

def load_aemo():
    """Half-hourly demand of five Australian states, December 2025 to September 2026."""
    out = []
    for region in ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]:
        files = sorted(glob.glob(os.path.join(HERE, "data", "aemo_2026", f"PRICE_AND_DEMAND_*_{region}.csv")))
        df = pd.concat([pd.read_csv(f) for f in files])
        df["t"] = pd.to_datetime(df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S")
        s = df.set_index("t")["TOTALDEMAND"].sort_index()
        s = s[~s.index.duplicated()].resample("30min", label="right", closed="right").mean().interpolate()
        out.append(s)
    return out


def prepare(data, H, n_test):
    """Test windows. For the held-out datasets these are the windows of run.prepare (identical to the
    main study for H = 24). For AEMO, origins are drawn in 2026 after 60 days of history."""
    if data == "heldout":
        run.H, run.N_TEST, run.LF = H, n_test, LONG
        out = {}
        for ds in run.DATASETS:
            d = run.prepare(ds)
            d.pop("train")
            out[ds] = d
        return out
    series = load_aemo()
    rng = np.random.default_rng(0)
    first = pd.Timestamp("2026-01-01")
    ctx, fut, sid, orig = [], [], [], []
    for _ in range(n_test):
        i = int(rng.integers(len(series)))
        s = series[i]
        lo = max(LONG, int(np.searchsorted(s.index, first)))
        o = int(rng.integers(lo, len(s) - H))
        v = s.to_numpy(dtype=float)
        mu, sd = v[:lo].mean(), v[:lo].std() + 1e-8               # scale with data before 2026
        ctx.append((v[o - LONG:o] - mu) / sd)
        fut.append((v[o:o + H] - mu) / sd)
        sid.append(i)
        orig.append(o)
    ctx, fut = np.stack(ctx), np.stack(fut)
    m = ctx[:, -336:].mean(1, keepdims=True)
    sc = ctx[:, -336:].std(1, keepdims=True) + 1e-3
    to = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
    return {"aemo": dict(ctx=to(ctx), fut=to(fut), m=to(m), sc=to(sc), season=48, n_series=len(series),
                         sid=np.array(sid), block=np.array(orig) // 336)}


# =====================================================================
# Forecasters. Each returns (kind, array, levels, sequential calls per forecast)
# =====================================================================

@torch.no_grad()
def f_chronos(ctx, H, seed, size="small"):
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(f"amazon/chronos-t5-{size}", device_map=str(DEV),
                                               torch_dtype=torch.float32)
    torch.manual_seed(seed)
    res, b, i = [], 5, 0
    while i < ctx.shape[0]:
        try:
            res.append(pipe.predict(ctx[i:i + b].cpu(), prediction_length=H, num_samples=S,
                                    limit_prediction_length=False).to(DEV).float())
            i += b
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            b = max(1, b // 2)
    return "samples", torch.cat(res), None, H


def f_chronos_base(ctx, H, seed):
    return f_chronos(ctx, H, seed, "base")


@torch.no_grad()
def f_bolt(ctx, H, seed, size="small"):
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(f"amazon/chronos-bolt-{size}", device_map=str(DEV),
                                               torch_dtype=torch.float32)
    q = torch.cat([pipe.predict_quantiles(ctx[i:i + 50].cpu(), prediction_length=H, quantile_levels=Q9,
                                          limit_prediction_length=False)[0].to(DEV).float()
                   for i in range(0, ctx.shape[0], 50)])
    return "quantiles", q, Q9, math.ceil(H / 64)


def f_bolt_base(ctx, H, seed):
    return f_bolt(ctx, H, seed, "base")


@torch.no_grad()
def f_chronos2(ctx, H, seed):
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=str(DEV))
    levels = [round(float(x), 4) for x in pipe.quantiles]
    qs = []
    for i in range(0, ctx.shape[0], 50):
        q, _ = pipe.predict_quantiles(ctx[i:i + 50, None].cpu(), prediction_length=H, quantile_levels=levels)
        qs.append(torch.stack([t[0] for t in q]).to(DEV).float())
    cfg = pipe.model.chronos_config
    return "quantiles", torch.cat(qs), levels, math.ceil(H / (cfg.output_patch_size * cfg.max_output_patches))


@torch.no_grad()
def f_timesfm(ctx, H, seed):
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch", torch_compile=False)
    model.compile(timesfm.ForecastConfig(max_context=LF, max_horizon=max(H, 128), normalize_inputs=True,
                                         per_core_batch_size=32, use_continuous_quantile_head=True,
                                         fix_quantile_crossing=True))
    _, q = model.forecast(horizon=H, inputs=[r for r in ctx.cpu().numpy()])
    q = torch.tensor(np.asarray(q)[:, :H, 1:], dtype=torch.float32, device=DEV)
    return "quantiles", q, Q9, math.ceil(H / 128)


@torch.no_grad()
def f_toto(ctx, H, seed):
    from toto.data.util.dataset import MaskedTimeseries
    from toto.inference.forecaster import TotoForecaster
    from toto.model.toto import Toto
    toto = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0").to(DEV).eval()
    fc = TotoForecaster(toto.model)
    torch.manual_seed(seed)
    res = []
    for i in range(0, ctx.shape[0], 10):
        c = ctx[i:i + 10, None]
        b, _, L = c.shape
        ts = MaskedTimeseries(series=c, padding_mask=torch.ones_like(c, dtype=torch.bool),
                              id_mask=torch.zeros_like(c, dtype=torch.int64),
                              timestamp_seconds=torch.zeros_like(c, dtype=torch.int64),   # not used by the model
                              time_interval_seconds=torch.full((b, 1), 3600, device=DEV, dtype=torch.int64))
        f = fc.forecast(ts, prediction_length=H, num_samples=S, samples_per_batch=S)
        res.append(f.samples[:, 0].permute(0, 2, 1).float())
    return "samples", torch.cat(res), None, math.ceil(H / 64)


@torch.no_grad()
def f_sundial(ctx, H, seed):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("thuml/sundial-base-128m", trust_remote_code=True).to(DEV).eval()
    torch.manual_seed(seed)
    res = []
    for i in range(0, ctx.shape[0], 25):
        c = ctx[i:i + 25]
        mu = c.mean(1, keepdim=True)
        sd = c.std(1, keepdim=True, unbiased=False).clamp_min(1e-2)
        p = model(input_ids=(c - mu) / sd, revin=False, num_samples=S, max_output_length=H,
                  use_cache=False, return_dict=True).logits
        res.append(p[..., :H].float() * sd[:, None] + mu[:, None])
    return "samples", torch.cat(res), None, math.ceil(H / 720) * 51


@torch.no_grad()
def f_aurora(ctx, H, seed):
    from aurora import load_model
    model = load_model().to(DEV).eval()
    torch.manual_seed(seed)
    res = [model.generate(inputs=ctx[i:i + 25], max_output_length=H, num_samples=S,
                          inference_token_len=48)[..., :H].float() for i in range(0, ctx.shape[0], 25)]
    return "samples", torch.cat(res), None, 51


@torch.no_grad()
def f_naive(ctx_long, H, seed, season):
    """Seasonal naive with paths of its own past errors: the forecast repeats the last season, and each
    sample path adds the error path that the same rule made from a random earlier origin in the context."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    N, L = ctx_long.shape
    reps = math.ceil(H / season)
    base = ctx_long[:, -season:].repeat(1, reps)[:, :H]
    lo, hi = season, L - H                                         # origins whose future lies in the context
    o = torch.randint(lo, hi, (N, S), device=DEV, generator=g)
    idx = o[..., None] + torch.arange(H, device=DEV)               # (N, S, H) positions of the future
    lag = o[..., None] - season + (torch.arange(H, device=DEV) % season)
    x = ctx_long[:, None, :].expand(N, S, L)
    err = x.gather(2, idx) - x.gather(2, lag)
    return "samples", base[:, None] + err, None, 0


RUNNERS = {"chronos": f_chronos, "chronos_base": f_chronos_base, "bolt": f_bolt, "bolt_base": f_bolt_base,
           "chronos2": f_chronos2, "timesfm": f_timesfm, "toto": f_toto, "sundial": f_sundial,
           "aurora": f_aurora, "naive": f_naive}
SAMPLED = {"chronos", "chronos_base", "toto", "sundial", "aurora", "naive"}


# =====================================================================
# Scoring
# =====================================================================

def quantile_samples(q, levels, gen):
    """Paths drawn independently at each step from per-step quantiles (piecewise-linear inverse CDF,
    extended linearly to the levels 0 and 1)."""
    N, Hh, K = q.shape
    lv = torch.tensor(levels, device=DEV, dtype=torch.float32)
    lo = q[..., :1] - (q[..., 1:2] - q[..., :1]) * lv[0] / (lv[1] - lv[0])
    hi = q[..., -1:] + (q[..., -1:] - q[..., -2:-1]) * (1 - lv[-1]) / (lv[-1] - lv[-2])
    qq = torch.cat([lo, q, hi], -1)
    ll = torch.cat([torch.zeros(1, device=DEV), lv, torch.ones(1, device=DEV)])
    u = torch.rand(N, S, Hh, device=DEV, generator=gen)
    j = torch.searchsorted(ll, u.contiguous()).clamp(1, len(ll) - 1)
    qe = qq[:, None].expand(N, S, Hh, len(ll))
    v0, v1 = qe.gather(3, (j - 1)[..., None])[..., 0], qe.gather(3, j[..., None])[..., 0]
    return v0 + (u - ll[j - 1]) / (ll[j] - ll[j - 1]) * (v1 - v0)


def window_scores(kind, arr, levels, d, gen):
    """Per-window scores on the scale of the last week of context."""
    m, sc = d["m"], d["sc"]
    y = (d["fut"] - m) / sc
    if kind == "quantiles":
        qn = (torch.sort(arr, -1).values - m[:, :, None]) / sc[:, :, None]
        idx = [min(range(len(levels)), key=lambda k: abs(levels[k] - v)) for v in Q9]
        q9 = qn[..., idx]
        x = quantile_samples(qn, levels, gen)
    else:
        x = (arr - m[:, None]) / sc[:, None]
        q9 = torch.quantile(x, torch.tensor(Q9, device=DEV), dim=1).permute(1, 2, 0)
    lv = torch.tensor(Q9, device=DEV)
    diff = y[..., None] - q9
    ql = (2 * torch.maximum(lv * diff, (lv - 1) * diff)).mean((1, 2))
    if kind == "quantiles":
        cov80 = ((y >= q9[..., 0]) & (y <= q9[..., -1])).float().mean(1)
    else:
        below = (x < y[:, None]).sum(1).float()
        u = (below + torch.rand(below.shape, device=DEV, generator=gen)) / (x.shape[1] + 1)
        cov80 = ((u > 0.1) & (u < 0.9)).float().mean(1)
    below = (x < y[:, None]).sum(1).float()
    u = (below + torch.rand(below.shape, device=DEV, generator=gen)) / (x.shape[1] + 1)
    cov90 = ((u > 0.05) & (u < 0.95)).float().mean(1)
    perm = torch.argsort(torch.rand(x.shape, device=DEV, generator=gen), dim=1)
    xs = x.gather(1, perm)
    crps_h = sim.crps_samples(x, y)                                  # (N, H)
    # dependence that does not depend on spread: lag-one rank correlation of the forecast paths within each
    # window (rho_model); the PIT values u are kept to compute the observed rank correlation across windows
    rx = x.argsort(1).argsort(1).float()
    rx = (rx - rx.mean(1, keepdim=True)) / rx.std(1, keepdim=True).clamp_min(1e-6)
    rho_model = ((rx[:, :, :-1] * rx[:, :, 1:]).sum(1) / (x.shape[1] - 1)).mean(1)
    chunk = max(1, int(2e8 // (x.shape[1] * x.shape[2] ** 2)))      # windows per chunk: memory grows with H^2
    out = dict(crps=crps_h.mean(1), es=sim.energy_score(x, y), vs=run.variogram_score(x, y, chunk=chunk),
               vs_shuf=run.variogram_score(xs, y, chunk=chunk), ql=ql, cov80=cov80, cov90=cov90,
               rho_model=rho_model, pit=u)
    lead = dict(crps=crps_h.mean(0), ql=(2 * torch.maximum(lv * diff, (lv - 1) * diff)).mean((0, 2)))
    return {k: v.float().cpu().numpy() for k, v in out.items()}, {k: v.float().cpu().numpy() for k, v in lead.items()}


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="heldout", choices=["heldout", "aemo"])
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tag", default="main", help="one result file per environment, so that runs do not collide")
    a = ap.parse_args()
    sim.ABAR = sim.cosine_abar(sim.CFG["k_train"])
    torch.backends.cuda.matmul.allow_tf32 = True
    jpath = os.path.join(OUT, f"bench_{a.tag}.json")
    npath = os.path.join(OUT, f"bench_{a.tag}_perwindow.npz")
    rows = json.load(open(jpath)) if os.path.exists(jpath) else []
    pw = dict(np.load(npath)) if os.path.exists(npath) else {}
    sets = prepare(a.data, a.horizon, a.n_test)
    for ds, d in sets.items():
        pw[f"{ds}|{a.horizon}|sid"], pw[f"{ds}|{a.horizon}|block"] = d["sid"], d["block"]
        ctx = d["ctx"][:, -LF:]
        for name in a.models:
            seeds = a.seeds if name in SAMPLED else [0]
            for seed in seeds:
                t0 = time.perf_counter()
                if name == "naive":
                    kind, arr, levels, calls = f_naive(d["ctx"], a.horizon, seed, d["season"])
                else:
                    kind, arr, levels, calls = RUNNERS[name](ctx, a.horizon, seed)
                el = time.perf_counter() - t0
                torch.cuda.empty_cache()
                gen = torch.Generator(device=DEV).manual_seed(7 + seed)
                win, lead = window_scores(kind, arr, levels, d, gen)
                key = f"{ds}|{a.horizon}|{name}|{seed}"
                for k, v in win.items():
                    pw[f"{key}|{k}"] = v
                row = {k: float(v.mean()) for k, v in win.items() if k != "pit"}
                ru = win["pit"].argsort(0).argsort(0).astype(float)          # observed lag-one rank correlation
                ru = (ru - ru.mean(0)) / ru.std(0)
                row["rho_obs"] = float((ru[:, :-1] * ru[:, 1:]).mean(0).mean())
                row.update(dataset=ds, horizon=a.horizon, model=name, seed=seed, kind=kind, calls=calls,
                           time_s=el, n_test=int(d["ctx"].shape[0]),
                           lead_crps=lead["crps"].tolist(), lead_ql=lead["ql"].tolist())
                rows = [r for r in rows if not (r["dataset"] == ds and r["horizon"] == a.horizon and
                                                r["model"] == name and r["seed"] == seed)] + [row]
                print(f"[bench] {ds:8s} H={a.horizon:<4d} {name:13s} s{seed} crps={row['crps']:.3f} "
                      f"ql={row['ql']:.3f} es={row['es']:.3f} gain={100 * (row['vs_shuf'] / row['vs'] - 1):+.1f}% "
                      f"cov80={row['cov80']:.2f} cov90={row['cov90']:.2f} t={el:.0f}s", flush=True)
                with open(jpath, "w") as f:
                    json.dump(rows, f)
                np.savez(npath, **pw)
    print("done", flush=True)


if __name__ == "__main__":
    main()
