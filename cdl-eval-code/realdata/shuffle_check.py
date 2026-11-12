"""Calibration of the shuffle test on forecasts whose dependence is known.

The observed 24-step path is drawn from N(0, R(0.9)), where R(phi) has entries phi^|i-j|, as the
conditional distribution of an autoregressive process. Each forecaster draws 100 paths from
N(0, s^2 R(phi')) with its own dependence phi' and spread s. For 300 windows we compute the variogram
score of the paths and of the same paths shuffled independently at each step, and report the relative
increase (the shuffle gain) with a bootstrap interval over windows.

Usage:  python shuffle_check.py      writes ../outputs/tables/shuffle_check_table.tex and results/shuffle_check.json
"""

import os

import numpy as np
import torch

import run

H, S, N, PHI = 24, 100, 300, 0.9
HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [("Correct dependence and spread", 0.9, 1.0), ("No dependence", 0.0, 1.0),
         ("Weaker dependence", 0.5, 1.0), ("Stronger dependence", 0.99, 1.0),
         ("Correct dependence, intervals too narrow", 0.9, 0.6),
         ("Correct dependence, intervals too wide", 0.9, 1.5),
         ("No dependence, intervals too narrow", 0.0, 0.6)]


def chol(phi):
    i = torch.arange(H, dtype=torch.float64)
    R = phi ** (i[:, None] - i[None, :]).abs()
    return torch.linalg.cholesky(R + 1e-9 * torch.eye(H, dtype=torch.float64))


def ranks(a, dim):
    return a.argsort(dim).argsort(dim).float()


def rank_dependence(x, y, g):
    """Lag-one rank correlation of the forecast paths (within each window, across samples, averaged) and
    of the observations' PIT values (across windows). They agree for a forecaster with correct dependence,
    whatever its spread."""
    rx = ranks(x, 1)
    rx = (rx - rx.mean(1, keepdim=True)) / rx.std(1, keepdim=True)
    rho_m = float(((rx[:, :, :-1] * rx[:, :, 1:]).sum(1) / (x.shape[1] - 1)).mean())
    below = (x < y[:, None]).sum(1).float()
    u = (below + torch.rand(below.shape, generator=g)) / (x.shape[1] + 1)
    ru = ranks(u, 0)
    ru = (ru - ru.mean(0)) / ru.std(0)
    rho_o = float((ru[:, :-1] * ru[:, 1:]).sum(0).mean() / (u.shape[0] - 1))
    return rho_m, rho_o


def main():
    g = torch.Generator().manual_seed(0)
    y = (chol(PHI) @ torch.randn(H, N, generator=g, dtype=torch.float64)).T.float()        # (N, H)
    rows = []
    for name, phi, s in CASES:
        x = s * (chol(phi) @ torch.randn(H, N * S, generator=g, dtype=torch.float64)).T.float().reshape(N, S, H)
        perm = torch.argsort(torch.rand(x.shape, generator=g), dim=1)
        vs = run.variogram_score(x, y).numpy()
        vss = run.variogram_score(x.gather(1, perm), y).numpy()
        rng = np.random.default_rng(1)
        boot = [vss[i].sum() / vs[i].sum() - 1 for i in (rng.integers(0, N, N) for _ in range(2000))]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rho_m, rho_o = rank_dependence(x, y, g)
        rows.append((name, phi, s, 100 * (vss.mean() / vs.mean() - 1), 100 * lo, 100 * hi, rho_m, rho_o))
        print(f"{name:45s} phi={phi:4.2f} s={s:3.1f} gain={rows[-1][3]:+6.1f}% [{100*lo:+.1f}, {100*hi:+.1f}] "
              f"model rho={rho_m:.2f} observed rho={rho_o:.2f}")
    L = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.1}",
         r"\caption{\textbf{Calibration of the shuffle test.} The observed 24-step path is drawn from a Gaussian "
         r"distribution with correlation $0.9^{|i-j|}$ between steps $i$ and $j$, as for an autoregressive process. "
         r"Each forecaster draws 100 paths from a Gaussian distribution with its own correlation parameter $\phi$ "
         r"and a standard deviation $s$ times the true one. Shuffle gain: increase of the variogram score when the "
         r"paths are shuffled independently at each step, over 300 windows, with a 95\% bootstrap interval over "
         r"windows. Correct dependence gives a clear positive gain; no dependence gives a gain near zero; "
         r"dependence stronger than the truth gives a negative gain, but so do intervals that are too narrow, so the gain mixes dependence with spread. $\rho$ model: rank correlation between adjacent steps of the forecast paths, within each window and averaged; $\rho$ observed: rank correlation between the PIT values of adjacent steps across windows. The two agree when the dependence is correct, whatever the spread.}",
         r"\label{stab:shufflecheck}",
         r"\begin{tabular}{@{}l r r r r r@{}}", r"\toprule",
         r"Forecaster & $\phi$ & $s$ & Shuffle gain (\%) & $\rho$ model & $\rho$ observed \\", r"\midrule"]
    for name, phi, s, gn, lo, hi, rm, ro in rows:
        L.append(f"{name} & {phi:.2f} & {s:.1f} & ${gn:+.1f}$ (${lo:+.1f}$, ${hi:+.1f}$) & {rm:.2f} & {ro:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    os.makedirs(os.path.join(HERE, "..", "outputs", "tables"), exist_ok=True)
    with open(os.path.join(HERE, "..", "outputs", "tables", "shuffle_check_table.tex"), "w") as f:
        f.write("\n".join(L) + "\n")
    import json
    with open(os.path.join(HERE, "results", "shuffle_check.json"), "w") as f:
        json.dump([dict(case=c, phi=p, s=s, gain=float(g), lo=float(lo), hi=float(hi), rho_model=float(rm), rho_obs=float(ro))
                   for c, p, s, g, lo, hi, rm, ro in rows], f, indent=1)


if __name__ == "__main__":
    main()
