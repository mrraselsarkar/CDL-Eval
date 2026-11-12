"""Mean ranks and significance tests of the 24-step benchmark (Section VI-C of the article).

Nine forecasters (the seasonal naive baseline and the eight foundation-model runs that were made on all four
datasets) are ranked on ERCOT, Traffic, Australian electricity 2002-2015 and AEMO 2026 by three scores: quantile
loss, CRPS and the absolute error of the 80% coverage. The tests follow Demsar (2006):

  - Friedman test and Iman-Davenport correction with one value per dataset, and the Nemenyi critical distance;
  - the same tests with blocks of test windows (series x week) as the unit, which gives far more units;
  - Wilcoxon signed-rank tests of Chronos-2 against every other model on block means, with Holm's correction.

Writes ../outputs/tables/rank_table.tex (Table V), ../outputs/tables/stats_table.tex and ../outputs/tables/wilcoxon_table.tex.
"""

import os

import numpy as np
from scipy import stats

from bench_report import SEC, SUPP, SHORT, load, mean, win

DATA = {"ercot": "ERCOT", "traffic": "Traffic", "ausel": "Austr. 2002--15", "aemo": "AEMO 2026"}
MODELS = ["naive", "chronos", "toto", "timesfm", "bolt", "bolt_base", "chronos2", "sundial", "aurora"]
METRICS = {"ql": "QL", "crps": "CRPS", "cal": "Cal. error"}
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}  # Nemenyi, alpha 0.05
H = 24


def value(rows, ds, m, k):
    if k == "cal":
        return abs(100 * mean(rows, ds, H, m, "cov80") - 80)
    return mean(rows, ds, H, m, k)


def block_values(pw, k):
    """Block means (series x week) of a per-window score, for every model, pooled over the four datasets."""
    out = []
    for ds in DATA:
        b = pw[f"{ds}|{H}|sid"].astype(np.int64) * 100000 + pw[f"{ds}|{H}|block"].astype(np.int64)
        ub, inv = np.unique(b, return_inverse=True)
        cols = []
        for m in MODELS:
            if k == "cal":
                x = win(pw, ds, H, m, "cov80")
                cols.append(np.abs(100 * np.bincount(inv, x, len(ub)) / np.bincount(inv, None, len(ub)) - 80))
            else:
                x = win(pw, ds, H, m, k)
                cols.append(np.bincount(inv, x, len(ub)) / np.bincount(inv, None, len(ub)))
        out.append(np.stack(cols, 1))
    return np.concatenate(out)                                        # (blocks, models)


def friedman(X):
    """X: (N units, k methods), lower is better. Returns mean ranks, chi2_F, p, F_ID, p_ID, CD."""
    N, k = X.shape
    R = np.apply_along_axis(stats.rankdata, 1, X).mean(0)
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(R ** 2) - k * (k + 1) ** 2 / 4)
    p = stats.chi2.sf(chi2, k - 1)
    F = (N - 1) * chi2 / (N * (k - 1) - chi2)
    pF = stats.f.sf(F, k - 1, (k - 1) * (N - 1))
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6 * N))
    return R, chi2, p, F, pF, cd


def pfmt(p):
    if p < 1e-10:
        return r"$<10^{-10}$"
    if p < 1e-4:
        e = int(np.floor(np.log10(p)))
        return rf"${p / 10 ** e:.1f}\times10^{{{e}}}$"
    return f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


def main():
    rows, pw = load()
    k = len(MODELS)
    # ---------------------------------------------------------------- per-dataset ranks (Table V)
    ranks = {}
    for met in METRICS:
        X = np.array([[value(rows, ds, m, met) for m in MODELS] for ds in DATA])
        ranks[met] = np.apply_along_axis(stats.rankdata, 1, X)        # (datasets, models)
    mr = {met: ranks[met].mean(0) for met in METRICS}
    overall = np.mean([mr[met] for met in METRICS], 0)
    order = np.argsort(overall)
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Average Rank Across the Four Datasets (Lower Is Better)}", r"\label{tab:ranks}",
         r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}l r r r r@{}}", r"\toprule",
         r"Model & QL & CRPS & Cal. error & Mean \\", r"\midrule"]
    for j in order:
        L.append(f"{SHORT[MODELS[j]]} & " + " & ".join(f"{mr[met][j]:.2f}" for met in METRICS) +
                 f" & {overall[j]:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\vspace{2pt}", r"\parbox{\columnwidth}{\footnotesize Ranks 1 to 9 per dataset and score, averaged over "
          r"ERCOT, Traffic, Australian electricity 2002--2015 and AEMO 2026 (24-step forecasts). Cal. error: absolute "
          r"difference between the coverage of the central 80\% interval and 80 points. Chronos base was not run on "
          r"the AEMO data and is not ranked. TimesFM 2.5 probably saw the Traffic and 2002--2015 Australian series "
          r"during pretraining (Section~\ref{sec:leakage}).}", r"\end{table}"]
    open(os.path.join(SEC, "rank_table.tex"), "w").write("\n".join(L) + "\n")
    print("mean ranks:", {SHORT[MODELS[j]]: (round(float(mr['ql'][j]), 2), round(float(mr['crps'][j]), 2),
                                             round(float(mr['cal'][j]), 2), round(float(overall[j]), 2)) for j in order})
    per_ds_rank = {ds: {SHORT[m]: int(ranks['ql'][i, j]) for j, m in enumerate(MODELS)} for i, ds in enumerate(DATA)}
    print("QL ranks per dataset:", per_ds_rank)

    # ---------------------------------------------------------------- Friedman, Iman-Davenport, Nemenyi
    S = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize",
         r"\caption{Friedman test, Iman--Davenport correction and Nemenyi critical distance (CD, $\alpha = 0.05$) "
         r"for the nine forecasters of Table~\ref{tab:ranks}, with the dataset or a block of test windows (one series "
         r"and one week) as the unit. Best: the forecaster with the lowest average rank; Gap: difference between its "
         r"average rank and that of the runner-up.}", r"\label{stab:friedman}",
         r"\begin{tabular}{@{}l l r r r r r r l r@{}}", r"\toprule",
         r"Unit & Score & $N$ & $\chi^2_F$ & $p$ & $F_{ID}$ & $p_{ID}$ & CD & Best & Gap \\", r"\midrule"]
    summary = {}
    for unit in ["dataset", "block"]:
        for met, name in METRICS.items():
            if unit == "dataset":
                X = np.array([[value(rows, ds, m, met) for m in MODELS] for ds in DATA])
            else:
                X = block_values(pw, met)
            R, chi2, p, F, pF, cd = friedman(X)
            o = np.argsort(R)
            summary[(unit, met)] = (X.shape[0], chi2, p, F, pF, cd, R)
            S.append(f"{unit.capitalize() if met == 'ql' else ''} & {name} & {X.shape[0]} & {chi2:.1f} & {pfmt(p)} & "
                     f"{F:.2f} & {pfmt(pF)} & {cd:.2f} & {SHORT[MODELS[o[0]]]} & {R[o[1]] - R[o[0]]:.2f} \\\\")
            print(f"friedman {unit:7s} {met:4s} N={X.shape[0]:4d} chi2={chi2:7.1f} p={p:.2e} F={F:.2f} pF={pF:.2e} "
                  f"CD={cd:.2f} ranks " + ", ".join(f"{SHORT[MODELS[j]]} {R[j]:.2f}" for j in o))
        if unit == "dataset":
            S.append(r"\midrule")
    S += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(SUPP, "stats_table.tex"), "w").write("\n".join(S) + "\n")

    # ---------------------------------------------------------------- Wilcoxon, Chronos-2 against each model
    W = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize",
         r"\caption{Wilcoxon signed-rank tests of Chronos-2 against each other forecaster, on block means of the "
         r"quantile loss and the CRPS (one block: one series and one week; 24-step forecasts, four datasets "
         r"pooled). Better: share of blocks in which Chronos-2 had the lower score; $p_{\mathrm{Holm}}$: two-sided "
         r"$p$ value after Holm's correction for the eight comparisons of each score.}", r"\label{stab:wilcoxon}",
         r"\begin{tabular}{@{}l r r r r@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{Quantile loss} & \multicolumn{2}{c}{CRPS} \\",
         r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
         r"Chronos-2 against & Better (\%) & $p_{\mathrm{Holm}}$ & Better (\%) & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
    ref = MODELS.index("chronos2")
    res = {}
    for met in ["ql", "crps"]:
        X = block_values(pw, met)
        others = [j for j in range(k) if j != ref]
        ps = np.array([stats.wilcoxon(X[:, ref], X[:, j]).pvalue for j in others])
        better = [float(np.mean(X[:, ref] < X[:, j])) for j in others]
        o = np.argsort(ps)
        adj = np.empty_like(ps)
        run_max = 0
        for r_, i in enumerate(o):                                       # Holm step-down
            run_max = max(run_max, min(1, (len(ps) - r_) * ps[i]))
            adj[i] = run_max
        for i, j in enumerate(others):
            res[(met, MODELS[j])] = (better[i], adj[i])
        print(f"wilcoxon {met}: n_blocks={X.shape[0]} " +
              ", ".join(f"{SHORT[MODELS[j]]} better {100 * better[i]:.0f}% p={adj[i]:.1e}" for i, j in enumerate(others)))
    for j in range(k):
        if j == ref:
            continue
        m = MODELS[j]
        W.append(f"{SHORT[m]} & {100 * res[('ql', m)][0]:.0f} & {pfmt(res[('ql', m)][1])} & "
                 f"{100 * res[('crps', m)][0]:.0f} & {pfmt(res[('crps', m)][1])} \\\\")
    W += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(SUPP, "wilcoxon_table.tex"), "w").write("\n".join(W) + "\n")


if __name__ == "__main__":
    main()
