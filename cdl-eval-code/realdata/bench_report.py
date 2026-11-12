"""Tables and figures of the revised benchmark (bench.py).

Reads results/bench_*.json and results/bench_*_perwindow.npz (all environments), results/robust.json
(small forecasters), results/sundial_official.json, and writes
  ../outputs/tables/real_table.tex            Table 2: held-out data, H = 24
  ../outputs/figures/accuracy.pdf               Fig. 2: quantile loss and coverage, H = 24
  ../outputs/figures/dependence.pdf             Fig. 3: dependence along the paths and the shuffle test
  ../outputs/figures/horizon.pdf                Fig. 4: accuracy by lead time at H = 720 and coverage by horizon
  ../outputs/figures/leakage.pdf                Fig. 5: leakage test on Australian demand, 2002-2015 and 2026
  ../outputs/tables/fresh_table.tex      the numbers behind Fig. 5
  ../outputs/tables/bench_all_table.tex  all scores at H = 24, with spread over sampling seeds
  ../outputs/tables/bench_long_table.tex scores at H = 192 and 720
  ../outputs/tables/bench_ci_table.tex   block-bootstrap intervals for the main comparisons
  ../outputs/tables/bench_dep_table.tex  shuffle gain with intervals and rank dependence
  ../outputs/tables/sundial_check_table.tex  Sundial: forward pass against its generation routine
and prints the numbers quoted in the text.
"""

import glob
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEC = SUPP = os.path.join(HERE, "..", "outputs", "tables")       # LaTeX tables of the article
FIG = os.path.join(HERE, "..", "outputs", "figures")
for _d in (SEC, FIG):
    os.makedirs(_d, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, ".."))
import figstyle  # noqa: E402

figstyle.apply()
DS = {"ercot": "ERCOT load", "traffic": "Traffic occupancy", "ausel": "Australian electricity"}
FM = ["chronos", "chronos_base", "toto", "timesfm", "bolt", "bolt_base", "chronos2", "sundial", "aurora"]
PATHS = {"naive", "chronos", "chronos_base", "toto", "sundial", "aurora"}       # models with their own paths
LABEL = {"naive": "Seasonal naive with error paths", "chronos": "Chronos small (AR, tokens)",
         "chronos_base": "Chronos base (AR, tokens)", "toto": "Toto (AR, patches)",
         "timesfm": "TimesFM 2.5 (AR, quantiles)", "bolt": "Chronos-Bolt small (direct)",
         "bolt_base": "Chronos-Bolt base (direct)", "chronos2": "Chronos-2 (direct)",
         "sundial": "Sundial (AR + flow head)", "aurora": "Aurora (full-window flow)",
         "ar": "AR, Gaussian head (trained)", "flow": "Full-window flow matching (trained)"}
SHORT = {"naive": "Naive", "chronos": "Chronos", "chronos_base": "Chronos base", "toto": "Toto",
         "timesfm": "TimesFM 2.5", "bolt": "Chronos-Bolt", "bolt_base": "Chronos-Bolt base",
         "chronos2": "Chronos-2", "sundial": "Sundial", "aurora": "Aurora"}
TYPEKEY = {"chronos": "chronos", "chronos_base": "chronos", "toto": "toto", "timesfm": "timesfm_ind",
           "bolt": "bolt_ind", "bolt_base": "bolt_ind", "chronos2": "chronos2_ind", "sundial": "sundial",
           "aurora": "aurora"}
LEAK = {("toto", "ercot"): "a", ("toto", "traffic"): "b", ("toto", "ausel"): "b",
        ("timesfm", "traffic"): "b", ("timesfm", "ausel"): "b",
        ("sundial", "traffic"): "c", ("sundial", "ausel"): "c", ("sundial", "ercot"): "c",
        ("chronos2", "traffic"): "c", ("chronos2", "ausel"): "c", ("chronos2", "ercot"): "c",
        ("aurora", "ercot"): "d", ("aurora", "ausel"): "d"}
B = 5000


def load():
    """All result files; a later file (in name order, e.g. bench_main_fix after bench_main) overrides rows and
    per-window scores of the same dataset, horizon, model and seed."""
    rows, pw = {}, {}
    for f in sorted(glob.glob(os.path.join(RES, "bench_*.json"))):
        for r in json.load(open(f)):
            rows[(r["dataset"], r["horizon"], r["model"], r["seed"])] = r
    for f in sorted(glob.glob(os.path.join(RES, "bench_*_perwindow.npz"))):
        pw.update(dict(np.load(f)))
    return list(rows.values()), pw


def sel(rows, ds, h, m):
    return [r for r in rows if r["dataset"] == ds and r["horizon"] == h and r["model"] == m]


def mean(rows, ds, h, m, k):
    v = [r[k] for r in sel(rows, ds, h, m) if k in r]
    return float(np.mean(v)) if v else float("nan")


def sd(rows, ds, h, m, k):
    v = [r[k] for r in sel(rows, ds, h, m) if k in r]
    return float(np.std(v)) if len(v) > 1 else float("nan")


def win(pw, ds, h, m, k):
    """Per-window score averaged over the sampling seeds."""
    keys = [x for x in pw if x.startswith(f"{ds}|{h}|{m}|") and x.endswith(f"|{k}")]
    return np.mean([pw[x] for x in keys], 0) if keys else None


def boot_ratio(pw, ds, h, a, b, k="ql", rng=None):
    """Ratio of mean scores of a and b with a block bootstrap over series and weeks."""
    xa, xb = win(pw, ds, h, a, k), win(pw, ds, h, b, k)
    if xa is None or xb is None:
        return None
    blocks = pw[f"{ds}|{h}|sid"].astype(np.int64) * 100000 + pw[f"{ds}|{h}|block"].astype(np.int64)
    ub, inv = np.unique(blocks, return_inverse=True)
    sa, sb = np.bincount(inv, xa, len(ub)), np.bincount(inv, xb, len(ub))
    rng = rng or np.random.default_rng(0)
    draw = rng.integers(0, len(ub), (B, len(ub)))
    r = sa[draw].sum(1) / sb[draw].sum(1)
    return xa.mean() / xb.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5), r


def fmt_cell(v, nd, bold=False, mark=""):
    s = "--" if v != v else f"{v:.{nd}f}"
    s = rf"\textbf{{{s}}}" if bold and v == v else s
    return s + (rf"$^{{{mark}}}$" if mark else "")


# ---------------------------------------------------------------------------------------------- main table
def table_main(rows, small):
    models = ["naive"] + [m for m in FM if sel(rows, "ercot", 24, m)]
    head = (r" & \multicolumn{3}{c}{" + DS["ercot"] + r"} & \multicolumn{3}{c}{" + DS["traffic"] +
            r"} & \multicolumn{3}{c}{" + DS["ausel"] + r"} \\")
    L = [r"\begin{table*}[!t]", r"\centering", r"\caption{Foundation Models on Held-Out Data, 24-Step Forecasts}",
         r"\label{tab:real}", r"\footnotesize", r"\renewcommand{\arraystretch}{1.1}", r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{@{}l r r r r r r r r r@{}}", r"\toprule", head,
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}",
         r"Model & QL & Cov. & ES & QL & Cov. & ES & QL & Cov. & ES \\"]
    clean = lambda m, ds: LEAK.get((m, ds), "") not in ("a", "b")
    best = {}
    for ds in DS:
        for k in ["ql", "es"]:
            cand = [m for m in models if clean(m, ds) and (k == "ql" or m in PATHS)]
            best[(ds, k)] = min(cand, key=lambda m: mean(rows, ds, 24, m, k))
    for group, ms in [("Baseline", ["naive"]), ("Foundation models, zero-shot", models[1:])]:
        L += [r"\midrule", rf"\multicolumn{{10}}{{@{{}}l}}{{\textit{{{group}}}}} \\"]
        for m in ms:
            cells = []
            for ds in DS:
                cells.append(fmt_cell(mean(rows, ds, 24, m, "ql"), 3, best[(ds, "ql")] == m, LEAK.get((m, ds), "")))
                cells.append(f"{np.floor(100 * mean(rows, ds, 24, m, 'cov80') + 0.5):.0f}")
                cells.append(fmt_cell(mean(rows, ds, 24, m, "es"), 2, best[(ds, "es")] == m) if m in PATHS else "--")
            L.append(LABEL[m] + " & " + " & ".join(cells) + r" \\")
    L += [r"\midrule", r"\multicolumn{10}{@{}l}{\textit{Trained on each dataset (reference)}} \\"]
    for m in ["ar", "flow"]:
        cells = []
        for ds in DS:
            rr = [r for r in small if r["dataset"] == ds and r["model"].rsplit("_s", 1)[0] == m]
            cells += [f"{np.mean([r['qloss'] for r in rr]):.3f}", f"{np.floor(100 * np.mean([r['pit80'] for r in rr]) + 0.5):.0f}",
                      f"{np.mean([r['es'] for r in rr]):.2f}"]
        L.append(LABEL[m] + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"", r"\vspace{3pt}",
          r"\parbox{\textwidth}{\footnotesize QL: quantile loss (twice the mean pinball loss) at the levels 0.1 to 0.9, "
          r"which every model provides; Cov.: coverage (\%) of the central 80\% interval between these levels, ideally "
          r"80; ES: energy score of the sample paths, given for models that generate their own paths. Lower is "
          r"better for QL and ES. Scores are computed on 300 test windows per dataset, scaled by the last week of each "
          r"window's context. Foundation models were applied without fine-tuning to 512 context steps and drew 100 paths "
          r"per window; sample-based models are averaged over three sampling seeds, and Table~\ref{S-stab:benchall} gives "
          r"the spread. Quantile models are scored at their native levels. The two small forecasters of "
          r"Section~\ref{S-snote:sim} were trained on each dataset with 48 context steps (three seeds). The best QL and "
          r"ES among models without known or probable leakage are in bold. The type labels refer to each model's "
          r"decoder; at 24 steps every model except Chronos produces the horizon in one step, as Table~\ref{tab:models} shows. "
          r"Leakage marks, from Table~\ref{tab:models}: $^{a}$ in the pretraining corpus; $^{b}$ probably in it; $^{c}$ "
          r"exclusion stated by the authors while the corpus contains copies of the data; $^{d}$ not known.}",
          r"\end{table*}"]
    open(os.path.join(SEC, "real_table.tex"), "w").write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------------------- fresh-data test
def table_fresh(rows, pw):
    models = [m for m in FM + ["naive"] if m != "bolt" and sel(rows, "aemo", 24, m) and sel(rows, "ausel", 24, m)]
    rng = np.random.default_rng(0)
    L = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.12}",
         r"\caption{\textbf{A test for leakage on Australian electricity demand.} Quantile loss of each model "
         r"divided by that of Chronos-Bolt (small), which was trained without these data, on the Monash series "
         r"(2002--2015, contained in the GIFT-Eval and LOTSA pretraining collections) and on the same five states in "
         r"January to September 2026 from AEMO, which post-date every model. Values below 1 favour the model; "
         r"95\% intervals from a block bootstrap over states and weeks. A model that learned the old series "
         r"should lose its advantage on the new data; a model that did not should keep it.}",
         r"\label{stab:fresh}", r"\begin{tabular}{@{}l l r r r@{}}", r"\toprule",
         r"Model & Leakage of the 2002--2015 series & 2002--2015 & 2026 & Change \\", r"\midrule"]
    leak_txt = {"a": "In corpus", "b": "Probably in corpus", "c": "Exclusion stated; copies in corpus",
                "d": "Not known", "": "Excluded"}
    out = {}
    for m in models:
        old = boot_ratio(pw, "ausel", 24, m, "bolt", rng=rng)
        new = boot_ratio(pw, "aemo", 24, m, "bolt", rng=rng)
        if old is None or new is None:
            continue
        ch = new[3] / old[3] - 1
        out[m] = (old, new, np.percentile(ch, [2.5, 97.5]))
        L.append(f"{SHORT[m]} & {leak_txt[LEAK.get((m, 'ausel'), '')]} & {old[0]:.2f} ({old[1]:.2f}, {old[2]:.2f}) & "
                 f"{new[0]:.2f} ({new[1]:.2f}, {new[2]:.2f}) & ${100 * (new[0] / old[0] - 1):+.0f}\\%$ \\\\")
        print(f"fresh {m:12s} old {old[0]:.2f} [{old[1]:.2f},{old[2]:.2f}] new {new[0]:.2f} [{new[1]:.2f},{new[2]:.2f}] "
              f"change {100*(new[0]/old[0]-1):+.0f}% CI [{100*out[m][2][0]:+.0f},{100*out[m][2][1]:+.0f}]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(SUPP, "fresh_table.tex"), "w").write("\n".join(L) + "\n")
    return out


# ---------------------------------------------------------------------------------------------- figure 6
def bar_style(m):
    base = figstyle.MODEL_COLOR.get(TYPEKEY.get(m, m), figstyle.GREY)
    shade = {"chronos": 0.62, "chronos_base": 0.45, "timesfm": 1.35, "bolt_base": 0.8, "chronos2": 0.62}.get(m, 1.0)
    fill = figstyle.darker(base, shade) if shade < 1 else figstyle.lighter(base, shade - 1) if shade > 1 else base
    return fill, figstyle.MODEL_HATCH.get(TYPEKEY.get(m, m), "")


def fig_real(rows):
    models = ["naive"] + [m for m in FM if sel(rows, "ercot", 24, m)]
    pathm = [m for m in models if m in PATHS]
    figstyle.MODEL_COLOR.setdefault("naive", figstyle.GREY)
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 7.0))
    w = 0.8 / len(models)
    for r_, (key, yl) in enumerate([("ql", "Quantile loss (lower is better)"), ("cov80", "Coverage of 80% intervals (%)")]):
        ax = axes[r_]
        for i, m in enumerate(models):
            hs = [mean(rows, ds, 24, m, key) * (100 if key == "cov80" else 1) for ds in DS]
            x = np.arange(len(DS)) + (i - (len(models) - 1) / 2) * w
            labels = [f"{np.floor(h + 0.5):.0f}" for h in hs] if key == "cov80" else [f"{h:.3f}" for h in hs]
            fill, hatch = bar_style(m)
            figstyle.bars(ax, x, hs, w * 0.9, fill, hatch, labels=labels, fontsize=5.0, rotation=90, pad=1.0)
        if key == "cov80":
            ax.axhline(80, color=figstyle.INK, lw=0.8, ls=(0, (3, 2)), zorder=1)
            ax.set_ylim(0, 108)
        else:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
        ax.set_xticks(range(len(DS)))
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(yl)
        ax.set_title(f"{'ab'[r_]}  " + ["Probabilistic accuracy", "Calibration"][r_])
    ax = axes[2]
    w2 = 0.8 / len(pathm)
    for i, m in enumerate(pathm):
        x = np.arange(len(DS)) + (i - (len(pathm) - 1) / 2) * w2
        hm = [mean(rows, ds, 24, m, "rho_model") for ds in DS]
        ho = [mean(rows, ds, 24, m, "rho_obs") for ds in DS]
        fill, hatch = bar_style(m)
        figstyle.bars(ax, x, hm, w2 * 0.9, fill, hatch, labels=[f"{h:.2f}" for h in hm], fontsize=5.0, rotation=90, pad=1.0)
        ax.scatter(x, ho, marker="_", s=90, color=figstyle.INK, linewidths=1.6, zorder=5)
    ax.set_ylim(-0.1, 1.15)
    ax.axhline(0, color=figstyle.INK, lw=0.6)
    ax.set_xticks(range(len(DS)))
    ax.set_xticklabels(list(DS.values()))
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Rank correlation of adjacent steps")
    ax.set_title("c  Dependence along the sample paths (bars: model; dashes: observed)")
    handles = [figstyle.patch(*bar_style(m), LABEL[m]) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.985), columnspacing=1.0)
    fig.tight_layout(h_pad=1.0)
    fig.savefig(os.path.join(FIG, "real.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- figure 7
def fig_horizon(rows):
    h = 720
    models = [m for m in ["naive", "toto", "timesfm", "bolt", "chronos2", "sundial", "aurora"] if sel(rows, "ercot", h, m)]
    if not models:
        return
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5), sharey=False)
    for c, ds in enumerate(DS):
        ax = axes[c]
        for m in models:
            rr = sel(rows, ds, h, m)
            if not rr:
                continue
            v = np.array(rr[0]["lead_ql"])
            day = 24 if ds != "ausel" else 48
            v = v[: (len(v) // day) * day].reshape(-1, day).mean(1)                # daily means
            fill, _ = bar_style(m)
            ax.plot(np.arange(1, len(v) + 1), v, color=fill, lw=1.2, label=SHORT[m],
                    ls="--" if m == "naive" else "-")
        ax.set_title(f"{'abc'[c]}  {DS[ds]}")
        ax.set_xlabel("Forecast day")
        if c == 0:
            ax.set_ylabel("Quantile loss")
    axes[0].legend(fontsize=6.2, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "horizon.pdf"))
    plt.close(fig)


# ------------------------------------------------------------------------------------------- SI tables
def table_all(rows):
    models = ["naive"] + [m for m in FM if sel(rows, "ercot", 24, m)]
    head = r"Model & QL & CRPS & ES & VS & Cov.\ 80 & Cov.\ 90 & Calls \\"
    L = [r"{\footnotesize", r"\setlength{\tabcolsep}{1.4pt}", r"\begin{longtable}{@{}l r r r r r r r@{}}",
         r"\caption{\textbf{All scores of the benchmark, 24-step forecasts.} Mean, with the standard deviation over "
         r"three sampling seeds in parentheses for sample-based models. QL: quantile loss at the levels 0.1 to 0.9; "
         r"CRPS from 100 paths (for quantile models, paths drawn independently at each step from their native "
         r"levels); ES and VS: energy and variogram score (for quantile models they depend on the imposed sampling "
         r"and are given in italics); Cov.: coverage (\%) of central intervals; Calls: sequential network "
         r"evaluations per forecast, counting the backbone and, for the flow models, the 50 steps of the flow head. "
         r"Wall-clock times are not reported because the GPU was shared with other jobs during the runs.}",
         r"\label{stab:benchall}\\", r"\toprule", head, r"\endfirsthead",
         r"\multicolumn{8}{@{}l}{\textit{Table~\ref{stab:benchall} continued}}\\", r"\toprule", head,
         r"\endhead", r"\bottomrule", r"\endfoot"]
    for j, ds in enumerate(list(DS) + ["aemo"]):
        name = DS.get(ds, "Australian electricity, AEMO 2026")
        L += ([r"\midrule"] if j else []) + [rf"\multicolumn{{8}}{{@{{}}l}}{{\textit{{{name}}}}} \\"]
        for m in models:
            if not sel(rows, ds, 24, m):
                continue
            def c(k, nd, scale=1):
                v, s = mean(rows, ds, 24, m, k) * scale, sd(rows, ds, 24, m, k) * scale
                t = f"{v:.{nd}f}" + (f" ({s:.{nd}f})" if s == s else "")
                return t if m in PATHS or k not in ("es", "vs") else rf"\textit{{{t}}}"
            L.append(f"{LABEL[m]} & {c('ql', 3)} & {c('crps', 3)} & {c('es', 2)} & {c('vs', 4)} & "
                     f"{c('cov80', 1, 100)} & {c('cov90', 1, 100)} & {sel(rows, ds, 24, m)[0]['calls']} \\\\")
    L += [r"\end{longtable}", r"}"]
    open(os.path.join(SUPP, "bench_all_table.tex"), "w").write("\n".join(L) + "\n")


def table_long(rows):
    models = [m for m in ["naive", "chronos", "toto", "timesfm", "bolt", "chronos2", "sundial", "aurora"]]
    hs = [h for h in (24, 192, 720) if any(r["horizon"] == h for r in rows)]
    head = r"Model & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$H={h}$}}" for h in hs) + r" & Calls at $H=720$ \\"
    sub = r" & " + " & ".join(["QL & Cov." for _ in hs]) + r" & \\"
    L = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.08}",
         r"\caption{\textbf{Longer horizons.} Quantile loss (QL) and coverage (\%) of central 80\% intervals over the "
         r"whole horizon, for $H = 24$ (300 windows) and $H = 192$ and 720 (150 windows each, one sampling seed). "
         r"Chronos was run up to 192 steps; it generates one value per network call. Calls: sequential network "
         r"evaluations per forecast at $H = 720$.}",
         r"\label{stab:benchlong}", r"\begin{tabular}{@{}l " + "r r " * len(hs) + r"r@{}}", r"\toprule", head, sub]
    for ds in DS:
        L += [r"\midrule", rf"\multicolumn{{{2 * len(hs) + 2}}}{{@{{}}l}}{{\textit{{{DS[ds]}}}}} \\"]
        for m in models:
            cells = []
            for h in hs:
                q, c = mean(rows, ds, h, m, "ql"), mean(rows, ds, h, m, "cov80")
                cells += ["--" if q != q else f"{q:.3f}", "--" if c != c else f"{100 * c:.0f}"]
            calls = sel(rows, ds, 720, m)
            L.append(f"{SHORT[m]} & " + " & ".join(cells) + f" & {calls[0]['calls'] if calls else '--'} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(SUPP, "bench_long_table.tex"), "w").write("\n".join(L) + "\n")


def table_ci(rows, pw):
    comps = [("chronos2", "bolt"), ("chronos2", "toto"), ("chronos2", "timesfm"), ("toto", "chronos"),
             ("bolt_base", "bolt"), ("chronos_base", "chronos"), ("sundial", "bolt"), ("aurora", "sundial"),
             ("chronos2", "naive"), ("sundial", "naive")]
    rng = np.random.default_rng(1)
    L = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.08}",
         r"\caption{\textbf{Uncertainty of the main comparisons, 24-step forecasts.} Relative difference in mean "
         r"quantile loss of model A and model B (negative values favour A), with 95\% intervals from a block "
         rf"bootstrap that resamples blocks of windows from the same series and week ({B:,} resamples). The "
         r"intervals for sample-based models use scores averaged over the sampling seeds.}",
         r"\label{stab:benchci}", r"\begin{tabular}{@{}l l l r@{}}", r"\toprule",
         r"Data & Model A & Model B & A vs B (95\% CI) \\", r"\midrule"]
    for j, ds in enumerate(DS):
        if j:
            L.append(r"\addlinespace")
        for a, b in comps:
            res = boot_ratio(pw, ds, 24, a, b, rng=rng)
            if res is None:
                continue
            L.append(f"{DS[ds] if (a, b) == comps[0] else ''} & {SHORT[a]} & {SHORT[b]} & "
                     f"${100 * (res[0] - 1):+.1f}\\%$ (${100 * (res[1] - 1):+.1f}$, ${100 * (res[2] - 1):+.1f}$) \\\\")
            print(f"ci {ds:8s} {a:12s} vs {b:12s} {100*(res[0]-1):+.1f}% [{100*(res[1]-1):+.1f}, {100*(res[2]-1):+.1f}]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(SUPP, "bench_ci_table.tex"), "w").write("\n".join(L) + "\n")


def table_dep(rows, pw):
    models = [m for m in ["naive", "chronos", "chronos_base", "toto", "sundial", "aurora"] if sel(rows, "ercot", 24, m)]
    rng = np.random.default_rng(2)
    L = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.08}",
         r"\caption{\textbf{Dependence along the sample paths, 24-step forecasts.} Shuffle gain: increase of the "
         r"variogram score when the paths are shuffled independently at each step, with a 95\% block-bootstrap "
         r"interval; as Table~\ref{stab:shufflecheck} shows, it mixes dependence with spread. $\rho$ model: "
         r"rank correlation between adjacent steps of the model's paths, within each window; $\rho$ observed: rank "
         r"correlation between the PIT values of adjacent steps across windows. The two agree for a model whose "
         r"dependence is correct, whatever its spread. The table shows the models that generate their own paths.}",
         r"\label{stab:benchdep}", r"\begin{tabular}{@{}l l r r r@{}}", r"\toprule",
         r"Data & Model & Shuffle gain (\%) & $\rho$ model & $\rho$ observed \\", r"\midrule"]
    for j, ds in enumerate(DS):
        if j:
            L.append(r"\addlinespace")
        for i, m in enumerate(models):
            vs, vss = win(pw, ds, 24, m, "vs"), win(pw, ds, 24, m, "vs_shuf")
            if vs is None or vss is None:
                continue
            blocks = pw[f"{ds}|24|sid"].astype(np.int64) * 100000 + pw[f"{ds}|24|block"].astype(np.int64)
            ub, inv = np.unique(blocks, return_inverse=True)
            sa, sb = np.bincount(inv, vss, len(ub)), np.bincount(inv, vs, len(ub))
            d = rng.integers(0, len(ub), (B, len(ub)))
            r = sa[d].sum(1) / sb[d].sum(1) - 1
            g = vss.mean() / vs.mean() - 1
            L.append(f"{DS[ds] if i == 0 else ''} & {SHORT[m]} & ${100 * g:+.1f}$ (${100 * np.percentile(r, 2.5):+.1f}$, "
                     f"${100 * np.percentile(r, 97.5):+.1f}$) & {mean(rows, ds, 24, m, 'rho_model'):.2f} & "
                     f"{mean(rows, ds, 24, m, 'rho_obs'):.2f} \\\\")
            print(f"dep {ds:8s} {m:12s} gain {100*g:+.1f}% [{100*np.percentile(r,2.5):+.1f},{100*np.percentile(r,97.5):+.1f}] "
                  f"rho_model {mean(rows, ds, 24, m, 'rho_model'):.2f} rho_obs {mean(rows, ds, 24, m, 'rho_obs'):.2f}")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(SUPP, "bench_dep_table.tex"), "w").write("\n".join(L) + "\n")


def table_sundial(rows):
    path = os.path.join(RES, "sundial_official.json")
    if not os.path.exists(path):
        return
    off = json.load(open(path))
    L = [r"\begin{table}[!htbp]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.1}",
         r"\caption{\textbf{Sundial through its forward pass and through its released generation routine.} The "
         r"benchmark calls Sundial's forward pass with \nolinkurl{transformers} 4.57; the released routine "
         r"(\nolinkurl{generate}) was run in a separate environment with \nolinkurl{transformers} 4.40.1, the version "
         r"its model card requires. Same test windows, 24-step forecasts, mean over three sampling seeds.}",
         r"\label{stab:sundialcheck}", r"\begin{tabular}{@{}l l r r r@{}}", r"\toprule",
         r"Data & Interface & CRPS & Cov.\ 80 & Cov.\ 90 \\", r"\midrule"]
    for ds in DS:
        o = [r for r in off if r["dataset"] == ds]
        L.append(f"{DS[ds]} & Forward pass & {mean(rows, ds, 24, 'sundial', 'crps'):.3f} & "
                 f"{100 * mean(rows, ds, 24, 'sundial', 'cov80'):.1f} & {100 * mean(rows, ds, 24, 'sundial', 'cov90'):.1f} \\\\")
        L.append(f" & Generation routine & {np.mean([r['crps'] for r in o]):.3f} & {100 * np.mean([r['cov80'] for r in o]):.1f} & "
                 f"{100 * np.mean([r['cov90'] for r in o]):.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(SUPP, "sundial_check_table.tex"), "w").write("\n".join(L) + "\n")


def summary(rows):
    for h in (24, 192, 720):
        for ds in list(DS) + ["aemo"]:
            ms = sorted({r["model"] for r in rows if r["dataset"] == ds and r["horizon"] == h})
            if not ms:
                continue
            print(f"== {ds} H={h}")
            for m in ms:
                g = lambda k: mean(rows, ds, h, m, k)
                print(f"  {m:13s} ql={g('ql'):.3f} crps={g('crps'):.3f} es={g('es'):.3f} cov80={100*g('cov80'):.0f} "
                      f"cov90={100*g('cov90'):.0f} rho_m={g('rho_model'):.2f} rho_o={g('rho_obs'):.2f} "
                      f"gain={100*(g('vs_shuf')/g('vs')-1):+.1f}%")


# ------------------------------------------------------------------------------------- figures of Paper 2
def _dataset_axes(ax, labels=True):
    ax.set_xticks(range(len(DS)))
    ax.set_xticklabels(list(DS.values()) if labels else [])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, len(DS) - 0.45)


LEGEND = {"naive": "Seasonal naive", "chronos": "Chronos small (AR)", "chronos_base": "Chronos base (AR)",
          "toto": "Toto (AR)", "timesfm": "TimesFM 2.5 (AR)", "bolt": "Chronos-Bolt small (direct)",
          "bolt_base": "Chronos-Bolt base (direct)", "chronos2": "Chronos-2 (direct)",
          "sundial": "Sundial (AR + flow head)", "aurora": "Aurora (full-window flow)"}
DS_TICK = {"ercot": "ERCOT", "traffic": "Traffic", "ausel": "Australia"}
TEXT_W, COL_W = 7.16, 3.5      # inches: text width and column width of an IEEE journal page (figures print at 100%)


def _group_axes(ax):
    ax.set_xticks(range(len(DS)))
    ax.set_xticklabels([DS_TICK[d] for d in DS])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(DS) - 0.5)


def _dashed(label):
    from matplotlib.lines import Line2D
    return Line2D([0], [0], color=figstyle.INK, lw=0.8, ls=(0, (3, 2)), label=label)


LEGEND = {"naive": "Seasonal naive", "chronos": "Chronos small (AR)", "chronos_base": "Chronos base (AR)",
          "toto": "Toto (AR)", "timesfm": "TimesFM 2.5 (AR)", "bolt": "Chronos-Bolt small (direct)",
          "bolt_base": "Chronos-Bolt base (direct)", "chronos2": "Chronos-2 (direct)",
          "sundial": "Sundial (AR + flow head)", "aurora": "Aurora (full-window flow)"}
DS_TICK = {"ercot": "ERCOT", "traffic": "Traffic", "ausel": "Australia"}
TEXT_W, COL_W = 7.16, 3.5      # inches: text width and column width of an IEEE journal page (figures print at 100%)


def _group_axes(ax):
    ax.set_xticks(range(len(DS)))
    ax.set_xticklabels([DS_TICK[d] for d in DS])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(DS) - 0.5)


def _dashed(label):
    from matplotlib.lines import Line2D
    return Line2D([0], [0], color=figstyle.INK, lw=0.8, ls=(0, (3, 2)), label=label)


def fig_accuracy(rows):
    """Fig. 3: quantile loss and coverage of the central 80% interval, H = 24 (full text width)."""
    models = ["naive"] + [m for m in FM if sel(rows, "ercot", 24, m)]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.45))
    w = 0.84 / len(models)
    for ax, key, title, yl in [(axes[0], "ql", "a  Accuracy", "Quantile loss"),
                               (axes[1], "cov80", "b  Calibration", "Coverage of 80% interval (%)")]:
        for i, m in enumerate(models):
            hs = [mean(rows, ds, 24, m, key) * (100 if key == "cov80" else 1) for ds in DS]
            x = np.arange(len(DS)) + (i - (len(models) - 1) / 2) * w
            figstyle.bars(ax, x, hs, w * 0.92, *bar_style(m), labels=False)
        if key == "cov80":
            ax.axhline(80, color=figstyle.INK, lw=0.8, ls=(0, (3, 2)), zorder=1)
            ax.set_ylim(0, 100)
        else:
            ax.set_ylim(0, 0.4)
        _group_axes(ax)
        ax.set_ylabel(yl)
        ax.set_title(title)
    handles = [figstyle.patch(*bar_style(m), LEGEND[m]) for m in models] + [_dashed("Nominal coverage (80%)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.98), fontsize=6.8,
               handlelength=1.3, handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(FIG, "accuracy.pdf"))
    plt.close(fig)


def fig_dependence(rows, pw):
    """Fig. 4: rank dependence of the paths (a), shuffle gain on the same forecasts (b), and the shuffle test on
    synthetic forecasts with known dependence (c) (full text width)."""
    from matplotlib.lines import Line2D
    pathm = [m for m in ["naive", "chronos", "chronos_base", "toto", "sundial", "aurora"] if sel(rows, "ercot", 24, m)]
    fig = plt.figure(figsize=(TEXT_W, 4.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.95], hspace=0.55, wspace=0.25,
                          top=0.84, bottom=0.11, left=0.075, right=0.995)
    w = 0.84 / len(pathm)
    ax = fig.add_subplot(gs[0, 0])
    for i, m in enumerate(pathm):
        x = np.arange(len(DS)) + (i - (len(pathm) - 1) / 2) * w
        hm = [mean(rows, ds, 24, m, "rho_model") for ds in DS]
        ho = [mean(rows, ds, 24, m, "rho_obs") for ds in DS]
        figstyle.bars(ax, x, hm, w * 0.92, *bar_style(m), labels=False)
        ax.scatter(x, ho, marker="_", s=70, color=figstyle.INK, linewidths=1.5, zorder=5)
    ax.set_ylim(0, 1.05)
    _group_axes(ax)
    ax.set_ylabel("Rank correlation")
    ax.set_title("a  Dependence of adjacent steps")
    ax = fig.add_subplot(gs[0, 1])
    top = 25
    rng = np.random.default_rng(3)
    for i, m in enumerate(pathm):
        x = np.arange(len(DS)) + (i - (len(pathm) - 1) / 2) * w
        g, lo, hi = [], [], []
        for ds in DS:
            vs, vss = win(pw, ds, 24, m, "vs"), win(pw, ds, 24, m, "vs_shuf")
            blocks = pw[f"{ds}|24|sid"].astype(np.int64) * 100000 + pw[f"{ds}|24|block"].astype(np.int64)
            ub, inv = np.unique(blocks, return_inverse=True)
            sa, sb = np.bincount(inv, vss, len(ub)), np.bincount(inv, vs, len(ub))
            d = rng.integers(0, len(ub), (2000, len(ub)))
            r = 100 * (sa[d].sum(1) / sb[d].sum(1) - 1)
            g.append(100 * (vss.mean() / vs.mean() - 1))
            lo.append(g[-1] - np.percentile(r, 2.5))
            hi.append(np.percentile(r, 97.5) - g[-1])
        fill, hatch = bar_style(m)
        ax.bar(x, np.minimum(g, top), w * 0.92, color=fill, edgecolor=figstyle.INK, linewidth=figstyle.FRAME,
               hatch=hatch, zorder=2)
        ax.errorbar(x, g, yerr=[lo, hi], fmt="none", ecolor=figstyle.INK, elinewidth=figstyle.FRAME, capsize=1.5,
                    zorder=3)
        for xi, v in zip(x, g):
            if v > top:                                   # bar cut at the top of the axis: print its value
                ax.text(xi, top - 0.8, f"{v:.0f}", ha="center", va="top", fontsize=6, color="white", zorder=4)
    ax.axhline(0, color=figstyle.INK, lw=0.6)
    ax.set_ylim(-5, top)
    _group_axes(ax)
    ax.set_ylabel("Shuffle gain (%)")
    ax.set_title("b  Shuffle gain on the same forecasts")
    sub = gs[1, :].subgridspec(1, 2, width_ratios=[0.25, 0.75], wspace=0)
    ax = fig.add_subplot(sub[0, 1])
    sc = json.load(open(os.path.join(RES, "shuffle_check.json")))
    y = np.arange(len(sc))[::-1]
    gains = [c["gain"] for c in sc]
    ax.barh(y, gains, 0.6, color=figstyle.lighter(figstyle.GREY, 0.3), edgecolor=figstyle.INK,
            linewidth=figstyle.FRAME, zorder=2)
    ax.errorbar(gains, y, xerr=[[c["gain"] - c["lo"] for c in sc], [c["hi"] - c["gain"] for c in sc]], fmt="none",
                ecolor=figstyle.INK, elinewidth=figstyle.FRAME, capsize=1.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([c["case"] for c in sc], fontsize=6.8)
    ax.axvline(0, color=figstyle.INK, lw=0.6)
    ax.grid(axis="x", color=figstyle.GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Shuffle gain (%)")
    ax.set_title("c  Shuffle test on synthetic forecasts with known dependence")
    handles = [figstyle.patch(*bar_style(m), LEGEND[m]) for m in pathm] + [
        Line2D([0], [0], color=figstyle.INK, lw=0, marker="_", ms=9, mew=1.5, label="Observed dependence (PIT)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=6.8,
               handlelength=1.3, handletextpad=0.4, columnspacing=1.2)
    fig.savefig(os.path.join(FIG, "dependence.pdf"))
    plt.close(fig)


def fig_horizon2(rows):
    """Fig. 5: quantile loss by forecast day at H = 720 (top) and coverage of 80% intervals against the
    horizon (bottom) (full text width)."""
    models = [m for m in ["naive", "chronos", "toto", "timesfm", "bolt", "chronos2", "sundial", "aurora"]]
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_W, 3.9))
    for c, ds in enumerate(DS):
        ax = axes[0, c]
        for m in models:
            rr = sel(rows, ds, 720, m)
            if not rr:
                continue
            v = np.array(rr[0]["lead_ql"])
            day = 24 if ds != "ausel" else 48
            v = v[: (len(v) // day) * day].reshape(-1, day).mean(1)
            ax.plot(np.arange(1, len(v) + 1), v, color=bar_style(m)[0], lw=1.1, label=SHORT[m],
                    ls="--" if m == "naive" else "-")
        ax.set_title(f"{'abc'[c]}  {DS[ds]}")
        ax.set_xlabel("Forecast day")
        if c == 0:
            ax.set_ylabel("Quantile loss")
        ax = axes[1, c]
        for m in models:
            hs = [(h, 100 * mean(rows, ds, h, m, "cov80")) for h in (24, 192, 720) if sel(rows, ds, h, m)]
            if len(hs) < 2:
                continue
            ax.plot([np.log(h) for h, _ in hs], [v for _, v in hs], marker="o", ms=3, color=bar_style(m)[0], lw=1.1,
                    ls="--" if m == "naive" else "-", label=SHORT[m])
        ax.axhline(80, color=figstyle.INK, lw=0.7, ls=(0, (3, 2)))
        ax.set_xticks([np.log(24), np.log(192), np.log(720)])
        ax.set_xticklabels(["24", "192", "720"])
        ax.set_ylim(20, 95)
        ax.set_xlabel("Horizon (steps)")
        ax.set_title(f"{'def'[c]}  {DS[ds]}")
        if c == 0:
            ax.set_ylabel("Coverage of 80% interval (%)")
    h_, l_ = axes[1, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=8, bbox_to_anchor=(0.5, 0.985), fontsize=6.8, columnspacing=1.0,
               handlelength=1.8, handletextpad=0.4)
    fig.tight_layout(h_pad=0.9, w_pad=1.2)
    fig.savefig(os.path.join(FIG, "horizon.pdf"))
    plt.close(fig)


def fig_leakage(out):
    """Fig. 6: quantile loss relative to Chronos-Bolt on the 2002-2015 and the 2026 Australian data (one column)."""
    names = {"timesfm": "TimesFM\n2.5", "aurora": "Aurora", "toto": "Toto", "sundial": "Sundial", "chronos2": "Chronos-2",
             "bolt_base": "Chronos-\nBolt base", "chronos": "Chronos", "naive": "Naive"}
    order = [m for m in ["timesfm", "aurora", "toto", "sundial", "chronos2", "bolt_base", "chronos", "naive"] if m in out]
    fig, ax = plt.subplots(figsize=(COL_W, 2.25))
    for i, m in enumerate(order):
        old, new, _ = out[m]
        for j, (val, col, mk) in enumerate([(old, figstyle.GREY, "o"), (new, figstyle.INK, "s")]):
            ax.errorbar([i + (j - 0.5) * 0.3], [val[0]], yerr=[[val[0] - val[1]], [val[2] - val[0]]], fmt=mk, ms=3.6,
                        color=col, ecolor=col, elinewidth=0.8, capsize=1.5,
                        label=(["2002–2015 series (Monash)", "2026 data (AEMO)"][j] if i == 0 else None))
    ax.axhline(1, color=figstyle.INK, lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([names[m] for m in order], fontsize=6.5)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylabel("QL relative to Chronos-Bolt")
    ax.legend(loc="upper left", fontsize=6.5, handletextpad=0.3, borderaxespad=0.3)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(FIG, "leakage.pdf"))
    plt.close(fig)


def main():
    rows, pw = load()
    small = json.load(open(os.path.join(RES, "robust.json")))
    table_main(rows, small)
    out = table_fresh(rows, pw)
    fig_accuracy(rows)
    fig_dependence(rows, pw)
    fig_horizon2(rows)
    fig_leakage(out)
    table_all(rows)
    table_long(rows)
    table_ci(rows, pw)
    table_dep(rows, pw)
    table_sundial(rows)
    summary(rows)


if __name__ == "__main__":
    main()
