"""Paper figures and a TeX fragment from accepted CSVs; isolated uv figure_env."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
RESULTS, OUTPUT = STUDY / "results", STUDY / "figures"
BLUE, GOLD, GRAY, INK = "#365F85", "#B77932", "#686B70", "#30343A"
SOURCES, FIGURES = {}, []


def read(name):
    path = RESULTS / name
    SOURCES[str(path.relative_to(STUDY))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def verify(name, csv_name=None, hash_key=None):
    path = RESULTS / name
    value = json.loads(path.read_text())
    SOURCES[str(path.relative_to(STUDY))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if csv_name and hashlib.sha256((RESULTS / csv_name).read_bytes()).hexdigest() != value[hash_key]:
        raise ValueError("Accepted source checksum changed")
    return value


def style(ax, grid="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999999")
    ax.grid(axis=grid, color="#DEDEDE", linewidth=.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5)


def save(fig, name, contract):
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUTPUT / f"{name}.{extension}", bbox_inches="tight", dpi=250, facecolor="white")
    FIGURES.append({"name": name, **contract})
    plt.close(fig)


def macro(values, **criteria):
    selected = [row for row in values if row["label"] == "__macro__" and all(row.get(key) == value for key, value in criteria.items())]
    if len(selected) != 1:
        raise ValueError(f"Expected one accepted macro record: {criteria}")
    return selected[0]


def replication_discussion(values):
    zh = macro(values, model="Zenith", mode="history_only")
    zt = macro(values, model="Zenith", mode="token_only")
    hq = macro(values, model="HyFormer", mode="query_only")
    assert float(zh["delta_AUC_vs_Baseline"]) > 0 > float(zt["delta_AUC_vs_Baseline"])
    assert float(zh["Full_RISE_AUC"]) > float(zh["Baseline_AUC"])
    assert float(zh["Full_RISE_logloss"]) < float(zh["Placement_logloss"])
    for row in values:
        if row["model"] == "HyFormer" and row["label"] in ("follow", "share"):
            assert float(row["Placement_AUC"]) < float(row["Baseline_AUC"])
            assert float(row["Full_RISE_AUC"]) < float(row["Baseline_AUC"])
    gap = float(hq["Full_RISE_AUC"]) - float(hq["Placement_AUC"])
    return ("\n\\paragraph{Cross-backbone interpretation.}\n"
        "On Zenith, history-only insertion improves macro AUC over native A, whereas\n"
        "token-only insertion reduces it. Full RISE remains above native A and has\n"
        f"lower macro Logloss than history-only ({float(zh['Full_RISE_logloss']):.8f} versus {float(zh['Placement_logloss']):.8f}),\n"
        "so the AUC-best placement is not uniformly best across metrics. On HyFormer,\n"
        f"full RISE has the highest macro AUC, but its margin over query-only is just {gap:.8f};\n"
        "follow and share remain below native A under full, sequence-only, and\n"
        "query-only insertion. HyFormer's sequence encoder and query-to-history\n"
        "decoder are distinct sequence-processing stages, not field-token\n"
        "ablation counterparts. These one-seed comparisons support architecture-\n"
        "and metric-dependent placement choices, without establishing seed-stable\n"
        "superiority for a universal insertion rule.\n")


def main():
    OUTPUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
        "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.labelcolor": INK, "text.color": INK, "axes.titlesize": 10})
    targeted = read("targeted_training_comparisons.csv")
    new = read("new_training_comparisons.csv")
    target = macro(targeted, model="HiFormer", dataset="QK_Video_Action", mode="target_only")
    field = macro(targeted, model="HiFormer", dataset="QK_Video_Action", mode="field_only")
    values = [float(target["Baseline_AUC"]), float(target["Full_RISE_AUC"]),
              float(target["Targeted_AUC"]), float(field["Targeted_AUC"])]
    fig, ax = plt.subplots(figsize=(5.5, 3.25))
    for y, (value, color, marker, fill) in enumerate(zip(values, [GRAY, BLUE, GOLD, BLUE], ["o", "o", "^", "s"], [GRAY, BLUE, GOLD, "white"])):
        ax.scatter(value, y, s=44, marker=marker, edgecolors=color, facecolors=fill, linewidths=1.2, zorder=3)
        ax.annotate(f"{value:.6f}", (value, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9)
    span = max(values) - min(values)
    ax.set_xlim(min(values) - .12 * span, max(values) + .5 * span)
    ax.set_ylim(3.5, -.5)
    ax.set_yticks(range(4), ["Native (A)", "Full RISE (E)", "Target-only", "Field-only"])
    ax.axvline(values[0], color=GRAY, linestyle="--", linewidth=.8)
    ax.set_xlabel("Label-macro ROC-AUC (zoomed axis)")
    style(ax)
    fig.suptitle("HiFormer–QK-Video: insertion-site comparison", x=.02, ha="left", fontsize=11)
    fig.text(.02, .025, "Matched seed 20262028; one epoch; four action labels.\nPoint estimates only: no training-seed uncertainty intervals.", fontsize=8)
    fig.subplots_adjust(left=.22, right=.98, top=.83, bottom=.24)
    save(fig, "extended_hiformer_placement", {"family": "categorical dot", "data_rows": 4,
         "uncertainty": "none; one matched training seed", "auc_values": values})

    lengths = ["0", "1-20", "21-50", "51-99", "100"]
    group_records = {}
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 4.1), sharex=True)
    for ax, dataset, title in zip(axes, ["QK_Video_Action", "TencentGR_10M_Action"], ["QK-Video · 4 actions", "TAAC-25 · 2 actions"]):
        folder = f"hiformer_{dataset}_groups_v1"
        receipt = verify(f"{folder}/complete.json", f"{folder}/group_results.csv", "group_results_sha256")
        if not receipt["identity_verified"]:
            raise ValueError("Unverified grouped identities")
        selected = [r for r in read(f"{folder}/group_results.csv") if r["dimension"] == "effective_length" and r["label"].startswith("macro_")]
        assert sum(int(r["rows"]) for r in selected) == receipt["rows"]
        group_records[dataset] = {r["group"]: r for r in selected}
        ticks = []
        for y, group in enumerate(lengths):
            row = group_records[dataset].get(group)
            if row:
                center, lo, hi = [100 * float(row[key]) for key in ("delta_auc", "auc_ci_lower", "auc_ci_upper")]
                assert lo <= center <= hi
                ax.errorbar(center, y, xerr=[[center-lo], [hi-center]], fmt="o", color=BLUE, capsize=3, markersize=4.5, linewidth=1)
                ticks.append(f"{group}\n(n={int(row['rows']):,})")
            else:
                ticks.append(f"{group}\n(n=0)")
                ax.text(.04, y, "no retained rows", transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color=GRAY)
        ax.set_yticks(range(5), ticks)
        ax.set_ylim(4.6, -.6)
        ax.set_xlim(-2.05, 2.8)
        ax.axvline(0, color=INK, linewidth=.9)
        ax.set_title(title, loc="left", pad=8)
        ax.set_xlabel("Macro ΔAUC: E − A (percentage points)")
        style(ax)
    fig.suptitle("HiFormer: native-controlled effects by effective history length", x=.01, ha="left", fontsize=11)
    fig.text(.01, .012, "Seed 20262028; identical paired rows within each dataset. Nominal 95% user-cluster intervals;\ntrained models fixed, no multiplicity adjustment or across-seed uncertainty. Empty bins have no point.", fontsize=8)
    fig.subplots_adjust(left=.13, right=.99, top=.82, bottom=.23, wspace=.64)
    save(fig, "extended_hiformer_history_forest", {"family": "faceted dot and interval", "data_rows": 8,
         "uncertainty": "nominal 95% user-cluster linearized intervals; fixed trained models", "empty_bins_retained": True})

    costs = read("extended_benchmark_v1/cost_measurements.csv")
    receipt = verify("extended_benchmark_v1/complete.json")
    assert receipt["state"] == "COMPLETED" and receipt["same_gpu_uuid"] and receipt["all_raw_blocks_retained"] and len(costs) == 18
    colors, names = {"A": GRAY, "B": GOLD, "E": BLUE}, {"A": "Native A (Flash)", "B": "Content B (efficient)", "E": "Full E (efficient)"}
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8))
    for bi, batch in enumerate((64, 256, 512)):
        for vi, variant in enumerate(("A", "B", "E")):
            for oi, order in enumerate(("forward01", "reverse01")):
                selected = [r for r in costs if int(r["batch"]) == batch and r["variant"] == variant and r["order"] == order]
                assert len(selected) == 1
                row = selected[0]
                assert row["kernel"] == ("flash" if variant == "A" else "efficient")
                x = bi + (vi-1) * .24 + (oi-.5) * .08
                y, lo, hi = [float(row[key]) for key in ("median_ms", "q25_ms", "q75_ms")]
                face = colors[variant] if oi == 0 else "white"
                marker = "o" if oi == 0 else "s"
                axes[0].errorbar(x, y, yerr=[[y-lo], [hi-y]], fmt=marker, color=colors[variant], mfc=face, mec=colors[variant], markersize=4.5, capsize=2, linewidth=.8)
                axes[1].scatter(x, float(row["incremental_peak_MiB"]), marker=marker, s=22, edgecolors=colors[variant], facecolors=face, linewidths=1)
    for ax in axes:
        ax.set_xticks(range(3), ["64", "256", "512"])
        ax.set_xlim(-.5, 2.5)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Resident input batch size")
        style(ax, "y")
    axes[0].set_ylabel("Model forward latency (ms)")
    axes[1].set_ylabel("Peak above resident baseline (MiB)")
    handles = [plt.Line2D([], [], marker="o", color=colors[v], linestyle="none", label=names[v]) for v in ("A", "B", "E")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .91), ncol=3, frameon=False, fontsize=8)
    fig.suptitle("HyFormer–MerRec: compiled BF16 inference on one H100", x=.01, ha="left", fontsize=11)
    fig.text(.01, .012, "Filled circles: A→B→E order; open squares: E→B→A. Whiskers: IQR of 10 block means, not confidence intervals.\n1,000 warm-ups; 10×100 timed forwards. GPU-resident inputs; load/compile/H2D excluded. Both rounds retained.", fontsize=7.7)
    fig.subplots_adjust(left=.10, right=.99, top=.72, bottom=.24, wspace=.37)
    save(fig, "extended_inference_cost", {"family": "grouped dot and interval", "data_rows": 18,
         "uncertainty": "within-round block IQR; both execution orders shown", "memory": "peak allocated minus resident baseline; not total allocation"})

    receipt = verify("compiled_mechanism_v1/complete.json")
    assert receipt["accepted"] and receipt["examples"] == 512
    mechanism = read("compiled_mechanism_v1/site_quantiles.csv")
    expected_hash = receipt["artifact_sha256"]["site_quantiles.csv"]
    assert SOURCES["results/compiled_mechanism_v1/site_quantiles.csv"] == expected_hash
    sites = [f"unified_interaction_layers.{kind}_layers.{i}" for i in range(3) for kind in ("seq", "decode")]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for variant, offset, color, marker in (("B", -.13, GOLD, "s"), ("E", .13, BLUE, "o")):
        for i, site in enumerate(sites):
            row = next(r for r in mechanism if r["variant"] == variant and r["site"] == site and r["metric"] == "tv_mean")
            x, mean, lo, hi = i+offset, float(row["mean"]), float(row["q10"]), float(row["q90"])
            ax.vlines(x, lo, hi, color=color, linewidth=1.2)
            ax.hlines([lo, hi], x-.045, x+.045, color=color, linewidth=1)
            ax.scatter(x, mean, color=color, marker=marker, s=28, label=("Content-only B" if variant == "B" else "Full RISE E") if i == 0 else None, zorder=3)
    ax.set_xticks(range(6), ["Seq. 1", "Decode 1", "Seq. 2", "Decode 2", "Seq. 3", "Decode 3"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Total variation distance")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    style(ax, "y")
    fig.suptitle("Compiled selection redistribution across attention sites", x=.01, ha="left", fontsize=11)
    fig.text(.01, .895, "HyFormer–MerRec; fixed 512 interactions / 455 users; matched seed 20262028", fontsize=8.5)
    fig.text(.01, .015, "Dots: equal-example means; whiskers: sample 10th–90th percentiles, not confidence intervals.\nActual compiled Q/K/Δ; prediction identity verified. Sequence-site means exclude one empty-history row.", fontsize=8)
    fig.subplots_adjust(left=.09, right=.99, top=.80, bottom=.23)
    save(fig, "extended_compiled_site_tv", {"family": "grouped descriptive distribution", "data_rows": 12,
         "uncertainty": "sample 10–90 percentiles; no inferential confidence intervals", "population_representative": False,
         "selection_only": True, "internal_AM_PA_observation_claim": False})

    local, fixed = [macro(new, model="RankMixer", dataset="KuaiRand_Video_Action", setting=s) for s in ("local", "fixed")]
    hyfixed = macro(targeted, model="HyFormer", dataset="MerRec_Action", mode="fixed_cumulative")
    replication_path = RESULTS / "site_replication_comparisons.csv"
    replication = read("site_replication_comparisons.csv") if replication_path.exists() else []
    rep_rows = [r for r in replication if r["label"] == "__macro__"]
    rep_table = ""
    if rep_rows:
        rep_table = "\n\\begin{table}[t]\n\\centering\\small\n\\caption{Additional QK-Video placement runs, all at seed 20262028. Native A is the first numerical column. No across-seed intervals are estimated.}\n\\label{tab:extended_cross_placement}\n\\begin{tabular}{llrrr}\\toprule\nModel & Placement & Native A & Full E & Selective \\\\\n\\midrule\n"
        for r in rep_rows:
            mode = r["mode"].replace("_", "\\_")
            rep_table += f"{r['model']} & {mode} & {float(r['Baseline_AUC']):.7f} & {float(r['Full_RISE_AUC']):.7f} & {float(r['Placement_AUC']):.7f} \\\\\n"
        rep_table += "\\bottomrule\\end{tabular}\n\\end{table}\n"
    qk = group_records["QK_Video_Action"]
    taac = group_records["TencentGR_10M_Action"]
    ratio_values = [float(r["E_over_A"]) for r in costs if r["variant"] == "E"]
    incremental = {(int(r["batch"]), r["variant"]): float(r["incremental_peak_MiB"]) for r in costs}
    tex = r"""% Generated by plot_extended_results.py from accepted CSVs.
% Requires graphicx, booktabs, amsmath. Put the study figures directory on graphicspath.
% Insert after the matched native/component analyses; no external hyperlinks.
\subsection{Placement, input-independent controls, and realized cost}
\label{sec:rise_extended_analyses}

\paragraph{Insertion-site sensitivity.}
We retain the matched seed-20262028, one-epoch protocol and compare the native
backbone, full RISE, and selective-site variants. On HiFormer--QK-Video,
macro ROC-AUC is @@A@@ for native A, @@E@@ for full RISE, @@T@@ for
target-only insertion, and @@F@@ for field-only insertion
(Figure~\ref{fig:extended_placement}). The target-only variant improves over
native A by @@TD@@ and full RISE by @@TE@@ in absolute AUC, whereas the
field-only variant is below both. This identifies insertion layout as a
plausible contributor to the observed regression; it does not isolate a
unique temporal mechanism. Selective-site variants have different auxiliary
parameter counts, so this comparison is not capacity-matched. All results are
single-seed point estimates. Additional QK-Video placement results are retained
in Table~\ref{tab:extended_cross_placement}, rather than selecting only the
best-performing placement for each backbone.

\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{extended_hiformer_placement.pdf}
\caption{HiFormer--QK-Video placement comparison. Macro AUC equally weights
four actions. The displayed AUC axis is zoomed; there are no training-seed
uncertainty intervals.}
\label{fig:extended_placement}
\end{figure}

\paragraph{Native-controlled history groups.}
The grouped comparison uses @@QKN@@ identical paired QK-Video rows and
@@TAACN@@ TAAC-25 rows. Full-minus-native macro AUC is negative in all four
observed QK-Video length groups, with nominal intervals below zero. On TAAC-25,
the 1--20 estimate is @@SHORT@@ with interval $[@@SHORTLO@@,@@SHORTHI@@]$;
the estimates at lengths 21--50 and 51--99 are @@MID@@ and @@LONG@@.
Empty-history TAAC-25 has a wide interval containing zero; QK-Video has no
empty-history rows, and TAAC-25 has no length-100 rows in the retained test set.
Figure~\ref{fig:extended_history} retains these empty categories explicitly.
The intervals are normal 95\% user-cluster linearized intervals conditional on
fixed trained models. Macro influences are averaged across actions before
variance estimation. They contain neither retraining uncertainty nor a
multiplicity adjustment; differences in whether separate intervals cross zero
do not establish a between-group interaction. Grouped AUCs are not an additive
partition of overall AUC because cross-group positive--negative pairs remain.

\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{extended_hiformer_history_forest.pdf}
\caption{Full RISE minus independently trained native HiFormer by effective
history length. Differences are percentage points of label-macro AUC; QK-Video
and TAAC-25 have four and two actions, respectively. Error bars are nominal
95\% user-cluster intervals for fixed models, not across-seed intervals.}
\label{fig:extended_history}
\end{figure*}

\paragraph{A qualified case for cumulative and adaptive controls.}
On RankMixer--KuaiRand, the equal-active-parameter local control obtains
@@LOCAL@@ macro AUC, compared with @@RANKE@@ for full RISE, a difference of
@@LOCALD@@. The local construction also uses a fixed zero centering gauge;
without a successful matching control-state audit, this comparison cannot
attribute every difference solely to accumulation under active clipping.
The learned input-independent cumulative control instead obtains @@RANKFIX@@,
slightly above full RISE in point estimate. On HyFormer--MerRec, the same
input-independent principle obtains @@HYFIX@@, below full RISE at @@HYE@@
but above native A at @@HYA@@. Thus, these experiments do not support a
universal necessity claim for input-dependent controls. The fixed-control
comparisons also change active parameter count and initialization; their
single-seed point differences establish neither statistical equivalence nor
a capacity-controlled causal effect of adaptivity.

\paragraph{Compiled selection observations.}
On a fixed diagnostic sample of 512 test interactions from 455 users,
instrumented and uninstrumented compiled predictions are exactly identical
for both checkpoints and all five actions. We reconstruct pre-dropout selection
from actual captured queries, keys, and auxiliary corrections while retaining
native padding semantics. Figure~\ref{fig:extended_tv} shows equal-example
mean total variation distance and sample percentiles at six sites. The full
model's first sequence site has mean TV @@TVE@@, compared with @@TVB@@ for
Content-only B; the first decoding site instead has mean TV @@TVDE@@ versus
@@TVDB@@. These are within-model conditional changes, not native-backbone
performance effects or a causal decomposition between separately trained
models. Sample percentile whiskers are not confidence intervals. The sampling
window is not population-representative. No successful observation of fused
internal AM/PA states is claimed; unvalidated eager-state reconstructions are
excluded from the interpretation.

\begin{figure*}[t]
\centering\includegraphics[width=.94\textwidth]{extended_compiled_site_tv.pdf}
\caption{Selection redistribution in the compiled HyFormer--MerRec
checkpoints. B retains auxiliary content projections and is not native A.
Dots are means over the fixed diagnostic interactions; whiskers are their
10th--90th percentiles, not inferential intervals. One empty-history example
has no valid sequence query and is excluded from sequence-site means.}
\label{fig:extended_tv}
\end{figure*}

\paragraph{Measured cost and attention backend.}
We benchmark native A, Content-only B, and full RISE E on one H100 using the
same cached real inputs at batches 64, 256, and 512. Each variant runs in a
fresh process with compiled BF16 inference, 1,000 warm-up forwards, and ten
blocks of 100 timed forwards. Both execution orders, A--B--E and E--B--A,
are retained. Across the six batch/order rounds, E/A forward-latency ratios
range from @@RMIN@@ to @@RMAX@@. Profiling identifies a Flash SDPA path for
native A and an efficient-attention path for B/E with explicit additive score
bias. This backend difference is part of the measured implementation cost,
not evidence that every mathematically equivalent implementation has the same
overhead. At batch 512, peak allocated memory above the resident baseline is
@@MEM_A@@ MiB for A and @@MEM_E@@ MiB for E; these quantities are not total
model memory. Figure~\ref{fig:extended_cost} shows both orders and within-round
block IQRs, without selecting the faster round. The benchmark excludes loading,
compilation, input transfer, and the data pipeline; it is neither serving
tail latency nor end-to-end throughput. Low auxiliary dimension therefore does
not by itself establish negligible realized inference cost.

\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{extended_inference_cost.pdf}
\caption{GPU-resident model-forward cost on the same H100. Filled circles
and open squares retain the forward and reverse execution orders. Latency
whiskers show the IQR of ten block means, not a confidence interval. Memory
is incremental peak allocated above the warmed resident baseline.}
\label{fig:extended_cost}
\end{figure*}

\paragraph{Data and inferential boundaries.}
The new comparisons remain single-seed, fixed-budget offline experiments.
History length, activity, frequency, and label difficulty are correlated;
descriptive strata do not identify their independent effects. For MerRec,
the processed historical action encoding can contain later-attributed outcomes,
while original action-event timestamps were unavailable in the scoped audit.
Prediction-time availability and cross-split attribution rates therefore
remain unresolved. These comparisons characterize the supplied offline
encoding; they do not establish a leakage rate or certify online suitability.
"""
    tv = {(r["variant"], r["site"]): float(r["mean"]) for r in mechanism if r["metric"] == "tv_mean"}
    tokens = {"A": f"{values[0]:.7f}", "E": f"{values[1]:.7f}", "T": f"{values[2]:.7f}", "F": f"{values[3]:.7f}",
        "TD": f"{float(target['delta_AUC_vs_Baseline']):+.7f}", "TE": f"{float(target['delta_AUC_vs_Full_RISE']):+.7f}",
        "QKN": f"{sum(int(r['rows']) for r in qk.values()):,}", "TAACN": f"{sum(int(r['rows']) for r in taac.values()):,}",
        "SHORT": f"{float(taac['1-20']['delta_auc']):+.6f}", "SHORTLO": f"{float(taac['1-20']['auc_ci_lower']):+.6f}",
        "SHORTHI": f"{float(taac['1-20']['auc_ci_upper']):+.6f}", "MID": f"{float(taac['21-50']['delta_auc']):+.6f}", "LONG": f"{float(taac['51-99']['delta_auc']):+.6f}",
        "LOCAL": f"{float(local['new_run_auc']):.7f}", "RANKE": f"{float(local['full_E_auc']):.7f}", "LOCALD": f"{float(local['delta_auc_E_minus_new']):+.7f}",
        "RANKFIX": f"{float(fixed['new_run_auc']):.7f}", "HYFIX": f"{float(hyfixed['Targeted_AUC']):.7f}", "HYE": f"{float(hyfixed['Full_RISE_AUC']):.7f}", "HYA": f"{float(hyfixed['Baseline_AUC']):.7f}",
        "TVE": f"{tv['E',sites[0]]:.3f}", "TVB": f"{tv['B',sites[0]]:.3f}", "TVDE": f"{tv['E',sites[1]]:.3f}", "TVDB": f"{tv['B',sites[1]]:.3f}",
        "RMIN": f"{min(ratio_values):.2f}", "RMAX": f"{max(ratio_values):.2f}", "MEM_A": f"{incremental[512,'A']:.2f}", "MEM_E": f"{incremental[512,'E']:.2f}"}
    for key, value in tokens.items():
        tex = tex.replace("@@"+key+"@@", value)
    assert "@@" not in tex
    (STUDY / "paper/extended_analyses.tex").write_text(tex + rep_table + (replication_discussion(replication) if rep_rows else ""))
    manifest = {"renderer": f"Matplotlib {matplotlib.__version__} in isolated uv figure_env",
        "palette_policy": "two nonneutral roots (blue/gold) plus neutral native reference; marker/fill distinctions",
        "figures": FIGURES, "source_sha256": SOURCES,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tex_sha256": hashlib.sha256((STUDY / "paper/extended_analyses.tex").read_bytes()).hexdigest(),
        "cross_model_macro_rows": len(rep_rows), "visual_qa": "recorded separately in notes/extended_figures_validation.md"}
    (OUTPUT / "extended_figures_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"figures": [entry['name'] for entry in FIGURES], "cross_model_rows": len(rep_rows)}))


if __name__ == "__main__":
    main()
