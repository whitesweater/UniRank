"""Export static, publication-oriented figures from verified component results.

Use the isolated uv project in figure_env; it does not alter training packages.
"""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np


STUDY = Path(__file__).resolve().parents[1]
INPUT = STUDY / "results/group_analysis_components_v1"
OUTPUT = STUDY / "figures"
COLORS = {"E-B": "#365F85", "C-B": "#B77932", "D-B": "#6E8061"}
NEUTRAL = "#30343A"
LENGTHS = ("0", "1-20", "21-50", "51-99", "100")
LABELS = ("Like", "Cart", "Offer", "Checkout", "Purchase")


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def row_lookup(values):
    return {(r["dimension"], r["group"], r["label"], r["comparison"]): r for r in values}


def style(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(colors=NEUTRAL, labelsize=9)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.axhline(0, color=NEUTRAL, linewidth=0.7)


def save(fig, name):
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUTPUT / f"{name}.{extension}", dpi=220,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def effect_arrays(lookup, dimension, groups, comparison, label="macro_5_tasks"):
    selected = [lookup[dimension, group, label, comparison] for group in groups]
    estimate = 100 * np.array([float(row["delta_auc"]) for row in selected])
    lower = 100 * np.array([float(row["auc_ci_lower"]) for row in selected])
    upper = 100 * np.array([float(row["auc_ci_upper"]) for row in selected])
    return estimate, np.vstack([estimate - lower, upper - estimate]), selected


def main():
    complete = json.loads((INPUT / "complete.json").read_text())
    if not complete["identity_verified"] or complete["rows"] != 11_599_872:
        raise ValueError("Verified complete paired result required")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    values = rows(INPUT / "group_effects.csv")
    lookup = row_lookup(values)
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                         "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.labelcolor": NEUTRAL,
                         "text.color": NEUTRAL, "axes.titlesize": 11})

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    offsets = {"C-B": -0.18, "D-B": 0, "E-B": 0.18}
    labels = {"C-B": "AM-only − B", "D-B": "PA-only − B", "E-B": "Full RISE − B"}
    markers = {"C-B": "s", "D-B": "^", "E-B": "o"}
    for comparison in ("C-B", "D-B", "E-B"):
        estimate, errors, selected = effect_arrays(lookup, "effective_length", LENGTHS, comparison)
        ax.errorbar(np.arange(5) + offsets[comparison], estimate, yerr=errors,
                    fmt=markers[comparison], linestyle="none", color=COLORS[comparison],
                    markerfacecolor=COLORS[comparison] if comparison == "E-B" else "white",
                    markersize=5, capsize=2, elinewidth=0.9, label=labels[comparison])
    shares = [int(row["rows"]) / complete["rows"] * 100 for row in selected]
    ax.set_xticks(np.arange(5), [f"{length}\n({share:.2f}% of rows)" for length, share in zip(LENGTHS, shares)])
    ax.set_xlabel("Effective history length (maximum 100; candidate excluded)")
    ax.set_ylabel("Label-macro ΔAUC (percentage points)")
    ax.set_title("Component effects by effective history length", loc="left", pad=35)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=3, fontsize=8)
    style(ax)
    fig.text(0.01, -0.06, "HyFormer–MerRec, seed 20262028; 11,599,872 paired test rows.\n"
             "Bars: 95% user-cluster linearized intervals. B = Content-only; it is not the native baseline.", fontsize=8)
    fig.tight_layout()
    save(fig, "component_effect_by_history")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.7), sharey=True)
    specifications = [
        ("item_train_target_frequency", ("train_target_unseen", "positive_le_1", "gt_1_le_4", "gt_4"),
         ("0", "1", "2–4", "≥5"), "Item training-target frequency"),
        ("user_train_target_frequency", ("train_target_unseen", "positive_le_14", "gt_14_le_80", "gt_80"),
         ("0", "1–14", "15–80", "≥81"), "User training-target frequency"),
    ]
    for ax, (dimension, groups, display, title) in zip(axes, specifications):
        estimate, errors, selected = effect_arrays(lookup, dimension, groups, "E-B")
        ax.errorbar(np.arange(4), estimate, yerr=errors, fmt="o", color=COLORS["E-B"],
                    markersize=5, capsize=3, linewidth=1)
        shares = [int(row["rows"]) / complete["rows"] * 100 for row in selected]
        ax.set_xticks(np.arange(4), [f"{label}\n({share:.1f}%)" for label, share in zip(display, shares)])
        ax.set_xlabel(title)
        ax.set_title("Full RISE − Content-only", loc="left", pad=12)
        style(ax)
    axes[0].set_ylabel("Label-macro ΔAUC (percentage points)")
    axes[0].set_ylim(bottom=0)
    fig.text(0.01, -0.07, "HyFormer–MerRec; 95% user-cluster linearized intervals. Parentheses: share of retained test rows.\n"
             "Cut points use positive-frequency training entities (Q25/Q75); 0 means unseen as a training target.", fontsize=8)
    fig.tight_layout(w_pad=2)
    save(fig, "component_effect_by_frequency")

    heat = np.array([[100 * float(lookup["effective_length", group, label, "E-B"]["delta_auc"])
                      for group in LENGTHS] for label in LABELS])
    maximum = max(1e-8, float(np.max(np.abs(heat))))
    palette = LinearSegmentedColormap.from_list("signed_blue_gold", ["#B77932", "#FAFAF8", "#365F85"])
    fig, ax = plt.subplots(figsize=(6.9, 3.7))
    artist = ax.imshow(heat, cmap=palette, norm=TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum), aspect="auto")
    for (i, j), value in np.ndenumerate(heat):
        ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=10,
                color="white" if abs(value) > 0.62 * maximum else NEUTRAL)
    ax.set_xticks(np.arange(5), LENGTHS)
    ax.set_yticks(np.arange(5), LABELS)
    ax.set_xlabel("Effective history length")
    ax.set_title("Task-level Full RISE − Content-only AUC", loc="left", pad=12)
    fig.colorbar(artist, ax=ax, label="ΔAUC (percentage points)", fraction=0.04, pad=0.04)
    fig.text(0.01, -0.015, "HyFormer–MerRec, seed 20262028. Point estimates on identical test rows; no significance stars.\n"
             "All five tasks retained; per-cell counts and user-cluster intervals are supplied in the result tables.", fontsize=8)
    fig.tight_layout()
    save(fig, "task_effect_by_history")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.7))
    for ax, label, comparison, title in (
        (axes[0], "Offer", "E-D", "Offer: Full RISE − PA-only"),
        (axes[1], "macro_5_tasks", "E-C-D+B", "Macro AUC interaction: E − C − D + B"),
    ):
        estimate, errors, _ = effect_arrays(lookup, "effective_length", LENGTHS, comparison, label)
        ax.errorbar(np.arange(5), estimate, yerr=errors, fmt="o", color=COLORS["E-B"],
                    capsize=3, linewidth=1, markersize=5)
        ax.set_xticks(np.arange(5), LENGTHS)
        ax.set_xlabel("Effective history length")
        ax.set_ylabel("AUC difference (percentage points)")
        ax.set_title(title, loc="left", fontsize=10, pad=12)
        style(ax)
    fig.text(0.01, -0.07, "HyFormer–MerRec; nominal 95% user-cluster intervals, fixed trained models.\n"
             "These trained-variant contrasts do not isolate capacity or prove a causal mechanism; interaction is on the AUC scale.", fontsize=8)
    fig.tight_layout(w_pad=2)
    save(fig, "conditional_component_contrasts")

    manifest = {"renderer": "Matplotlib 3.10.6 in isolated uv figure_env", "scope": "matched components, not native-baseline comparison",
                "figure_names": ["component_effect_by_history", "component_effect_by_frequency", "task_effect_by_history", "conditional_component_contrasts"],
                "source_sha256": {str(path.relative_to(STUDY)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (INPUT / "group_effects.csv", INPUT / "complete.json", Path(__file__).resolve())},
                "chart_contract": "Static paper figures; restrained declared colors plus marker shapes; zero reference and explicit AUC percentage-point units; intervals conditional on fixed models.",
                "visual_qa": "pending rendered inspection"}
    (OUTPUT / "component_figures_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
