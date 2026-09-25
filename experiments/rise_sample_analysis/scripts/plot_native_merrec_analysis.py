"""Publication figures for the matched native A versus full RISE E comparison."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_component_analysis import STUDY, OUTPUT, COLORS, NEUTRAL, LENGTHS, rows, row_lookup, style, save, effect_arrays


def main():
    source = STUDY / "results/group_analysis_native_v1"
    complete = json.loads((source / "complete.json").read_text())
    if not complete["identity_verified"] or complete["rows"] != 11_599_872:
        raise ValueError("Completed same-sample native analysis required")
    lookup = row_lookup(rows(source / "group_effects.csv"))
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "text.color": NEUTRAL})
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    estimate, errors, records = effect_arrays(lookup, "effective_length", LENGTHS, "E-A")
    ax.errorbar(np.arange(5), estimate, yerr=errors, fmt="o", color=COLORS["E-B"],
                markersize=5, capsize=3, linewidth=1)
    shares = [int(row["rows"]) / complete["rows"] * 100 for row in records]
    ax.set_xticks(np.arange(5), [f"{group}\n({share:.2f}% of rows)" for group, share in zip(LENGTHS, shares)])
    ax.set_xlabel("Effective history length (candidate excluded)")
    ax.set_ylabel("Label-macro ΔAUC (percentage points)")
    ax.set_title("Full RISE − Native baseline by history length", loc="left", pad=13)
    style(ax)
    fig.text(0.01, -0.075, "HyFormer–MerRec, matched seed 20262028; identical 11,599,872 retained test rows.\n"
             "Nominal 95% user-cluster intervals, trained models fixed; the empty-history interval crosses zero.", fontsize=8)
    fig.tight_layout()
    save(fig, "native_merrec_effect_by_history")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.7), sharey=True)
    specs = [
        ("item_train_target_frequency", ("train_target_unseen", "positive_le_1", "gt_1_le_4", "gt_4"),
         ("0", "1", "2–4", "≥5"), "Item training-target frequency"),
        ("user_train_target_frequency", ("train_target_unseen", "positive_le_14", "gt_14_le_80", "gt_80"),
         ("0", "1–14", "15–80", "≥81"), "User training-target frequency"),
    ]
    for ax, (dimension, groups, labels, title) in zip(axes, specs):
        estimate, errors, records = effect_arrays(lookup, dimension, groups, "E-A")
        ax.errorbar(np.arange(4), estimate, yerr=errors, fmt="o", color=COLORS["E-B"],
                    markersize=5, capsize=3, linewidth=1)
        shares = [int(row["rows"]) / complete["rows"] * 100 for row in records]
        ax.set_xticks(np.arange(4), [f"{label}\n({share:.1f}%)" for label, share in zip(labels, shares)])
        ax.set_xlabel(title)
        ax.set_title("Full RISE − Native baseline", loc="left", fontsize=10, pad=12)
        style(ax)
    axes[0].set_ylabel("Label-macro ΔAUC (percentage points)")
    axes[0].set_ylim(bottom=0)
    fig.text(0.01, -0.075, "HyFormer–MerRec; nominal 95% user-cluster intervals. Parentheses: share of retained test rows.\n"
             "Cuts use positive-frequency training entities; 0 means unseen as a training target, not a strict cold-start claim.", fontsize=8)
    fig.tight_layout(w_pad=2)
    save(fig, "native_merrec_effect_by_frequency")
    manifest = {"comparison": "Full RISE E minus independently trained native baseline A",
                "figure_names": ["native_merrec_effect_by_history", "native_merrec_effect_by_frequency"],
                "source_sha256": {str(p.relative_to(STUDY)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (source / "group_effects.csv", source / "complete.json",
                                            Path(__file__).resolve(), Path(__file__).with_name("plot_component_analysis.py"))},
                "visual_qa": "pending exported-image inspection"}
    (OUTPUT / "native_merrec_figures_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
