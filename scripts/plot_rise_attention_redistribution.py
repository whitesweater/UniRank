"""Render audited, same-checkpoint attention redistribution as paper figures.

Chart contract (declared before implementation):
  Question: how does a fixed local score correction redistribute attention?
  Surface: standalone PDF/SVG/PNG, rendered with the existing uv figure environment.
  Matrix: four aligned strips/bars for a0, delta, a1 and a1-a0, with identical
      weight scales and a zero-centered score scale; full original key axis.
  Distribution: an empirical CDF over nonempty fixed-window samples, head 0.
  Sufficiency: at least one nonempty sample; if fewer than 20, flag the small
      diagnostic cohort in the receipt. No synthetic or interpolated samples.
  Selection: nearest to linear TV quantiles P50/P10/P90, then part/row identity;
      predictions and labels are never read by the selector.
  Palette: hard two-root cap, blue/orange plus neutrals; signed bars and a
      visible zero line, shape/line-style distinctions on the CDF.
  QA: numerical consistency checks first, then inspect exported PNGs at final
      figure size. Outputs also include source hashes, case receipts, chart map,
      exact plotted position values and a compact numerical validation receipt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np


BLUE = "#003F7A"
ORANGE = "#923B0F"
INK = "#282828"
MUTED = "#686B70"
PADDING = "#D9D9D9"
GRID = "#DEDEDE"
HEAD = 0
CASE_SPECS = (
    ("main_p50", 0.5, "Median-TV case"),
    ("appendix_p10", 0.1, "Low-TV case"),
    ("appendix_p90", 0.9, "High-TV case"),
)
ARRAY_KEYS = (
    "native_scores", "delta", "weight_before", "weight_after", "valid_mask",
    "history_item_ids", "history_actions", "target_item_ids",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, content: Any) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty evidence table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reconstruct(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """FP32 softmax over the original allowed key set; nonempty rows only."""
    masked = np.where(valid[:, None, :], scores, -np.inf).astype(np.float32)
    shifted = masked - masked.max(axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=-1, keepdims=True)


def load_evidence(input_dir: Path) -> tuple[dict, list, dict, dict, dict]:
    names = ("raw_attention.npz", "meta.csv", "stats.csv", "manifest.json")
    sources = {name: sha256(input_dir / name) for name in names}
    complete = json.loads((input_dir / "complete.json").read_text())
    if not complete.get("completed"):
        raise ValueError("Attention export is not complete")
    for key in ("prediction_bitwise_equal", "historical_cross_batch_gate_passed"):
        if complete.get("summary", {}).get(key) is not True:
            raise ValueError(f"Attention export failed required evidence gate: {key}")
    for name, digest in sources.items():
        if complete.get("files_sha256", {}).get(name) != digest:
            raise ValueError(f"Completed-export hash mismatch: {name}")
    sources["complete.json"] = sha256(input_dir / "complete.json")
    with np.load(input_dir / "raw_attention.npz", allow_pickle=False) as source:
        missing = set(ARRAY_KEYS) - set(source.files)
        if missing:
            raise ValueError(f"Missing attention arrays: {sorted(missing)}")
        arrays = {key: source[key] for key in ARRAY_KEYS}
    meta = read_csv(input_dir / "meta.csv")
    raw_stats = read_csv(input_dir / "stats.csv")
    manifest = json.loads((input_dir / "manifest.json").read_text())
    shape = arrays["weight_before"].shape
    if len(shape) != 3 or shape[0] != len(meta) or shape[1] < 1 or shape[2] < 1:
        raise ValueError(f"Invalid sample/head/key dimensions: {shape}")
    n_samples, n_heads, n_keys = shape
    for name in ("native_scores", "delta", "weight_after"):
        if arrays[name].shape != shape:
            raise ValueError(f"Inconsistent dimensions: {name}")
    for name in ("valid_mask", "history_item_ids", "history_actions"):
        if arrays[name].shape != (n_samples, n_keys):
            raise ValueError(f"Inconsistent dimensions: {name}")
    if arrays["target_item_ids"].shape != (n_samples,):
        raise ValueError("Invalid target item dimensions")
    if arrays["valid_mask"].dtype != np.dtype(bool):
        raise ValueError("valid_mask must be boolean")
    stats = {}
    for row in raw_stats:
        key = (int(row["sample_index"]), int(row["head"]))
        if key in stats:
            raise ValueError(f"Duplicate sample/head statistics: {key}")
        stats[key] = row
    seen = set()
    for index, row in enumerate(meta):
        key = int(row["sample_index"])
        if key in seen:
            raise ValueError(f"Duplicate sample identity: {key}")
        seen.add(key)
        actual_length = int(arrays["valid_mask"][index].sum())
        if int(row["effective_length"]) != actual_length:
            raise ValueError(f"Length/mask disagreement: sample {key}")
        for head in range(n_heads):
            if (key, head) not in stats:
                raise ValueError(f"Missing sample/head statistics: {(key, head)}")
    nonempty = arrays["valid_mask"].any(axis=-1)
    if not nonempty.any():
        raise ValueError("No nonempty history is available for a diagnostic case")
    valid = arrays["valid_mask"][nonempty]
    before = arrays["weight_before"][nonempty]
    after = arrays["weight_after"][nonempty]
    valid_all_heads = np.broadcast_to(valid[:, None, :], before.shape)
    for name in ("native_scores", "delta"):
        if not np.isfinite(arrays[name][nonempty][valid_all_heads]).all():
            raise ValueError(f"Nonfinite allowed-key values: {name}")
    reconstruction_errors = {}
    for name, scores, weights in (
        ("weight_before", arrays["native_scores"][nonempty], before),
        ("weight_after", arrays["native_scores"][nonempty] + arrays["delta"][nonempty], after),
    ):
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError(f"Invalid probability weights: {name}")
        if not np.allclose(weights.sum(axis=-1), 1, atol=2e-6, rtol=0):
            raise ValueError(f"Weights do not sum to one: {name}")
        if np.any(np.abs(weights[~valid_all_heads]) > 1e-8):
            raise ValueError(f"Padding unexpectedly has probability mass: {name}")
        reconstructed = reconstruct(scores, valid)
        error = float(np.max(np.abs(reconstructed - weights)))
        reconstruction_errors[name] = error
        if not np.allclose(reconstructed, weights, atol=2e-6, rtol=2e-5):
            raise ValueError(f"FP32 reconstruction mismatch for {name}: {error:g}")
    max_tv_error = 0.0
    for index, row in enumerate(meta):
        for head in range(n_heads):
            reported = float(stats[(int(row["sample_index"]), head)]["tv"])
            if not nonempty[index]:
                if not np.isnan(reported):
                    raise ValueError("Empty-history TV must be NaN, not zero")
                continue
            tv = 0.5 * float(np.abs(
                arrays["weight_after"][index, head] - arrays["weight_before"][index, head]
            ).astype(np.float64).sum())
            if not np.isfinite(reported) or abs(tv - reported) > 2e-6:
                raise ValueError(f"TV mismatch for sample {row['sample_index']}, head {head}")
            max_tv_error = max(max_tv_error, abs(tv - reported))
    validation = {
        "state": "PASS", "sample_count": n_samples, "heads": n_heads,
        "key_positions": n_keys, "nonempty_count": int(nonempty.sum()),
        "empty_history_count": int((~nonempty).sum()),
        "weights_reconstructed_with_fp32": True,
        "reconstruction_max_absolute_errors": reconstruction_errors,
        "tv_max_absolute_error": max_tv_error,
        "probability_sum_atol": 2e-6, "softmax_atol": 2e-6, "softmax_rtol": 2e-5,
        "small_cohort_warning": int(nonempty.sum()) < 20,
    }
    return arrays, meta, stats, manifest, {"sources": sources, "validation": validation}


def select_cases(meta: list, stats: dict) -> tuple[list[dict], np.ndarray]:
    candidates = []
    for array_index, row in enumerate(meta):
        tv = float(stats[(int(row["sample_index"]), HEAD)]["tv"])
        if np.isfinite(tv) and int(row["effective_length"]) > 0:
            candidates.append({
                "array_index": array_index, "sample_index": int(row["sample_index"]),
                "part_id": int(row["part_id"]), "row_offset": int(row["row_offset"]),
                "effective_length": int(row["effective_length"]), "tv": tv,
            })
    values = np.array([row["tv"] for row in candidates], dtype=np.float64)
    selected = []
    for name, quantile, label in CASE_SPECS:
        target = float(np.quantile(values, quantile, method="linear"))
        winner = min(candidates, key=lambda row: (
            abs(row["tv"] - target), row["part_id"], row["row_offset"], row["sample_index"]
        ))
        selected.append({
            **winner, "name": name, "label": label, "quantile": quantile,
            "quantile_target_tv": target, "distance_to_target_tv": abs(winner["tv"] - target),
            "empirical_cdf_at_case": float((values <= winner["tv"]).mean()), "head": HEAD,
        })
    return selected, values


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "mathtext.fontset": "dejavusans",
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.edgecolor": MUTED, "axes.linewidth": 0.65,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
    })


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> list[str]:
    paths = []
    for extension in ("pdf", "svg", "png"):
        filename = f"{name}.{extension}"
        fig.savefig(output_dir / filename, dpi=280, facecolor="white")
        paths.append(filename)
    plt.close(fig)
    return paths


def figure_context(manifest: dict) -> tuple[str, str]:
    dataset = str(manifest.get("dataset", "TencentGR_10M_Action"))
    display_dataset = {"TencentGR_10M_Action": "TencentGR"}.get(dataset, dataset)
    model = str(manifest.get("model", "HiFormer"))
    seed = str(manifest.get("seed", "unknown"))
    return f"{model}–{display_dataset}", f"Same checkpoint · seed {seed} · target attention · head {HEAD}"


def action_labels(manifest: dict) -> dict[int, str]:
    """Decode only dataset-confirmed history actions; never interpret target 0."""
    labels = {int(code): str(name) for name, code in manifest["action_vocab"].items()}
    if len(labels) != len(manifest["action_vocab"]) or 0 in labels:
        raise ValueError("Invalid history action vocabulary")
    return labels


def plot_case(
    arrays: dict, case: dict, manifest: dict, output_dir: Path,
    weight_max: float, delta_max: float,
) -> tuple[list[str], list[dict]]:
    index = case["array_index"]
    valid = arrays["valid_mask"][index]
    n_keys = len(valid)
    before = arrays["weight_before"][index, HEAD]
    after = arrays["weight_after"][index, HEAD]
    delta = arrays["delta"][index, HEAD]
    change = after - before
    positions = np.arange(1, n_keys + 1)
    valid_positions = np.flatnonzero(valid)
    annotated = sorted(valid_positions, key=lambda key: (-abs(float(change[key])), int(key)))[:5]
    annotated.sort()
    actions = action_labels(manifest)
    sequential = LinearSegmentedColormap.from_list("rise_weight", ["#F5FAFF", BLUE])
    sequential.set_bad(PADDING)
    diverging = LinearSegmentedColormap.from_list("rise_signed", [BLUE, "#FFFFFF", ORANGE])
    diverging.set_bad(PADDING)
    weight_norm = Normalize(0, weight_max)
    delta_norm = TwoSlopeNorm(vmin=-delta_max, vcenter=0, vmax=delta_max)
    fig = plt.figure(figsize=(10.4, 7.65))
    context, checkpoint = figure_context(manifest)
    fig.text(.045, .96, f"{context}: attention redistribution", fontsize=14, weight="bold")
    fig.text(.045, .926, checkpoint, fontsize=9.5)
    fig.text(.045, .891,
             f"{case['label']} (P{int(case['quantile'] * 100):02d}) · TV = {case['tv']:.4f} · "
             f"{case['effective_length']}/{n_keys} valid keys · target item {int(arrays['target_item_ids'][index])}",
             fontsize=9.5)
    grid = fig.add_gridspec(5, 1, height_ratios=[.34, .64, .64, .64, 1.35],
                            left=.177, right=.86, top=.855, bottom=.416, hspace=.22)
    axes = [fig.add_subplot(grid[row, 0]) for row in range(5)]
    for ax in axes:
        ax.set_xlim(.5, n_keys + .5)
        for side in ("top", "left", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylim(0, 1)
    axes[0].set_yticks([])
    axes[0].spines["bottom"].set_visible(False)
    axes[0].set_xticks([])
    axes[0].set_ylabel("History keys", rotation=0, ha="right", va="center", labelpad=11)
    for key, allowed in enumerate(valid):
        axes[0].add_patch(Rectangle((key + .5, .1), 1, .75,
                                   facecolor="#F4F4F4" if allowed else PADDING,
                                   edgecolor="white", linewidth=.35))
    for key in annotated:
        axes[0].add_patch(Rectangle((key + .5, .1), 1, .75,
                                   facecolor="none", edgecolor=INK, linewidth=.7))
    strips = (
        (before, r"Before  $a^{(0)}$", sequential, weight_norm, [0, weight_max]),
        (delta, r"Score change  $\Delta$", diverging, delta_norm, [-delta_max, 0, delta_max]),
        (after, r"After  $a$", sequential, weight_norm, [0, weight_max]),
    )
    for ax, (values, label, cmap, norm, ticks) in zip(axes[1:4], strips):
        displayed = np.ma.masked_where(~valid, values)
        mesh = ax.pcolormesh(np.arange(n_keys + 1) + .5, [0, 1], displayed[None, :],
                             cmap=cmap, norm=norm, shading="flat", rasterized=False)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=11)
        bounds = ax.get_position()
        color_axis = fig.add_axes([.884, bounds.y0 + .08 * bounds.height, .012, .84 * bounds.height])
        colorbar = fig.colorbar(mesh, cax=color_axis, ticks=ticks)
        colorbar.outline.set_linewidth(.5)
        colorbar.ax.tick_params(labelsize=7, length=2)
        colorbar.ax.set_yticklabels([f"{value:.3g}" for value in ticks])
    bar_axis = axes[4]
    for key in np.flatnonzero(~valid):
        bar_axis.axvspan(key + .5, key + 1.5, color=PADDING, linewidth=0, zorder=0)
    bar_axis.bar(positions[valid], change[valid], width=.78,
                 color=np.where(change[valid] >= 0, ORANGE, BLUE), edgecolor="none", zorder=2)
    limit = max(float(np.abs(change[valid]).max()) * 1.18, 1e-7)
    bar_axis.set_ylim(-limit, limit)
    bar_axis.axhline(0, color=INK, linewidth=.7, zorder=3)
    bar_axis.set_ylabel(r"Weight change" + "\n" + r"$a-a^{(0)}$", rotation=0,
                        ha="right", va="center", labelpad=11)
    bar_axis.tick_params(axis="y", labelsize=7, length=2)
    bar_axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    bar_axis.yaxis.get_offset_text().set_fontsize(7)
    ticks = sorted(set([1, n_keys] + list(range(10, n_keys + 1, 10))))
    bar_axis.set_xticks(ticks)
    bar_axis.tick_params(axis="x", labelsize=8, length=3)
    bar_axis.set_xlabel("Original history key position (1-based; gray = padding)", labelpad=5)
    rows = []
    for key in annotated:
        rows.append({
            "position_1based": int(key + 1), "history_item_id": int(arrays["history_item_ids"][index, key]),
            "history_action_code": int(arrays["history_actions"][index, key]),
            "history_action_name": actions.get(int(arrays["history_actions"][index, key]), "unmapped"),
            "score_delta": float(delta[key]), "weight_before": float(before[key]),
            "weight_after": float(after[key]), "weight_change": float(change[key]),
        })
    fig.text(.045, .332, "Position annotations: five largest absolute weight changes within this selected case",
             fontsize=9, weight="bold")
    table_axis = fig.add_axes([.045, .131, .913, .182])
    table_axis.axis("off")
    cells = [[str(row["position_1based"]), str(row["history_item_id"]),
              f"{row['history_action_name']} ({row['history_action_code']})", f"{row['score_delta']:+.4f}",
              f"{row['weight_before']:.5f}", f"{row['weight_after']:.5f}",
              f"{row['weight_change']:+.5f}"] for row in rows]
    table = table_axis.table(cellText=cells,
                            colLabels=["Position", "History item ID", "History action", r"$\Delta$",
                                       r"$a^{(0)}$", r"$a$", r"$a-a^{(0)}$"],
                            colWidths=[.09, .20, .14, .14, .14, .14, .15],
                            cellLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, _), cell in table.get_celld().items():
        cell.set_linewidth(.4)
        cell.set_edgecolor(GRID)
        cell.set_facecolor("#F4F4F4" if row == 0 else "white")
        if row == 0:
            cell.set_text_props(weight="bold")
    fig.text(.045, .091,
             f"Sample {case['sample_index']} · part {case['part_id']} · row {case['row_offset']} · "
             "case chosen by TV quantile, without prediction/label selection.", fontsize=7.6)
    fig.text(.045, .069,
             "Same-checkpoint local comparison, not an independently trained baseline. FP32 reconstruction; "
             "full allowed-key softmax; no re-normalization.", fontsize=7.6)
    fig.text(.045, .047,
             "Before/after share a color scale; score colors are zero-centered. Item IDs are anonymous; "
             "key position does not imply elapsed time.", fontsize=7.6)
    vocabulary = ", ".join(f"{name}={code}" for code, name in sorted(actions.items()))
    fig.text(.045, .025,
             f"History action vocabulary: {vocabulary}. Target action 0 is not interpreted as a behavior.",
             fontsize=7.6)
    return save_figure(fig, output_dir, f"attention_{case['name']}"), rows


def plot_distribution(values: np.ndarray, cases: list[dict], manifest: dict,
                      total: int, output_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.0, 4.65))
    context, checkpoint = figure_context(manifest)
    fig.text(.055, .951, f"{context}: attention-change distribution", fontsize=12, weight="bold")
    fig.text(.055, .908, checkpoint, fontsize=9)
    fig.text(.055, .865,
             f"{total} fixed-window diagnostic samples · {len(values)} nonempty histories · head {HEAD}", fontsize=9)
    ordered = np.sort(values)
    cdf = np.arange(1, len(ordered) + 1) / len(ordered)
    ax.step(np.r_[0, ordered, 1], np.r_[0, cdf, 1], where="post", color=BLUE, linewidth=1.5)
    case_markers = {"main_p50": ("o", "-"), "appendix_p10": ("v", "--"), "appendix_p90": ("^", ":")}
    for case in sorted(cases, key=lambda row: row["quantile"]):
        marker, line = case_markers[case["name"]]
        ax.axvline(case["tv"], color=MUTED, linestyle=line, linewidth=.8)
        ax.plot(case["tv"], case["empirical_cdf_at_case"], marker=marker, markersize=6,
                markerfacecolor="white", markeredgecolor=INK, linestyle="none", zorder=4,
                label=f"P{int(case['quantile'] * 100):02d} case: TV {case['tv']:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"Total variation  $\frac{1}{2}\sum_j |a_j-a_j^{(0)}|$")
    ax.set_ylabel("Cumulative fraction of nonempty samples")
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.grid(axis="y", color=GRID, linewidth=.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.text(.055, .100,
             "Fixed windows: not a representative full-test sample. Empty histories excluded from the ECDF.", fontsize=8)
    fig.text(.055, .062,
             "Same-checkpoint comparison, not an independently trained baseline. FP32 reconstruction.", fontsize=8)
    fig.text(.055, .024,
             "TV measures redistribution magnitude; it does not measure prediction improvement.", fontsize=8)
    fig.subplots_adjust(left=.14, right=.965, top=.815, bottom=.260)
    return save_figure(fig, output_dir, "attention_tv_ecdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    arrays, meta, stats, manifest, audit = load_evidence(args.input_dir)
    cases, values = select_cases(meta, stats)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    weight_max = max(float(arrays[key][case["array_index"], HEAD].max())
                     for case in cases for key in ("weight_before", "weight_after"))
    delta_max = max(float(np.abs(arrays["delta"][case["array_index"], HEAD][
        arrays["valid_mask"][case["array_index"]]]).max()) for case in cases)
    weight_max, delta_max = max(weight_max, 1e-8), max(delta_max, 1e-8)
    receipt = {
        "input_sources_sha256": audit["sources"],
        "selection_rule": "nearest to numpy linear TV quantile; ties by part_id,row_offset,sample_index",
        "fixed_head": HEAD, "candidate_count": len(values),
        "excluded_empty_count": len(meta) - len(values),
        "prediction_or_label_based_selection": False,
        "annotation_rule": "five valid keys with largest absolute weight change; ties by position; shown in key order",
        "shared_case_weight_color_range": [0, weight_max],
        "shared_case_delta_color_range": [-delta_max, delta_max],
        "scope": "fixed-window diagnostics; no full-test representativeness claim",
        "cases": cases,
    }
    write_json(args.output_dir / "case_selection_receipt.json", receipt)
    write_json(args.output_dir / "figure_numerical_validation.json", audit["validation"])
    figure_map = []
    all_positions = []
    for case in cases:
        files, annotations = plot_case(arrays, case, manifest, args.output_dir, weight_max, delta_max)
        write_csv(args.output_dir / f"attention_{case['name']}_annotations.csv", annotations)
        for key in range(arrays["valid_mask"].shape[-1]):
            index = case["array_index"]
            all_positions.append({
                "case": case["name"], "sample_index": case["sample_index"], "head": HEAD,
                "position_1based": key + 1, "valid_key": bool(arrays["valid_mask"][index, key]),
                "history_item_id": int(arrays["history_item_ids"][index, key]),
                "history_action_code": int(arrays["history_actions"][index, key]),
                "native_score": float(arrays["native_scores"][index, HEAD, key]),
                "score_delta": float(arrays["delta"][index, HEAD, key]),
                "weight_before": float(arrays["weight_before"][index, HEAD, key]),
                "weight_after": float(arrays["weight_after"][index, HEAD, key]),
            })
        figure_map.append({
            "section": "main" if case["name"] == "main_p50" else "appendix",
            "question": "How does the local correction redistribute weight across original history positions?",
            "family": "aligned heatmap strips and signed bars", "case": case,
            "fields": ["weight_before", "delta", "weight_after", "weight_after-weight_before"],
            "supported_takeaway": "local attention redistribution in one fixed model/input/site/head",
            "palette_policy": "hard two-root cap: blue/orange plus neutrals",
            "exports": files,
        })
    write_csv(args.output_dir / "selected_case_positions.csv", all_positions)
    files = plot_distribution(values, cases, manifest, len(meta), args.output_dir)
    figure_map.append({
        "section": "main", "question": "How typical is each selected case within the fixed diagnostic windows?",
        "family": "distribution", "variant": "empirical CDF", "field": "tv",
        "supported_takeaway": "distribution of local attention-change magnitude in nonempty diagnostic histories",
        "palette_policy": "single-root blue plus neutral quantile references",
        "sample_count": len(values), "exports": files,
    })
    write_json(args.output_dir / "chart_map.json", {
        "renderer": "matplotlib standalone PDF/SVG/PNG", "sources_sha256": audit["sources"],
        "figures": figure_map, "qa": "numerical checks passed; exported PNG visual inspection required",
    })
    print(json.dumps({"state": "RENDERED", "output_dir": str(args.output_dir),
                      "nonempty_count": len(values), "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
