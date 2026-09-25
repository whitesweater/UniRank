"""Render semantic KuaiRand examples from audited, fixed-checkpoint observations.

Chart contract:
  Question: which named history items change weight, and does the fixed model's
      prediction move toward this sample's observed click label?
  Surface: standalone Chinese PDF/SVG/PNG for the existing paper study.
  Family: paired horizontal bars, with target text and prediction/loss readout.
  Grain: one positive-click interaction per loss-direction stratum, head zero;
      show at most three largest positive and negative weight changes.
  Selection: nearest to within-stratum median BCE gain, then physical part/row;
      skip only unavailable/too-short verified text and retain every reason.
  Palette: neutral gray versus blue; direct labels distinguish off/on without
      color. Common bar axis across cases, zero baseline, original full-key
      normalization. Omitted keys retain their original aggregate mass.
  QA: validate probabilities, attention sums, metadata identity and selection;
      export exact full text and numeric evidence, then inspect real PNGs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import unicodedata
from typing import Any

import numpy as np


HEAD = 0
EPSILON = 1e-6
MIN_TEXT_CHARACTERS = 4
LABEL = "is_click"
BLUE = "#245B8A"
GRAY = "#969B9F"
INK = "#242A2E"
MUTED = "#656D73"
GRID = "#E1E4E7"
GROUP_NAMES = {"improved": "损失降低案例", "degraded": "损失升高案例"}
ACTION_TRANSLATIONS = {
    "is_click": "点击", "is_like": "点赞", "is_follow": "关注",
    "is_comment": "评论", "is_forward": "转发", "long_view": "长观看",
    "exposure": "曝光（六项反馈均为0）",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields=None) -> None:
    if not rows and not fields:
        raise ValueError(f"No rows or schema for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bce(probability: float, label: int) -> float:
    if not 0 <= probability <= 1 or label not in (0, 1):
        raise ValueError("Invalid binary probability or label")
    # Consistent, explicit FP64 clamp also handles exactly saturated predictions.
    probability = min(max(probability, 1e-7), 1 - 1e-7)
    return -math.log(probability if label else 1 - probability)


def changed_positions(before: np.ndarray, after: np.ndarray, valid: np.ndarray) -> list[int]:
    delta = after.astype(np.float64) - before.astype(np.float64)
    positions = np.flatnonzero(valid).tolist()
    increase = sorted((p for p in positions if delta[p] > 0), key=lambda p: (-delta[p], p))[:3]
    decrease = sorted((p for p in positions if delta[p] < 0), key=lambda p: (delta[p], p))[:3]
    return increase + decrease


def select_ranking(meta: list[dict], predictions: list[dict], arrays: dict) -> list[dict]:
    """Return all eligible candidates in a frozen, content-independent order."""
    by_sample = {int(row["sample_index"]): row for row in predictions}
    if len(by_sample) != len(predictions):
        raise ValueError("Duplicate prediction sample IDs")
    if set(by_sample) != {int(row["sample_index"]) for row in meta}:
        raise ValueError("Prediction and attention sample IDs differ")
    groups: dict[str, list[dict]] = {"improved": [], "degraded": []}
    for array_index, row in enumerate(meta):
        sample_index = int(row["sample_index"])
        prediction = by_sample[sample_index]
        label = int(float(prediction[LABEL]))
        # The export writes float32 prediction columns with pandas' float32
        # shortest representation. Recover that dtype before FP64 loss checking.
        on = float(np.float32(prediction[f"{LABEL}_plain"]))
        off = float(np.float32(prediction[f"{LABEL}_local_off"]))
        loss_on, loss_off = bce(on, label), bce(off, label)
        gain = float(prediction[f"{LABEL}_bce_gain"])
        if abs(gain - (loss_off - loss_on)) > 1e-12:
            raise ValueError("Exported BCE gain disagrees with captured probabilities")
        if label != 1 or not arrays["valid_mask"][array_index].any() or abs(gain) <= EPSILON:
            continue
        group = "improved" if gain > EPSILON else "degraded"
        groups[group].append({
            "group": group, "array_index": array_index, "sample_index": sample_index,
            "part_id": int(row["part_id"]), "row_offset": int(row["row_offset"]),
            "target_item_id": int(arrays["target_item_ids"][array_index]),
            "label": label, "p_local_off": off, "p_full_on": on,
            "bce_local_off": loss_off, "bce_full_on": loss_on, "gain": gain,
            "head": HEAD, "effective_length": int(arrays["valid_mask"][array_index].sum()),
        })
    ranking = []
    for group, candidates in groups.items():
        if not candidates:
            continue
        target = float(np.median([row["gain"] for row in candidates]))
        ordered = sorted(candidates, key=lambda row: (
            abs(row["gain"] - target), row["part_id"], row["row_offset"], row["sample_index"]))
        for rank, row in enumerate(ordered, 1):
            ranking.append({**row, "group_count": len(ordered), "group_median_gain": target,
                            "distance_to_median": abs(row["gain"] - target), "rank": rank})
    return ranking


def caption_text(row: dict) -> tuple[str, str]:
    caption = str(row.get("caption") or "").strip()
    cover = str(row.get("show_cover_text", row.get("cover_text")) or "").strip()
    count = lambda value: sum(char.isalnum() for char in value)
    if count(caption) >= MIN_TEXT_CHARACTERS:
        return caption, "caption"
    if count(cover) >= MIN_TEXT_CHARACTERS:
        return cover, "show_cover_text"
    return "", "missing_or_fewer_than_4_alphanumeric_characters"


def metadata_status(identifier: int, captions: dict[int, dict]) -> tuple[str, str]:
    if identifier not in captions:
        return "pending", "annotation_not_requested_or_not_returned"
    row = captions[identifier]
    if not row.get("exact_id_verified") or not row.get("overlap_verified"):
        return "skip", "annotation_exact_id_or_overlap_verification_failed"
    if int(row["final_video_id"]) + 1 != identifier:
        raise ValueError(f"Incorrect encoded-to-official ID mapping for {identifier}")
    for gate in ("mapping_verified", "duration_bucket_verified"):
        if not row.get(gate):
            return "skip", f"annotation_{gate}_failed"
    text, field = caption_text(row)
    if not text:
        return "skip", field
    if not row.get("record_usable"):
        return "skip", "annotation_record_usable_failed"
    return "ready", field


def choose_cases(ranking: list[dict], arrays: dict, captions: dict[int, dict]) -> tuple[list, list, list]:
    selected, requests, audit = [], [], []
    resolved = set()
    for candidate in ranking:
        row = dict(candidate)
        group = row["group"]
        if group in resolved:
            audit.append({**row, "status": "not_reached", "reason": "earlier_candidate_selected_or_pending"})
            continue
        index = row["array_index"]
        positions = changed_positions(arrays["weight_before"][index, HEAD],
                                      arrays["weight_after"][index, HEAD], arrays["valid_mask"][index])
        required = [(row["target_item_id"], "target", -1)] + [
            (int(arrays["history_item_ids"][index, pos]), "history", pos) for pos in positions]
        statuses = [(identifier, role, pos, *metadata_status(identifier, captions))
                    for identifier, role, pos in required]
        failures = [value for value in statuses if value[3] == "skip"]
        if failures:
            reason = "; ".join(f"{role}[{pos}] item={identifier}: {why}"
                               for identifier, role, pos, _, why in failures)
            audit.append({**row, "status": "skipped", "reason": reason})
            continue
        pending = [value for value in statuses if value[3] == "pending"]
        if pending:
            audit.append({**row, "status": "pending_metadata", "reason": "; ".join(
                f"{role}[{pos}] item={identifier}" for identifier, role, pos, _, _ in pending)})
            for identifier, role, pos, _, _ in pending:
                requests.append({"group": group, "rank": row["rank"], "sample_index": row["sample_index"],
                                 "encoded_item_id": identifier, "final_video_id": identifier - 1,
                                 "role": role, "history_position_zero_based": pos,
                                 "duration_bucket": int(arrays["target_duration_bucket"][index]) if role == "target"
                                 else int(arrays["history_duration_bucket"][index, pos])})
        else:
            row["positions"] = positions
            selected.append(row)
            audit.append({**candidate, "status": "selected", "reason": "all_required_text_verified_and_readable"})
        resolved.add(group)
    return selected, requests, audit


def load_captions(path: Path | None) -> dict[int, dict]:
    if path is None:
        return {}
    rows = read_csv(path) if path.suffix == ".csv" else json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Caption file must contain a list of exact source rows")
    for row in rows:
        row["item_id"] = int(row.get("encoded_item_id", row.get("item_id")))
        row["show_cover_text"] = row.get("show_cover_text", row.get("cover_text", ""))
        for gate in ("exact_id_verified", "overlap_verified", "mapping_verified", "record_usable",
                     "duration_bucket_verified", "duration_valid"):
            if isinstance(row.get(gate), str):
                if row[gate].lower() not in ("true", "false", ""):
                    raise ValueError(f"Invalid metadata boolean: {gate}={row[gate]}")
                row[gate] = row[gate].lower() == "true"
        if row.get("duration_valid"):
            row["duration_ms"] = float(row["duration_ms"])
    result = {int(row["item_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate metadata item IDs")
    return result


def load_evidence(directory: Path) -> tuple[dict, list, list, dict]:
    complete = json.loads((directory / "complete.json").read_text())
    if not complete.get("completed"):
        raise ValueError("Semantic export is incomplete")
    gate = json.loads((directory / "prediction_gate.json").read_text())
    if not gate.get("passed") or not gate.get("bitwise_equal") or not gate.get("full_forward_predictions_bitwise_equal"):
        raise ValueError("Observation changed model predictions; refusing figures")
    local_gate = json.loads((directory / "local_intervention_gate.json").read_text())
    if not local_gate.get("passed") or not local_gate.get("selected_qkv_bitwise_equal"):
        raise ValueError("Local-off intervention changed upstream Q/K/V; refusing figures")
    filenames = ("raw_attention.npz", "meta.csv", "predictions.csv", "manifest.json", "prediction_gate.json",
                 "local_intervention_gate.json")
    sources = {}
    for filename in filenames:
        digest = sha256(directory / filename)
        if complete.get("files_sha256", {}).get(filename) != digest:
            raise ValueError(f"Complete-export source hash mismatch: {filename}")
        sources[filename] = digest
    with np.load(directory / "raw_attention.npz", allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    meta = read_csv(directory / "meta.csv")
    predictions = read_csv(directory / "predictions.csv")
    shape = arrays["weight_before"].shape
    if len(shape) != 3 or shape[0] != len(meta) or shape[1] != 1:
        raise ValueError("Malformed sample/head/key dimensions")
    for name in ("native_scores", "delta", "weight_after"):
        if arrays[name].shape != shape:
            raise ValueError(f"Mismatched shape: {name}")
    n, _, keys = shape
    for name in ("valid_mask", "history_item_ids", "history_actions", "history_duration_bucket"):
        if arrays[name].shape != (n, keys):
            raise ValueError(f"Mismatched shape: {name}")
    for name in ("target_item_ids", "target_duration_bucket"):
        if arrays[name].shape != (n,):
            raise ValueError(f"Mismatched shape: {name}")
    valid = arrays["valid_mask"]
    if valid.dtype != bool:
        raise ValueError("valid_mask must be boolean")
    nonempty = valid.any(axis=-1)
    checks = {}
    for name in ("weight_before", "weight_after"):
        weight = arrays[name][nonempty]
        allowed = np.broadcast_to(valid[nonempty, None, :], weight.shape)
        if not np.isfinite(weight).all() or (weight < 0).any():
            raise ValueError(f"Invalid probabilities: {name}")
        error = float(np.max(np.abs(weight.sum(axis=-1) - 1)))
        if error > 2e-6 or np.any(weight[~allowed] != 0):
            raise ValueError(f"Invalid full-key normalization/padding: {name}")
        scores = arrays["native_scores"][nonempty].astype(np.float32)
        if name == "weight_after":
            scores = scores + arrays["delta"][nonempty].astype(np.float32)
        scores = np.where(allowed, scores, -np.inf)
        probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        reconstruction_error = float(np.max(np.abs(probabilities - weight)))
        if not np.allclose(probabilities, weight, atol=2e-6, rtol=2e-5):
            raise ValueError(f"Softmax reconstruction failed: {name}")
        checks[name] = {"normalization_max_error": error,
                        "reconstruction_max_error": reconstruction_error}
    sources["prediction_gate.json"] = sha256(directory / "prediction_gate.json")
    return arrays, meta, predictions, {"sources": sources, "checks": checks,
                                     "observation_bitwise_equal": True, "n": n}


def compact_text(value: str, max_units: int, max_lines: int) -> str:
    """Faithfully truncate/wrap source text; retain the full text in source CSV."""
    value = " ".join(value.split())
    lines, current, units = [], "", 0
    for char in value:
        width = 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        if units + width > max_units:
            lines.append(current)
            current, units = "", 0
            if len(lines) >= max_lines:
                lines[-1] = lines[-1].rstrip() + "…"
                return "\n".join(lines)
        current += char
        units += width
    if current:
        lines.append(current)
    return "\n".join(lines)


def action_text(code: int, action_vocab: dict) -> tuple[str, str]:
    reverse = {int(value): name for name, value in action_vocab.items()}
    if code not in reverse:
        raise ValueError(f"Unrecognized nonpadding action code {code}")
    raw = reverse[code]
    return raw, "、".join(ACTION_TRANSLATIONS[name] for name in raw.split("|"))


def case_rows(case: dict, arrays: dict, captions: dict, action_vocab: dict) -> list[dict]:
    index = case["array_index"]
    rows = []
    for position in np.flatnonzero(arrays["valid_mask"][index]):
        identifier = int(arrays["history_item_ids"][index, position])
        metadata = captions.get(identifier, {})
        text, field = caption_text(metadata)
        before = float(arrays["weight_before"][index, HEAD, position])
        after = float(arrays["weight_after"][index, HEAD, position])
        action_raw, action_cn = action_text(int(arrays["history_actions"][index, position]), action_vocab)
        rows.append({
            "group": case["group"], "sample_index": case["sample_index"],
            "part_id": case["part_id"], "row_offset": case["row_offset"], "head": HEAD,
            "position": int(position) + 1, "item_id": identifier, "final_video_id": identifier - 1,
            "displayed": int(position) in case["positions"], "caption": metadata.get("caption", ""),
            "show_cover_text": metadata.get("show_cover_text", ""), "chosen_text_field": field,
            "text_lookup_status": "retrieved" if metadata else "not_requested_for_undisplayed_position",
            "display_text": compact_text(text, 72, 2) if text else "",
            "action_code": int(arrays["history_actions"][index, position]),
            "action_names": action_raw, "actions_chinese": action_cn,
            "native_score": float(arrays["native_scores"][index, HEAD, position]),
            "score_delta": float(arrays["delta"][index, HEAD, position]),
            "weight_before": before, "weight_after": after, "weight_difference": after - before,
        })
    return rows


def render_case(case: dict, rows: list[dict], target: dict, directory: Path, axis_max: float) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    font_path = Path("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc")
    if not font_path.exists():
        raise FileNotFoundError("CJK font unavailable; cannot ship unreadable semantic figures")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font_path)).get_name(),
                         "axes.unicode_minus": False, "pdf.fonttype": 42,
                         "svg.fonttype": "path", "font.size": 11,
                         "text.color": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    ordered = [{row["position"] - 1: row for row in rows}[position] for position in case["positions"]]
    text, field = caption_text(target)
    figure = plt.figure(figsize=(15, 13.5), facecolor="white")
    figure.text(.045, .957, f"KuaiRand：候选内容、历史注意力与点击预测", fontsize=22, weight="bold")
    figure.text(.045, .925, f"{GROUP_NAMES[case['group']]}  ·  真实点击正例  ·  固定 RankMixer–RISE 模型  ·  target attention / head 0",
                fontsize=11.5, color=MUTED)
    figure.text(.045, .900, "当前候选视频", fontsize=11.5, color=MUTED)
    figure.text(.045, .880, compact_text(text, 170, 2), fontsize=14, va="top", linespacing=1.65)
    figure.text(.045, .823, "实际反馈：点击（y = 1）", fontsize=13, weight="bold")
    figure.text(.045, .783, "该位置修正关闭", fontsize=11, color=MUTED)
    figure.text(.300, .783, "该位置修正开启", fontsize=11, color=BLUE)
    figure.text(.580, .783, "该样本的交叉熵变化", fontsize=11, color=MUTED)
    figure.text(.045, .757, f"p(click) = {case['p_local_off']:.4%}", fontsize=16, color=MUTED)
    figure.text(.300, .757, f"p(click) = {case['p_full_on']:.4%}", fontsize=16, color=BLUE)
    figure.text(.580, .757, f"{case['bce_local_off']:.5f} → {case['bce_full_on']:.5f}", fontsize=16)
    direction = "降低" if case["gain"] > 0 else "升高"
    figure.text(.830, .757, f"{direction} {abs(case['gain']):.5f}", fontsize=12)
    figure.add_artist(Line2D([.045, .955], [.733, .733], transform=figure.transFigure, color=GRID, linewidth=1))
    figure.text(.045, .707, "历史视频原文与已记录动作", fontsize=12, weight="bold")
    figure.text(.550, .707, "注意力权重（完整历史归一化）", fontsize=12, weight="bold")
    for x, color, label in ((.550, GRAY, "修正前"), (.680, BLUE, "修正后")):
        figure.add_artist(Line2D([x, x + .018], [.687, .687], transform=figure.transFigure,
                                 color=color, linewidth=5, solid_capstyle="butt"))
        figure.text(x + .026, .683, label, color=BLUE if color == BLUE else MUTED, fontsize=10)
    left, bottom, height = .550, .202, .457
    axis = figure.add_axes([left, bottom, .365, height])
    y_positions = np.arange(len(ordered))[::-1]
    axis.set_ylim(-.5, len(ordered) - .5)
    axis.set_xlim(0, axis_max * 1.26)
    axis.set_yticks([])
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.set_axisbelow(True)
    axis.xaxis.grid(True, color=GRID, linewidth=.6)
    axis.set_xlabel("权重（%）", fontsize=10)
    for y, row in zip(y_positions, ordered):
        before, after = row["weight_before"] * 100, row["weight_after"] * 100
        axis.barh(y + .13, before, height=.21, color=GRAY, edgecolor=GRAY, linewidth=.5)
        axis.barh(y - .13, after, height=.21, color=BLUE, edgecolor=BLUE, linewidth=.5)
        for yy, weight, color in ((y + .13, before, MUTED), (y - .13, after, BLUE)):
            axis.text(weight + axis_max * .018, yy, f"{weight:.3f}%", va="center", color=color, fontsize=10)
        center = bottom + (y + .5) / len(ordered) * height
        sign_text = "增加" if row["weight_difference"] > 0 else "减少"
        figure.text(.045, center + .023, f"位置 {row['position']}  ·  {sign_text} {row['weight_difference'] * 100:+.3f} 个百分点",
                    fontsize=10.2, color=BLUE if row["weight_difference"] > 0 else MUTED)
        figure.text(.045, center + .005, row["display_text"], fontsize=11.1, va="top", linespacing=1.25)
        action_y = center - (.035 if "\n" in row["display_text"] else .025)
        figure.text(.045, action_y, "已记录动作：" + row["actions_chinese"], fontsize=9.2, color=MUTED)
    displayed = {row["position"] for row in ordered}
    others = [row for row in rows if row["position"] not in displayed]
    other_before = sum(row["weight_before"] for row in others)
    other_after = sum(row["weight_after"] for row in others)
    figure.text(.045, .155, f"其余 {len(others)} 个有效历史位置合计：{other_before:.3%} → {other_after:.3%}（保留原权重，未重新归一化）",
                fontsize=11)
    movement = "上升，更接近" if case["gain"] > 0 else "下降，远离"
    figure.text(.045, .118, f"如何读图：对这些历史内容的加权发生变化；开启局部修正后，点击概率{movement}本次真实点击标签。",
                fontsize=11)
    figure.text(.045, .090, "这只能判断该样本标签下的损失方向；不能据此认定每条历史更相关，也不能推断总体 AUC 提升。",
                fontsize=10.3, color=MUTED)
    figure.text(.045, .062, "固定单头 target attention，仅关闭该接入点分数修正；官方补充文案仅用于展示，未输入模型。",
                fontsize=9.5, color=MUTED)
    figure.text(.045, .037, f"选例：各方向真实点击样本中，最接近组内 BCE 增益中位数的可读案例；part {case['part_id']} / row {case['row_offset']}。文本截断处用 …；完整原文见 CSV。",
                fontsize=9.3, color=MUTED)
    stem = f"semantic_{case['group']}"
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    text_artists = list(figure.texts) + list(axis.texts) + [axis.xaxis.label] + list(axis.get_xticklabels())
    boxes = [(artist.get_text(), artist.get_window_extent(renderer)) for artist in text_artists
             if artist.get_visible() and artist.get_text()]
    collisions = []
    for index, (first_text, first_box) in enumerate(boxes):
        for second_text, second_box in boxes[index + 1:]:
            if first_box.overlaps(second_box):
                collisions.append([first_text, second_text])
    bounds = figure.bbox
    outside = [value for value, box in boxes if box.x0 < bounds.x0 or box.y0 < bounds.y0
               or box.x1 > bounds.x1 or box.y1 > bounds.y1]
    files = []
    for suffix in ("pdf", "svg", "png"):
        path = directory / f"{stem}.{suffix}"
        figure.savefig(path, dpi=180, facecolor="white", metadata={"Creator": "UniRank audited semantic attention study"})
        files.append({"name": path.name, "sha256": sha256(path)})
    plt.close(figure)
    return {"group": case["group"], "display_positions_one_based": [row["position"] for row in ordered],
            "target_display_text": compact_text(text, 170, 2), "target_text_field": field,
            "other_key_count": len(others), "other_weight_before": other_before,
            "other_weight_after": other_after, "axis_max_percent": axis_max * 1.26,
            "text_collisions": collisions, "text_outside_canvas": outside, "files": files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/rise_attention_semantic/figures"))
    parser.add_argument("--captions", type=Path)
    parser.add_argument("--mapping-validation", type=Path,
                        default=Path("artifacts/rise_attention_semantic/metadata/mapping_validation.json"))
    parser.add_argument("--dataset-meta", type=Path, default=Path("datasets/KuaiRand_Video_Action/meta_data.json"))
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays, meta, predictions, evidence = load_evidence(args.input_dir)
    ranking = select_ranking(meta, predictions, arrays)
    if not ranking:
        raise ValueError("No eligible positive-click candidate in either direction")
    captions = load_captions(args.captions)
    selected, requests, audit = choose_cases(ranking, arrays, captions)
    write_csv(args.output_dir / "candidate_ranking.csv", audit)
    write_json(args.output_dir / "requested_item_ids.json", sorted({row["encoded_item_id"] for row in requests}))
    write_csv(args.output_dir / "requested_metadata.csv", requests,
              fields=["group", "rank", "sample_index", "encoded_item_id", "final_video_id", "role",
                      "history_position_zero_based", "duration_bucket"])
    receipt = {"mode": "selection_only" if args.select_only else "render",
               "task": LABEL, "positive_label_only": True, "gain_epsilon": EPSILON,
               "median_rule": "nearest within-direction median; ties part_id,row_offset,sample_index",
               "readability_rule": "caption then cover text; at least 4 alphanumeric characters",
               "annotation_skips": [row for row in audit if row["status"] == "skipped"],
               "selected": selected, "pending_requests": requests,
               "candidate_group_counts": {group: sum(row["group"] == group for row in ranking)
                                          for group in GROUP_NAMES}, **evidence}
    if args.captions:
        receipt["caption_sha256"] = sha256(args.captions)
    write_json(args.output_dir / "selection_receipt.json", receipt)
    if args.select_only:
        print(json.dumps({"selected": len(selected), "requested_unique_items": len(set(row["encoded_item_id"] for row in requests)),
                          "candidate_group_counts": receipt["candidate_group_counts"]}))
        return
    if requests:
        raise ValueError("Required annotations pending; inspect requested_item_ids.json before rendering")
    if not selected:
        raise ValueError("No candidates have verified readable text")
    mapping = json.loads(args.mapping_validation.read_text())
    if mapping.get("status") != "passed" or mapping.get("duration_agreement") != 1.0:
        raise ValueError("Encoded-to-official ID mapping has not passed validation")
    duration_codes = mapping["observed_duration_label_to_code"]
    from prepare_kuairand_semantic_metadata import duration_label
    for case in selected:
        index = case["array_index"]
        identities = [(case["target_item_id"], int(arrays["target_duration_bucket"][index]))] + [
            (int(arrays["history_item_ids"][index, p]), int(arrays["history_duration_bucket"][index, p]))
            for p in case["positions"]]
        for identifier, actual in identities:
            caption = captions[identifier]
            if not caption.get("duration_valid"):
                raise ValueError(f"Selected item lacks independent duration verification: {identifier}")
            expected = duration_codes.get(duration_label(caption["duration_ms"]))
            if expected != actual:
                raise ValueError(f"Selected duration mismatch: {identifier}, expected {expected}, actual {actual}")
    action_vocab = json.loads(args.dataset_meta.read_text())["action_vocab"]
    all_rows = {case["group"]: case_rows(case, arrays, captions, action_vocab) for case in selected}
    axis_max = max(row[weight] * 100 for rows in all_rows.values() for row in rows if row["displayed"]
                   for weight in ("weight_before", "weight_after"))
    rendered = []
    target_rows = []
    for case in selected:
        rows = all_rows[case["group"]]
        write_csv(args.output_dir / f"semantic_{case['group']}_all_positions.csv", rows)
        target = captions[case["target_item_id"]]
        rendered.append(render_case(case, rows, target, args.output_dir, axis_max))
        target_rows.append({**{key: value for key, value in case.items() if key != "positions"},
                            "final_video_id": case["target_item_id"] - 1,
                            "caption": target.get("caption", ""),
                            "show_cover_text": target.get("show_cover_text", "")})
    write_csv(args.output_dir / "semantic_case_targets.csv", target_rows)
    write_json(args.output_dir / "numeric_qa.json", {
        "status": "PASS" if all(not row["text_collisions"] and not row["text_outside_canvas"]
                                   for row in rendered) else "FAIL_LAYOUT",
        "observation_bitwise_equal": True, "input_validation": evidence,
        "same_axis_across_cases": True, "axis_zero_origin": True,
        "original_normalization_retained": True, "head": HEAD,
        "mapping_sha256": sha256(args.mapping_validation), "action_vocab_sha256": sha256(args.dataset_meta),
        "script_sha256": sha256(Path(__file__)), "figures": rendered,
        "visual_inspection": "Recorded separately in visual_qa.json against inspected PNG SHA256 hashes",
        "interpretation_boundary": "Fixed-model local sensitivity and per-observed-label BCE direction; not independent baseline or overall AUC evidence",
    })
    print(json.dumps({"figures": len(rendered), "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
