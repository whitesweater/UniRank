"""Render one accepted recent-feedback case from audited local interventions.

Chart contract:
  Question: does the increased mass on recently clicked histories participate
      in this fixed model's prediction gain on an observed click?
  Surface: standalone Chinese PDF/SVG/PNG for the paper; no interactive runtime.
  Grain: one explicitly selected sample, one frozen target-attention head;
      last five histories and all five prespecified intervention routes.
  Family: evidence table, horizontal zero-origin group-mass and prediction bars.
  Selection gate: positive full-vs-local-off BCE gain, at least 50% attenuation
      on related-group mass rollback, and greater damage from equal-dose removal
      of clicked versus unclicked recent-history mass. No automatic re-selection.
  Palette: blue and gold roots plus neutrals; labels and hatching encode states.
  Footprint: 14 x 11 inches, common left alignment, vector PDF/SVG plus 180-dpi PNG.
  QA: source hashes, inference gates, decoded actions, exact mass/probability
      checks, all control results retained, source CSVs, layout boxes and PNG review.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from plot_rise_semantic_attention import (
    action_text,
    bce,
    caption_text,
    load_captions,
    load_evidence,
    metadata_status,
    read_csv,
    sha256,
    write_csv,
    write_json,
)


BLUE = "#245B8A"
GOLD = "#B38422"
INK = "#242A2E"
MUTED = "#656D73"
GRAY = "#A2A8AD"
GRID = "#DEE2E6"
ROUTES = ("full_hook", "related_revert", "related_equal_mass", "control_equal_mass", "position_template")
SUMMARIES = {
    273: {96: "草原上的孕妇生活", 97: "萌娃与棒棒糖", 98: "《汉武大帝》片段", 99: "出生第10天的宝宝", 100: "教育、育儿标签"},
    154: {96: "千层底布鞋", 97: "特战旅考核剧情", 98: "文案缺失", 99: "课堂剧情", 100: "军旅剧标签"},
    363: {96: "达摩游戏片段", 97: "老夫子、铠游戏片段", 98: "钟馗游戏直播", 99: "男生的统一话术", 100: "好兄弟话题"},
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_probe(directory: Path, groups: Path) -> tuple[list[dict], list[dict], dict]:
    complete = load_json(directory / "complete.json")
    if not complete.get("completed") or not complete.get("all_gates_passed"):
        raise ValueError("Probe has not completed with all inference gates passed")
    for filename, expected in complete["files_sha256"].items():
        if sha256(directory / filename) != expected:
            raise ValueError(f"Probe source changed after completion: {filename}")
    gates = load_json(directory / "gates.json")
    if not gates or any(not value.get("passed") for value in gates.values()):
        raise ValueError("A recorded probe inference gate failed")
    document = load_json(groups)
    if document.get("study_stage") != "exploratory_followup_to_failed_topic_groups":
        raise ValueError("Only the frozen second-stage behavior grouping is supported")
    if document.get("position_indexing") != "zero_based":
        raise ValueError("Expected zero-based group positions")
    manifest = load_json(directory / "manifest.json")
    # The exact group file is archived and hashed by the inference manifest.
    manifest_text = json.dumps(manifest, sort_keys=True)
    if sha256(groups) not in manifest_text:
        raise ValueError("Supplied group document is not the frozen probe input")
    return read_csv(directory / "predictions.csv"), read_csv(directory / "group_mass.csv"), {
        "complete_sha256": sha256(directory / "complete.json"),
        "groups_sha256": sha256(groups), "all_gates_passed": True,
        "manifest_sha256": sha256(directory / "manifest.json"),
    }


def build_case(args: argparse.Namespace) -> tuple[dict, list[dict], list[dict], dict]:
    arrays, meta, _, source_qa = load_evidence(args.source_dir)
    predictions, mass_rows, probe_qa = load_probe(args.probe_dir, args.groups_json)
    matches = [i for i, row in enumerate(meta) if int(row["sample_index"]) == args.sample_index]
    if len(matches) != 1:
        raise ValueError("Selected sample absent or ambiguous in source")
    index = matches[0]
    groups = load_json(args.groups_json)
    cases = [row for row in groups["cases"] if row["sample_index"] == args.sample_index]
    if len(cases) != 1:
        raise ValueError("Selected sample not present once in frozen grouping")
    group = cases[0]
    clicked = set(group["related_positions"])
    unclicked = set(group["control_positions"])
    last_five = set(np.flatnonzero(arrays["valid_mask"][index])[-5:].tolist())
    if clicked & unclicked or clicked | unclicked != last_five:
        raise ValueError("Frozen groups do not partition exactly the final five valid keys")
    labels = [row for row in predictions if int(row["sample_index"]) == args.sample_index and row["label"] == "is_click"]
    routes = {row["route"]: {key: (float(value) if key not in ("case_id", "route", "label") else value)
                              for key, value in row.items()} for row in labels}
    if set(routes) != set(ROUTES) or len(labels) != len(ROUTES):
        raise ValueError("Incomplete or duplicate prespecified routes")
    mass = {row["route"]: row for row in mass_rows if int(row["sample_index"]) == args.sample_index}
    if set(mass) != set(ROUTES):
        raise ValueError("Incomplete group masses")
    for route in routes.values():
        if route["truth"] != 1 or abs(bce(route["prediction"], 1) - route["bce"]) > 1e-12:
            raise ValueError("This figure requires verified positive-click BCE")
        if not np.isclose(sum(route[key] for key in ("related_mass", "control_mass", "receiver_mass")), 1, atol=2e-6, rtol=0):
            raise ValueError("Route group masses do not sum to one")
    full, revert = routes["full_hook"], routes["related_revert"]
    off_p, off_bce = full["source_local_off_prediction"], full["source_local_off_bce"]
    # Source CSV uses the original float32 shortest representation.
    off_p = float(np.float32(off_p))
    if abs(bce(off_p, 1) - off_bce) > 1e-10:
        raise ValueError("Local-off probability and source BCE disagree")
    gain = off_bce - full["bce"]
    attenuation = (revert["bce"] - full["bce"]) / gain if gain > 0 else -math.inf
    g_damage = routes["related_equal_mass"]["bce"] - full["bce"]
    c_damage = routes["control_equal_mass"]["bce"] - full["bce"]
    if gain <= 0 or attenuation < .5 or g_damage <= c_damage:
        raise ValueError("Selected case fails the frozen positive-case screening gate")
    before = arrays["weight_before"][index, 0]
    after = arrays["weight_after"][index, 0]
    g0, c0 = float(before[list(clicked)].sum()), float(before[list(unclicked)].sum())
    if not np.isclose(float(after[list(clicked)].sum()), full["related_mass"], atol=2e-7, rtol=0):
        raise ValueError("Source and probe full attention masses differ")
    if abs(revert["related_mass"] - g0) > 5e-4:
        raise ValueError("Actual BF16 rollback misses the original group mass")
    g_dose = full["related_mass"] - routes["related_equal_mass"]["related_mass"]
    c_dose = full["control_mass"] - routes["control_equal_mass"]["control_mass"]
    if min(g_dose, c_dose) <= 0 or abs(g_dose - c_dose) > 5e-4:
        raise ValueError("Equal-dose controls failed their actual-mass check")
    for route, fixed_key in (("related_equal_mass", "control_mass"), ("control_equal_mass", "related_mass")):
        if abs(routes[route][fixed_key] - full[fixed_key]) > 5e-4:
            raise ValueError("Equal-dose comparison changed the other group's mass")
    captions = load_captions(args.captions)
    vocab = load_json(args.source_dir / "frozen_inputs/meta_data.json")["action_vocab"]
    target_id = int(arrays["target_item_ids"][index])
    if metadata_status(target_id, captions)[0] != "ready":
        raise ValueError("Selected target has no verified readable official text")
    target = captions[target_id]
    target_text, target_field = caption_text(target)
    if len(target_text) > 70:
        raise ValueError("Target text needs an explicitly reviewed compact display")
    rows = []
    for position in np.flatnonzero(arrays["valid_mask"][index]):
        position = int(position)
        item_id = int(arrays["history_item_ids"][index, position])
        text = captions.get(item_id, {})
        raw_action, chinese_action = action_text(int(arrays["history_actions"][index, position]), vocab)
        membership = "G" if position in clicked else "C" if position in unclicked else "R"
        if position in last_five:
            if ("is_click" in raw_action.split("|")) != (membership == "G"):
                raise ValueError("Frozen group disagrees with raw recorded click action")
            if metadata_status(item_id, captions)[0] != "ready":
                raise ValueError("A shown history lacks verified readable official text")
        rows.append({"sample_index": args.sample_index, "position": position + 1, "group": membership,
                     "item_id": item_id, "final_video_id": item_id - 1,
                     "caption": text.get("caption", ""), "show_cover_text": text.get("show_cover_text", ""),
                     "display_summary": SUMMARIES.get(args.sample_index, {}).get(position + 1, ""),
                     "summary_kind": "human_reading_aid_not_model_input" if position in last_five else "",
                     "action_names": raw_action, "actions_chinese": chinese_action,
                     "weight_before": float(before[position]), "weight_full": float(after[position]),
                     "weight_difference": float(after[position] - before[position]),
                     "native_score": float(arrays["native_scores"][index, 0, position]),
                     "score_delta": float(arrays["delta"][index, 0, position]),
                     "shown": position in last_five})
    case = {"sample_index": args.sample_index, "target_id": target_id, "target_text": target_text,
            "target_text_field": target_field, "target_caption": target.get("caption", ""),
            "target_cover_text": target.get("show_cover_text", ""), "off_p": off_p, "off_bce": off_bce,
            "bce_gain": gain, "rollback_attenuation": attenuation, "g_damage": g_damage, "c_damage": c_damage,
            "g_dose": g_dose, "c_dose": c_dose, "g_mass_before": g0, "c_mass_before": c0,
            "g_share_within_recent_before": g0 / (g0 + c0),
            "g_share_within_recent_full": full["related_mass"] / (full["related_mass"] + full["control_mass"]),
            "g_positions": sorted(p + 1 for p in clicked), "c_positions": sorted(p + 1 for p in unclicked),
            "g_mean_position": float(np.mean(list(clicked))) + 1,
            "c_mean_position": float(np.mean(list(unclicked))) + 1,
            "routes": routes, "key_count": int(arrays["valid_mask"][index].sum()),
            "part_id": int(meta[index]["part_id"]), "row_offset": int(meta[index]["row_offset"])}
    qa = {"source": source_qa, "probe": probe_qa, "positive_case_gate_passed": True,
          "all_controls_retained": True, "caption_sha256": sha256(args.captions),
          "actual_g_rollback_error": revert["related_mass"] - g0,
          "actual_equal_dose_difference": g_dose - c_dose}
    return case, rows, labels, qa


def render(case: dict, rows: list[dict], output: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt
    from matplotlib.lines import Line2D

    font_path = Path("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc")
    if not font_path.exists():
        raise FileNotFoundError("CJK font is required for the evidence figure")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font_path)).get_name(),
                         "axes.unicode_minus": False, "pdf.fonttype": 42, "svg.fonttype": "path",
                         "font.size": 11, "text.color": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig = plt.figure(figsize=(14, 11), facecolor="white")
    fig.text(.045, .980, "近期点击历史的注意力与局部干预", fontsize=22, weight="bold", va="top")
    fig.text(.045, .925, "KuaiRand  ·  同一已训练 RankMixer–RISE  ·  target attention / head 0  ·  单个真实点击样本",
             fontsize=11, color=MUTED)
    fig.text(.045, .886, "当前候选：" + case["target_text"], fontsize=16, weight="bold")
    fig.text(.045, .855, "实际反馈：点击（y = 1）", fontsize=12)
    fig.text(.045, .814, "A  最近5条历史：先看内容，再看已记录反馈", fontsize=13, weight="bold")
    fig.text(.045, .787, "G：含点击反馈；C：无点击反馈。文案概述仅帮助阅读，不进入模型；原文完整保留于 CSV。", fontsize=10.5, color=MUTED)
    xs = [.047, .130, .399, .650, .790]
    for x, label in zip(xs, ("位置", "文案概述", "已记录反馈 / 分组", "修正前权重", "RISE 权重")):
        fig.text(x, .753, label, fontsize=10.5, color=MUTED)
    shown = [row for row in rows if row["shown"]]
    for i, row in enumerate(shown):
        y = .719 - .038 * i
        is_g = row["group"] == "G"
        color = BLUE if is_g else MUTED
        action = row["actions_chinese"].replace("曝光（六项反馈均为0）", "仅曝光")
        values = (str(row["position"]), row["display_summary"], action + "  /  " + row["group"],
                  f"{row['weight_before']:.2%}", f"{row['weight_full']:.2%}")
        for x, text in zip(xs, values):
            fig.text(x, y, text, fontsize=11, color=color, weight="bold" if is_g else "normal")
    fig.add_artist(Line2D([.045, .945], [.548, .548], transform=fig.transFigure, color=GRID, linewidth=1))
    fig.text(.045, .516, "B  G 组总权重（完整100条历史归一化）", fontsize=12.5, weight="bold")
    fig.text(.545, .516, "C  同一次点击预测", fontsize=12.5, weight="bold")
    full = case["routes"]["full_hook"]
    revert = case["routes"]["related_revert"]
    labels = ["局部修正关闭", "RISE 完整修正", "把 G 总权重退回原水平"]
    mass_values = [case["g_mass_before"] * 100, full["related_mass"] * 100, revert["related_mass"] * 100]
    prob_values = [case["off_p"] * 100, full["prediction"] * 100, revert["prediction"] * 100]
    axes = []
    for rect, values, xlabel, left_labels in (([.230, .333, .225, .149], mass_values, "G 占全部注意力（%）", True),
                                             ([.730, .333, .215, .149], prob_values, "点击概率（%）", False)):
        ax = fig.add_axes(rect)
        axes.append(ax)
        ax.set_xlim(0, max(values) * 1.21)
        ax.set_ylim(-.55, 2.55)
        ax.set_yticks([])
        ax.set_xlabel(xlabel, fontsize=10)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.xaxis.grid(True, color=GRID, linewidth=.6)
        ax.set_axisbelow(True)
        for j, (value, color, hatch) in enumerate(zip(values, [GRAY, BLUE, GOLD], ["", "", "///"])):
            y = 2 - j
            ax.barh(y, value, height=.43, color=color, edgecolor=INK, linewidth=.5, hatch=hatch)
            ax.text(value + max(values) * .035, y, f"{value:.2f}", va="center", fontsize=10.5, color=INK)
            fy = rect[1] + (y + .55) / 3.1 * rect[3]
            fig.text(.045 if left_labels else .545, fy, labels[j], va="center", fontsize=10.5)
    loss_statement = ("原有点击预测改善消失。" if case["rollback_attenuation"] >= 1 else
                      f"原有 BCE 收益削弱 {case['rollback_attenuation']:.1%}。")
    fig.text(.045, .273, "把近期点击历史 G 的总权重退回后，" + loss_statement, fontsize=13, weight="bold")
    g_loss_word = "增加" if case["g_damage"] >= 0 else "降低"
    c_loss_word = "增加" if case["c_damage"] >= 0 else "降低"
    fig.text(.045, .244, f"同剂量对照：从 G 撤回 {case['g_dose'] * 100:.3f} 个百分点，BCE {g_loss_word} {abs(case['g_damage']):.4f}；"
             f"从 C 撤回 {case['c_dose'] * 100:.3f} 个百分点，BCE {c_loss_word} {abs(case['c_damage']):.4f}。", fontsize=10.4)
    g_eq = case["routes"]["related_equal_mass"]
    c_eq = case["routes"]["control_equal_mass"]
    template = case["routes"]["position_template"]
    fig.text(.045, .215, f"对应点击概率：撤 G {g_eq['prediction']:.2%}；撤 C {c_eq['prediction']:.2%}；"
             f"留一等幅平均位置模板 {template['prediction']:.2%}。", fontsize=10.4)
    fig.text(.045, .180, f"G / C 平均位置：{case['g_mean_position']:.1f} / {case['c_mean_position']:.1f}，分别 {len(case['g_positions'])} / {len(case['c_positions'])} 条；"
             "同剂量移除的权重均转给其余历史，保持另一组总权重。", fontsize=9.6, color=MUTED)
    fig.text(.045, .154, f"G 在最近5条内部的占比为 {case['g_share_within_recent_before']:.1%} → {case['g_share_within_recent_full']:.1%}；"
             "这里检验其相对全部历史的增权，不声称只对已点击内容增权。", fontsize=9.6, color=MUTED)
    fig.text(.045, .126, "回退只恢复 G 的总权重，保留组内相对分布；比较均在同一模型中进行，并非独立训练 Baseline。", fontsize=9.6, color=MUTED)
    fig.text(.045, .099, "这是主题分组未通过后提出的探索性行为分组案例；不能据此推断主题识别、AM/PA 独立贡献或总体性能。", fontsize=9.6, color=MUTED)
    fig.text(.045, .072, f"sample {case['sample_index']} · part {case['part_id']} / row {case['row_offset']}。"
             "全部对照、原文和精确数值随图提供；概率未跨越 0.5，图示为该次真实点击下的损失改善。", fontsize=9.4, color=MUTED)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artists = list(fig.texts)
    for ax in axes:
        artists.extend(list(ax.texts) + [ax.xaxis.label] + list(ax.get_xticklabels()))
    boxes = [(artist.get_text(), artist.get_window_extent(renderer)) for artist in artists
             if artist.get_visible() and artist.get_text()]
    collisions = [[text, other] for i, (text, box) in enumerate(boxes)
                  for other, bbox in boxes[i + 1:] if box.overlaps(bbox)]
    outside = [text for text, box in boxes if box.x0 < 0 or box.y0 < 0
               or box.x1 > fig.bbox.x1 or box.y1 > fig.bbox.y1]
    if collisions or outside:
        plt.close(fig)
        raise ValueError(f"Figure layout failed: collisions={collisions}; outside={outside}")
    files = []
    for extension in ("pdf", "svg", "png"):
        path = output / f"mechanism_case_{case['sample_index']}.{extension}"
        fig.savefig(path, dpi=180, facecolor="white", metadata={"Creator": "UniRank audited local intervention study"})
        files.append({"name": path.name, "sha256": sha256(path)})
    plt.close(fig)
    return {"files": files, "text_collisions": collisions, "text_outside_canvas": outside,
            "axes_zero_origin": True, "figure_inches": [14, 11],
            "visual_inspection": "Required separately against exported PNG hash"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--groups-json", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/rise_attention_case_evidence/figures"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    case, rows, predictions, qa = build_case(args)
    if args.validate_only:
        print(json.dumps({"sample_index": case["sample_index"], "passed": True,
                          "rollback_attenuation": case["rollback_attenuation"],
                          "g_damage": case["g_damage"], "c_damage": case["c_damage"]}))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_qa = render(case, rows, args.output_dir)
    write_csv(args.output_dir / "mechanism_history_original.csv", rows)
    write_csv(args.output_dir / "mechanism_recent_five.csv", [row for row in rows if row["shown"]])
    write_csv(args.output_dir / "mechanism_predictions.csv", predictions)
    write_json(args.output_dir / "mechanism_case.json", case)
    write_json(args.output_dir / "numeric_qa.json", {"status": "PASS", **qa, **figure_qa,
               "script_sha256": sha256(Path(__file__)), "sample_index": case["sample_index"]})
    print(json.dumps({"sample_index": case["sample_index"], "output_dir": str(args.output_dir),
                      "rollback_attenuation": case["rollback_attenuation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
