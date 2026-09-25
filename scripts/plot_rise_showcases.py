"""Two concise, source-backed examples: negative evidence and local order reversal."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/rise_attention_showcases"
SOURCE = ROOT / "artifacts/rise_attention_semantic/kuairand_646410"
PROBE = ROOT / "artifacts/rise_attention_case_search/probe_646610"
METADATA = ROOT / "artifacts/rise_attention_case_search/recent_window_scan/metadata/selected_captions.json"
FONT = FontProperties(fname="/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc")
BOLD = FontProperties(fname="/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc")
BLUE, GRAY, INK, LIGHT, GOLD = "#245B86", "#8C949C", "#26323B", "#E1E6EA", "#B48935"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def text(fig, x, y, value, size=12, color=INK, bold=False, **kwargs):
    return fig.text(x, y, value, fontsize=size, color=color,
                    fontproperties=BOLD if bold else FONT, **kwargs)


def line(fig, y):
    fig.add_artist(Line2D([.055, .955], [y, y], color=LIGHT, linewidth=1, transform=fig.transFigure))


def export(fig, name, out):
    fig.canvas.draw()
    bounds = fig.bbox
    outside = []
    renderer = fig.canvas.get_renderer()
    for item in fig.texts:
        box = item.get_window_extent(renderer)
        if box.x0 < bounds.x0 or box.x1 > bounds.x1 or box.y0 < bounds.y0 or box.y1 > bounds.y1:
            outside.append(item.get_text())
    if outside:
        raise ValueError(f"Text outside figure: {outside}")
    rows = []
    for extension in ("pdf", "svg", "png"):
        path = out / f"{name}.{extension}"
        fig.savefig(path, dpi=230, facecolor="white", metadata={"Creator": "UniRank verified case study"})
        rows.append({"file": path.name, "sha256": sha(path)})
    plt.close(fig)
    return rows


def main():
    out = STUDY / "figures"
    out.mkdir(parents=True, exist_ok=True)
    complete = json.loads((PROBE / "complete.json").read_text())
    assert complete["completed"] and complete["all_gates_passed"]
    for file in ("predictions.csv", "group_mass.csv", "attention_full_hook.npz", "attention_position_template.npz"):
        assert sha(PROBE / file) == complete["files_sha256"][file]
    source_complete = json.loads((SOURCE / "complete.json").read_text())
    assert sha(SOURCE / "raw_attention.npz") == source_complete["files_sha256"]["raw_attention.npz"]
    raw = dict(np.load(SOURCE / "raw_attention.npz", allow_pickle=False))
    positional = dict(np.load(PROBE / "attention_position_template.npz", allow_pickle=False))
    data = {(int(row["sample_index"]), row["route"]): row
            for row in read_csv(PROBE / "predictions.csv") if row["label"] == "is_click"}
    caps = {int(row["item_id"]): row for row in json.loads(METADATA.read_text())}
    metadata = json.loads((ROOT / "datasets/KuaiRand_Video_Action/meta_data.json").read_text())
    actions = {int(value): key for key, value in metadata["action_vocab"].items()}
    needed = [int(raw["target_item_ids"][416]), int(raw["history_item_ids"][416, 97]),
              int(raw["target_item_ids"][24]), int(raw["history_item_ids"][24, 93]),
              int(raw["history_item_ids"][24, 94])]
    for identifier in needed:
        row = caps[identifier]
        assert row["record_usable"] and row["mapping_verified"] and row["duration_bucket_verified"]
        assert identifier == int(row["final_video_id"]) + 1
        for path_key, digest_key in (("source_range_path", "source_sha256"), ("verification_range_path", "verification_sha256")):
            assert sha(ROOT / row[path_key]) == row[digest_key]
    assert raw["target_item_ids"][416] == raw["history_item_ids"][416, 96]
    assert actions[int(raw["history_actions"][416, 96])] == "exposure"
    assert actions[int(raw["history_actions"][24, 93])] == "is_click|long_view"
    assert actions[int(raw["history_actions"][24, 94])] == "exposure"
    assert int(data[(416, "full_hook")]["truth"]) == 0
    assert int(data[(24, "full_hook")]["truth"]) == 1
    assert float(data[(416, "related_revert")]["bce"]) > float(data[(416, "full_hook")]["bce"])
    plt.rcParams.update({"font.family": FONT.get_name(), "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "svg.fonttype": "path", "font.size": 11})
    records = []

    # Case 1: an exact repeated video makes the evidence relation unambiguous.
    i = 416
    full, back = data[(i, "full_hook")], data[(i, "related_revert")]
    off_p = float(np.float32(full["source_local_off_prediction"]))
    on_p, back_p = float(full["prediction"]), float(back["prediction"])
    fig = plt.figure(figsize=(12, 7.1))
    text(fig, .055, .93, "高 attention，也可以是在利用负向证据", 22, bold=True)
    text(fig, .055, .882, "KuaiRand · 同一条视频的再次曝光 · 固定 RankMixer–RISE 模型", 11, color=GRAY)
    line(fig, .852)
    text(fig, .055, .798, "此前的历史记录  /  位置 97", 12, bold=True)
    text(fig, .57, .798, "当前候选  /  同一个视频 ID", 12, bold=True)
    title = "《逃跑吧少年》西游记联动视频"
    text(fig, .055, .746, title, 15)
    text(fig, .57, .746, title, 15)
    text(fig, .055, .695, "已有反馈：仅曝光，没有点击", 13, BLUE, True)
    text(fig, .57, .695, "本次真实反馈：没有点击（y = 0）", 13, BLUE, True)
    before = float(raw["weight_before"][i, 0, 96]) * 100
    after = float(raw["weight_after"][i, 0, 96]) * 100
    text(fig, .055, .635, f"这条历史的 attention：{before:.2f}% → {after:.2f}%", 17, BLUE, True)
    text(fig, .57, .635, f"点击预测：{100*off_p:.2f}% → {100*on_p:.2f}%", 17, BLUE, True)
    line(fig, .60)
    text(fig, .055, .55, "把这条历史的总权重退回去，会怎样？", 14, bold=True)
    ax = fig.add_axes([.39, .278, .49, .232])
    probs = np.array([off_p, on_p, back_p]) * 100
    labels = ["该处修正关闭", "RISE", "只回退这条历史的权重"]
    ax.barh([2, 1, 0], probs, height=.52, color=[GRAY, BLUE, GOLD], edgecolor=INK, linewidth=.5)
    ax.set_yticks([2, 1, 0], labels, fontproperties=FONT, fontsize=12)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.set_xlim(0, 75)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel("预测点击概率（%）", fontproperties=FONT, labelpad=7)
    ax.xaxis.grid(True, color=LIGHT)
    ax.set_axisbelow(True)
    for y, value in zip([2, 1, 0], probs):
        ax.text(value + 1, y, f"{value:.2f}%", va="center", fontsize=12, color=INK)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(LIGHT)
    text(fig, .055, .164, f"回退后，点击概率回升 {100*(back_p-on_p):.2f} 个百分点，偏离这次真实的“未点击”标签。", 13, bold=True)
    control = float(data[(i, "control_equal_mass")]["prediction"])
    position = float(data[(i, "position_template")]["prediction"])
    text(fig, .055, .114, f"同剂量近邻对照：回退另一款游戏记录后 {100*control:.2f}%；等幅平均位置模板：{100*position:.2f}%。", 10, GRAY)
    text(fig, .055, .072, "权重在完整100条历史上归一化；局部回退检验的是这条记录的作用，不能归因全部收益。", 10, GRAY)
    text(fig, .055, .037, "sample416 · 说明性选例；“关闭修正”不是独立训练Baseline。标题为官方文案主题概述，原文随图保存。", 9, GRAY)
    files = export(fig, "negative_evidence_416", out)
    records.append({"sample_index": i, "kind": "negative_evidence", "history_position": 97,
                    "same_video_verified": True, "weight_before": before / 100, "weight_after": after / 100,
                    "p_off": off_p, "p_full": on_p, "p_revert": back_p, "files": files})

    # Case 2: show exactly two neighboring histories; the newer one does not win.
    i = 24
    b = raw["weight_before"][i, 0, [93, 94]] * 100
    f = raw["weight_after"][i, 0, [93, 94]] * 100
    slot = int(np.flatnonzero(positional["sample_indices"] == i)[0])
    p = positional["weight_after"][slot, 0, [93, 94]] * 100
    assert b[0] < b[1] and p[0] < p[1] and f[0] > f[1]
    fig = plt.figure(figsize=(12, 7.1))
    text(fig, .055, .93, "较早的早餐条目，超过了更新的剧情条目", 22, bold=True)
    text(fig, .055, .883, "KuaiRand · 放大相邻的94—95位置 · 完整100条历史仍参与计算", 11, GRAY)
    text(fig, .055, .806, "当前候选：铜锣烧美食视频", 17, bold=True)
    text(fig, .65, .806, "真实反馈：点击（y = 1）", 13, BLUE, True)
    line(fig, .765)
    text(fig, .055, .707, "94  ·  3分钟手抓饼早餐做法", 14, BLUE, True)
    text(fig, .055, .659, "已点击、长观看  ·  位置较早", 11, GRAY)
    text(fig, .54, .707, "95  ·  《反转人生》剧情片段", 14, bold=True)
    text(fig, .54, .659, "仅曝光  ·  位置更新", 11, GRAY)
    ax = fig.add_axes([.11, .282, .80, .30])
    centers = np.array([0, 1.7, 3.4]); width=.30
    food, drama = np.array([b[0], p[0], f[0]]), np.array([b[1], p[1], f[1]])
    ax.bar(centers - width / 2, food, width=width, color=BLUE, edgecolor=INK, linewidth=.5, label="早餐条目（94）")
    ax.bar(centers + width / 2, drama, width=width, color=GRAY, edgecolor=INK, linewidth=.5, label="剧情条目（95）")
    ax.set_ylim(0, 5.7)
    ax.set_xlim(-.7, 4.1)
    ax.set_xticks(centers, ["修正前", "等幅平均位置模板", "RISE实际修正"], fontproperties=FONT, fontsize=12)
    ax.set_ylabel("原始 attention 权重（%）", fontproperties=FONT, labelpad=8)
    ax.yaxis.grid(True, color=LIGHT)
    ax.set_axisbelow(True)
    for x, left, right in zip(centers, food, drama):
        ax.text(x - width / 2, left + .09, f"{left:.2f}", ha="center", color=BLUE, fontsize=11)
        ax.text(x + width / 2, right + .09, f"{right:.2f}", ha="center", color=INK, fontsize=11)
    ax.legend(loc="upper left", frameon=False, prop=FONT, ncol=2, fontsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(LIGHT); ax.spines["bottom"].set_color(LIGHT)
    ax.annotate("局部排序翻转", xy=(3.4, 4.0), xytext=(3.4, 5.15), ha="center",
                fontproperties=FONT, fontsize=12, color=BLUE,
                arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1})
    text(fig, .055, .186, f"修正前：早餐 {b[0]:.2f}% < 剧情 {b[1]:.2f}%    →    RISE：早餐 {f[0]:.2f}% > 剧情 {f[1]:.2f}%", 13, bold=True)
    full = data[(i, "full_hook")]
    op = float(np.float32(full["source_local_off_prediction"])); fp = float(full["prediction"])
    rp = float(data[(i, "related_revert")]["prediction"])
    text(fig, .055, .127, f"整例 p(click)：{100*op:.2f}% → {100*fp:.2f}%；仅回退早餐条目时：{100*rp:.2f}%（该局部预测作用较小）。", 10.5, GRAY)
    text(fig, .055, .081, "这张局部图展示实际修正与位置模板的区别，不把整例收益全部归因于这两条历史。", 10, GRAY)
    text(fig, .055, .038, "sample24 · 说明性选例；原文、位置、反馈与全部对照随图保存。局部展示未重新归一化原权重。", 9, GRAY)
    files = export(fig, "local_order_24", out)
    records.append({"sample_index": i, "kind": "local_order", "positions": [94,95],
                    "before_weights": (b/100).tolist(), "position_template_weights": (p/100).tolist(),
                    "full_weights": (f/100).tolist(), "rank_reversal_verified": True,
                    "p_off": op, "p_full": fp, "p_revert": rp, "files": files})
    (out / "case_data.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    selected = [caps[identifier] for identifier in sorted(set(needed))]
    (out / "source_caption_records.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
    audit = {"numeric_status": "PASS", "visual_status": "PENDING", "script_sha256": sha(Path(__file__)),
             "caption_file_sha256": sha(METADATA), "displayed_records": len(selected),
             "scope": "Two intentionally selected examples; different explanatory roles; no population or component-necessity claim",
             "cases": records}
    (out / "figure_qa.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"rendered": True, "directory": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
