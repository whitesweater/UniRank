#!/usr/bin/env python3
"""Collect only validated actual-compiled B/E selection on the fixed 512 rows.

B is an explicitly accepted, complete validated substage of failed job 643911.
E is from completed job 643949. No eager observation or AM/PA reconstruction is
read as mechanism evidence. All prediction equalities are recomputed from saved
compiled predictions rather than inferred from a filename or a passed flag.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from export_predictions import ROOT, LABELS, META_COLUMNS, atomic_json, identity_hash, sha256

IDS = tuple(META_COLUMNS) + tuple(LABELS)
KEYS = ("part_id", "row_offset")
SITES = tuple(f"unified_interaction_layers.{kind}_layers.{layer}"
              for layer in range(3) for kind in ("seq", "decode"))
LENGTH_GROUPS = ("0", "1-20", "21-50", "51-99", "100")
TOLERANCE = 1e-6
METRICS = (
    "valid_attention_rows", "allowed_pairs", "omega_mean", "omega_max", "tv_mean", "tv_max",
    "tv_bound_mean", "tv_bound_violation_max", "top_key_changed_fraction", "delta_mean", "delta_abs_mean",
    "delta_positive_fraction", "delta_negative_fraction", "native_logit_sd_mean", "delta_centered_sd_mean",
    "delta_native_sd_ratio_mean", "native_top_gap_mean", "native_top_order_certified_fraction",
    "top_order_certificate_violation_fraction", "native_padding_mass_mean", "corrected_padding_mass_mean",
    "delta_abs_mean_valid_keys", "delta_positive_fraction_valid_keys", "delta_negative_fraction_valid_keys",
)
OBSERVATION_META = ("site", "variant", "capture_route", "native_mask_present", "native_q_dtype", "native_k_dtype", "delta_dtype")
SUMMARY_METRICS = ("omega_mean", "tv_mean", "tv_bound_mean", "top_key_changed_fraction",
                   "delta_abs_mean_valid_keys", "delta_native_sd_ratio_mean", "delta_positive_fraction_valid_keys",
                   "native_top_order_certified_fraction", "native_padding_mass_mean", "corrected_padding_mass_mean",
                   "padding_mass_change")
JOBS = {"B": {"job_id": "643911", "state": "FAILED", "exit_code": "11:0"},
        "E": {"job_id": "643949", "state": "COMPLETED", "exit_code": "0:0"}}


def ordered_metadata(frame):
    if not set(IDS).issubset(frame):
        raise ValueError("Missing sample identity/label columns")
    if frame.duplicated(list(KEYS)).any():
        raise ValueError("Repeated source row identity")
    return frame.loc[:, list(IDS)].sort_values(list(KEYS)).reset_index(drop=True)


def assert_same_metadata(actual, reference):
    actual, reference = ordered_metadata(actual), ordered_metadata(reference)
    if len(actual) != len(reference) or any(not np.array_equal(actual[key], reference[key]) for key in IDS):
        raise ValueError("Source row IDs, metadata, or labels differ from frozen reference")


def length_groups(values):
    values = np.asarray(values)
    if not np.isfinite(values).all() or np.any(values != values.astype(np.int64)) or np.any((values < 0) | (values > 100)):
        raise ValueError("Effective history length must be an integer in [0,100]")
    return np.select([values == 0, values <= 20, values <= 50, values < 100],
                     LENGTH_GROUPS[:4], default="100")


def validate_prediction_equality(frame, reference_predictions, metadata, gate, variant):
    assert_same_metadata(frame, metadata)
    assert_same_metadata(reference_predictions, metadata)
    for name in ("compiled_capture_vs_compiled_plain", "compiled_plain_vs_failed_run_compiled"):
        record = gate[name]
        if not record["passed"] or not record["bitwise_equal"] or record["atol"] != 5e-4 or record["rtol"] != 5e-3:
            raise ValueError(f"{variant}: missing unchanged-threshold bitwise gate: {name}")
        if set(record["labels"]) != set(LABELS):
            raise ValueError("Incomplete prediction labels in gate")
        for label, result in record["labels"].items():
            if (not result["passed"] or result["rows"] != len(metadata) or result["failed_rows"] != 0
                    or result["max_abs"] != 0 or result["mean_abs"] != 0):
                raise ValueError(f"{variant}: inconsistent bitwise receipt for {label}")
    rows = frame.sort_values(list(KEYS)).reset_index(drop=True)
    old = reference_predictions.sort_values(list(KEYS)).reset_index(drop=True)
    for label in LABELS:
        plain = rows[f"{label}_compiled_plain"].to_numpy()
        captured = rows[f"{label}_compiled_capture"].to_numpy()
        frozen = old[f"{label}_compiled"].to_numpy()
        if (not np.isfinite(plain).all() or not np.isfinite(captured).all() or not np.isfinite(frozen).all()
                or np.any((plain < 0) | (plain > 1))
                or not np.array_equal(plain, captured) or not np.array_equal(plain, frozen)):
            raise ValueError(f"{variant}: actual saved compiled predictions fail equality for {label}")


def validate_observations(frame, metadata, variant):
    expected_columns = set(IDS) | set(OBSERVATION_META) | set(METRICS)
    if set(frame) != expected_columns:
        raise ValueError(f"Unexpected selection schema, including possible nonselection columns: {set(frame) ^ expected_columns}")
    if len(frame) != len(metadata) * len(SITES) or set(frame.site) != set(SITES):
        raise ValueError("Missing/excess sample-site observations")
    if frame.duplicated([*KEYS, "site"]).any():
        raise ValueError("Duplicate sample/site observation")
    if set(frame.variant) != {variant} or set(frame.capture_route) != {"compiled_tensor_capture"}:
        raise ValueError("Only the actual compiled tensor-capture route is accepted")
    if frame.native_mask_present.astype(bool).any():
        raise ValueError("Native HyFormer masking contract changed")
    if any(set(frame[column]) != {"torch.bfloat16"} for column in ("native_q_dtype", "native_k_dtype", "delta_dtype")):
        raise ValueError("Actual captured Q/K/delta dtype differs from BF16 reference")
    for site, group in frame.groupby("site"):
        assert_same_metadata(group, metadata)
        expected_rows = (2 * group.effective_length.to_numpy() if ".seq_layers." in site
                         else np.full(len(group), 4))
        if (not np.array_equal(group.valid_attention_rows, expected_rows)
                or not np.array_equal(group.allowed_pairs, expected_rows * 100)):
            raise ValueError("Valid-query count or native allowed-key coverage differs")
    numeric = frame.loc[:, list(METRICS)].to_numpy(dtype=np.float64)
    if np.isinf(numeric).any():
        raise ValueError("Infinite mechanism statistic")
    active = frame.valid_attention_rows > 0
    always_finite = [metric for metric in METRICS if not metric.endswith("valid_keys")]
    if not np.isfinite(frame.loc[active, always_finite].to_numpy(dtype=np.float64)).all():
        raise ValueError("Nonfinite statistic on a valid attention query")
    with_valid_keys = active & (frame.effective_length > 0)
    if not np.isfinite(frame.loc[with_valid_keys, list(METRICS)].to_numpy(dtype=np.float64)).all():
        raise ValueError("Nonfinite valid-query/valid-key statistic")
    if (frame.tv_bound_violation_max.max() > TOLERANCE
            or frame.top_order_certificate_violation_fraction.max() > TOLERANCE
            or ((frame.tv_mean - frame.tv_bound_mean) > TOLERANCE).any()):
        raise ValueError("TV bound or native-order preservation certificate violated")
    if ((frame.tv_mean - frame.tv_max) > TOLERANCE).any() or ((frame.omega_mean - frame.omega_max) > TOLERANCE).any():
        raise ValueError("Mean exceeds recorded per-example maximum")
    bounded = [metric for metric in METRICS if "fraction" in metric or "padding_mass" in metric]
    bounded += ["tv_mean", "tv_max", "tv_bound_mean"]
    for metric in bounded:
        valid = frame[metric].dropna().to_numpy()
        if np.any((valid < -TOLERANCE) | (valid > 1 + TOLERANCE)):
            raise ValueError(f"Statistic outside [0,1]: {metric}")
    for suffix in ("", "_valid_keys"):
        if ((frame[f"delta_positive_fraction{suffix}"] + frame[f"delta_negative_fraction{suffix}"]) > 1 + TOLERANCE).any():
            raise ValueError("Positive/negative correction fractions overlap")
    if ((frame.delta_positive_fraction + frame.delta_negative_fraction
         - frame.effective_length / 100) > TOLERANCE).any():
        raise ValueError("Nonzero correction fraction is inconsistent with zero padded-key channels")
    for metric in ("omega_mean", "omega_max", "delta_abs_mean", "delta_abs_mean_valid_keys",
                   "native_logit_sd_mean", "delta_centered_sd_mean", "delta_native_sd_ratio_mean", "native_top_gap_mean"):
        if (frame[metric].dropna() < -TOLERANCE).any():
            raise ValueError(f"Negative nonnegative statistic: {metric}")
    no_padding = frame.effective_length == 100
    if (frame.loc[no_padding, ["native_padding_mass_mean", "corrected_padding_mass_mean"]].abs().to_numpy() > TOLERANCE).any():
        raise ValueError("Nonzero padding mass with 100 valid history keys")
    return {"rows": len(frame), "unique_examples": len(metadata), "sites": len(SITES),
            "tv_bound_violation_max": float(frame.tv_bound_violation_max.max()),
            "top_order_certificate_violation_max": float(frame.top_order_certificate_violation_fraction.max()),
            "mathematical_tolerance": TOLERANCE,
            "examples_with_zero_valid_queries": int((frame.valid_attention_rows == 0).sum())}


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    fields = ("mean", "std", "min", "q10", "q25", "median", "q75", "q90", "q95", "q99", "max")
    if not len(values):
        return {"n_finite": 0, **{key: None for key in fields}}
    quantiles = np.quantile(values, [0, .1, .25, .5, .75, .9, .95, .99, 1])
    return {"n_finite": len(values), "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else None,
            **dict(zip(("min", "q10", "q25", "median", "q75", "q90", "q95", "q99", "max"), map(float, quantiles)))}


def summarize(frame, by_length=False):
    summaries, quantiles = [], []
    for variant in ("B", "E"):
        for site in SITES:
            for length in LENGTH_GROUPS if by_length else ("all",):
                group = frame.loc[(frame.variant == variant) & (frame.site == site)]
                if by_length:
                    group = group.loc[group.length_group == length]
                identity = {"variant": variant, "site": site,
                            "site_kind": "sequence" if ".seq_layers." in site else "decode",
                            "layer": int(site.rsplit(".", 1)[1]) + 1, "length_group": length}
                row = {**identity, "n_examples": len(group), "n_users": group.user_id.nunique(),
                       "n_examples_with_valid_queries": int((group.valid_attention_rows > 0).sum()),
                       "attention_rows_total": float(group.valid_attention_rows.sum()),
                       "native_allowed_pairs_total": float(group.allowed_pairs.sum())}
                for metric in [*METRICS, "padding_mass_change"]:
                    stats = describe(group[metric])
                    quantiles.append({**identity, "metric": metric, "n_examples": len(group), **stats})
                    if metric in SUMMARY_METRICS:
                        row[f"{metric}_mean_over_examples"] = stats["mean"]
                        row[f"{metric}_median_over_examples"] = stats["median"]
                summaries.append(row)
    return pd.DataFrame(summaries), pd.DataFrame(quantiles)


def parse_accounting(text):
    records = {}
    for line in text.strip().splitlines():
        fields = line.split("|")
        if not fields or not fields[0].isdigit():
            continue
        if len(fields) < 5:
            raise ValueError("Malformed scheduler accounting record")
        records[fields[0]] = dict(zip(("job_id", "state", "exit_code", "elapsed", "node"), fields[:5]))
    for variant, expected in JOBS.items():
        record = records.get(expected["job_id"])
        if record is None or any(record[key] != expected[key] for key in ("state", "exit_code")):
            raise ValueError(f"Scheduler evidence differs from the locked {variant} source policy")
    return records


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
                      + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def main():
    base = ROOT / "artifacts/rise_sample_analysis/mechanism"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=base / "observed_643858")
    parser.add_argument("--b-source", type=Path, default=base / "precision_full_643911")
    parser.add_argument("--e-source", type=Path, default=base / "precision_E_643949")
    parser.add_argument("--accounting-file", type=Path, help="Use a previously archived scheduler snapshot")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/rise_sample_analysis/results/compiled_mechanism_v1")
    args = parser.parse_args()
    reference, output = args.reference_dir.resolve(), args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "experiments/rise_sample_analysis/results"):
        parser.error("Output must be below this study's results directory")
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing collector result")
    reference_manifest = json.loads((reference / "manifest.json").read_text())
    metadata = pd.read_parquet(reference / "sample_rows.parquet")
    if len(metadata) != 512 or metadata.duplicated(list(KEYS)).any():
        raise ValueError("Expected exactly 512 unique frozen source examples")
    length_groups(metadata.effective_length)
    if sha256(reference / "sample_batches.pt") != reference_manifest["sample_cache_sha256"]:
        raise ValueError("Frozen sample cache changed")
    accounting = (args.accounting_file.read_text() if args.accounting_file else subprocess.check_output(
        ["sacct", "-j", "643911,643949", "--format=JobIDRaw,State,ExitCode,Elapsed,NodeList", "--parsable2", "--noheader"], text=True))
    scheduler = parse_accounting(accounting)
    paths = {"B": args.b_source.resolve(), "E": args.e_source.resolve()}
    sources, frames, files = {}, [], [reference / "manifest.json", reference / "sample_rows.parquet"]
    for variant, directory in paths.items():
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if (manifest["failed_manifest_sha256"] != sha256(reference / "manifest.json")
                or manifest["sample_cache_sha256"] != reference_manifest["sample_cache_sha256"]
                or manifest["limit_batches"] != 8 or not manifest["compiled_selection_capture_only"]
                or not manifest["am_pa_internal_states_not_recovered"]
                or manifest["original_atol_unchanged"] != 5e-4 or manifest["original_rtol_unchanged"] != 5e-3):
            raise ValueError("Source manifest does not match the frozen compiled-selection contract")
        if manifest["original_observer_sha256"] != reference_manifest["source_sha256"]["experiments/rise_sample_analysis/scripts/observe_mechanism.py"]:
            raise ValueError("Observer provenance differs")
        assert_same_metadata(pd.read_parquet(directory / "sample_rows.parquet"), metadata)
        gate_path = directory / f"{variant}_diagnosis.json"
        gate = json.loads(gate_path.read_text())
        previous_gate_path = reference / f"{variant}_prediction_gate.json"
        if gate["checkpoint_sha256"] != json.loads(previous_gate_path.read_text())["checkpoint_sha256"]:
            raise ValueError("Checkpoint evidence differs")
        prediction_path = directory / f"{variant}_four_route_predictions.parquet"
        columns = list(IDS) + [f"{label}_{mode}" for label in LABELS for mode in ("compiled_plain", "compiled_capture")]
        predictions = pq.read_table(prediction_path, columns=columns).to_pandas()
        old_path = reference / f"{variant}_prediction_check.parquet"
        old = pq.read_table(old_path, columns=list(IDS) + [f"{label}_compiled" for label in LABELS]).to_pandas()
        validate_prediction_equality(predictions, old, metadata, gate, variant)
        observation_path = directory / f"{variant}_compiled_selection_validated.parquet"
        frame = pd.read_parquet(observation_path)
        validation = validate_observations(frame, metadata, variant)
        if variant == "E":
            completion_path = directory / "diagnosis_complete.json"
            completion = json.loads(completion_path.read_text())
            if not completion["diagnostic_run_completed"] or not completion["variants"]["E"]["compiled_capture_vs_compiled_plain"]["bitwise_equal"]:
                raise ValueError("Completed E source lacks its diagnostic receipt")
            files.append(completion_path)
        frame["length_group"] = length_groups(frame.effective_length)
        frame["padding_mass_change"] = frame.corrected_padding_mass_mean - frame.native_padding_mass_mean
        frames.append(frame)
        sources[variant] = {"directory": str(directory.relative_to(ROOT)),
            "scheduler": scheduler[JOBS[variant]["job_id"]], "validation": validation,
            "substage_prediction_identity_recomputed": True,
            "checkpoint_sha256": gate["checkpoint_sha256"], "diagnostic_script_sha256": manifest["script_sha256"],
            "device": manifest["device"],
            "acceptance": ("Complete, bitwise-validated B substage retained from overall FAILED 643911; E absent in that job"
                           if variant == "B" else "Independently COMPLETED E-only job 643949"),
            "whole_job_succeeded": variant == "E"}
        files += [manifest_path, directory / "sample_rows.parquet", gate_path, prediction_path,
                  observation_path, previous_gate_path, old_path]
        files += sorted((ROOT / "logs").glob(f"*{JOBS[variant]['job_id']}*.out"))
    if sources["B"]["diagnostic_script_sha256"] != sources["E"]["diagnostic_script_sha256"]:
        raise ValueError("B/E diagnostic code differs")
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in files}
    combined = pd.concat(frames, ignore_index=True)
    site_summary, site_quantiles = summarize(combined)
    length_summary, length_quantiles = summarize(combined, by_length=True)
    group_meta = metadata.copy()
    group_meta["length_group"] = length_groups(group_meta.effective_length)
    counts = []
    for length in LENGTH_GROUPS:
        rows = group_meta.loc[group_meta.length_group == length]
        record = {"length_group": length, "n_examples": len(rows), "n_users": rows.user_id.nunique()}
        for label in LABELS:
            record[f"{label}_positive"] = int(rows[label].sum())
            record[f"{label}_positive_rate"] = float(rows[label].mean()) if len(rows) else None
        counts.append(record)
    paired = []
    for site in SITES:
        b = combined.loc[(combined.variant == "B") & (combined.site == site)].sort_values(list(KEYS))
        e = combined.loc[(combined.variant == "E") & (combined.site == site)].sort_values(list(KEYS))
        for metric in SUMMARY_METRICS:
            stats = describe(e[metric].to_numpy() - b[metric].to_numpy())
            paired.append({"site": site, "metric": metric, "comparison": "E_minus_B_descriptive", **stats})
    output.mkdir(parents=True, exist_ok=False)
    site_summary.to_csv(output / "site_summary.csv", index=False)
    site_quantiles.to_csv(output / "site_quantiles.csv", index=False)
    length_summary.to_csv(output / "length_site_summary.csv", index=False)
    length_quantiles.to_csv(output / "length_site_quantiles.csv", index=False)
    pd.DataFrame(counts).to_csv(output / "length_sample_counts.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "paired_site_differences.csv", index=False)
    group_meta.to_csv(output / "sample_identities.csv", index=False)
    (output / "scheduler_accounting.txt").write_text(accounting)
    atomic_json(output / "sources.json", {"variants": sources, "source_sha256": hashes})
    table = []
    for _, row in site_summary.iterrows():
        table.append([row.variant, f"{row.site_kind}-{row.layer}", row.n_examples,
            *[f"{row[f'{metric}_mean_over_examples']:.6f}" for metric in
              ("omega_mean", "tv_mean", "top_key_changed_fraction", "native_padding_mass_mean", "corrected_padding_mass_mean")]])
    readout = """# 已验证compiled selection：HyFormer–MerRec固定512行诊断

本报告只使用同一次compiled前向捕获的实际Q/K/Δ，以FP32重构pre-dropout softmax selection。B为content-only，E为完整RISE；B不是原生Baseline。本报告不包含旧eager机制观测、AM/PA重算或内部控制状态结论。

B来自643911：整体job为FAILED 11:0，B子阶段已完成并逐行bitwise验证；E来自独立成功的643949（COMPLETED 0:0）。本次再次读取逐行compiled预测，确认B/E各512行、5标签的capture与plain以及冻结compiled参照完全相等。两variant均有512×6=3072条唯一site记录。

## 站点均值

以下先对每个样本内部head/query取源统计，再等权平均样本。它不是将所有query/key pair池化后的均值。每组完整分位数与有限值数量见同目录CSV。Padding质量保留原生允许的padding keys；仅无效history query不进入平均。

""" + markdown_table(["设置", "站点", "样本量", "Ω均值", "TV均值", "top-key改变比例", "原生padding mass", "加Δ后padding mass"], table)
    readout += "\n\n## 固定长度组样本量\n\n" + markdown_table(
        ["有效历史长度", "交互行", "用户数"], [[row["length_group"], row["n_examples"], row["n_users"]] for row in counts])
    readout += """

## 解释与限制

- 采样框为两历史rank的前四个原始批次，8个part各抽64行；这是固定诊断样本，不是全测试集均匀代表样本。长度100是截断后的有效长度。
- Ω/TV比较是在各自已训练模型、固定表示与native scores下添加Δ的条件比较；并不是native baseline模型对RISE的性能比较。B/E之间还存在分别训练导致的表示差异，因此E−B机制量差异只是描述性比较，不是AM/PA净因果效应。
- 大TV或较多top-key改变说明选择分布改变较强，不能直接推出AUC改善。Attention变化也不等同于对某个动作标签预测的因果贡献。
- TV上界和top-order证书按1e−6数值容差核验。表内分位数描述512个样本的site级平均量，不是所有head/query个体的分布，也不是训练seed置信区间。
- 零有效query的self-attention均值保留缺失值，不伪填零；空长度组不伪造样本。每组正例数量/比例在length_sample_counts.csv完整给出。

本目录为selection-only结果；内部AM/PA状态若后续另行验证，必须作为独立证据加入，不能从本表推断其已被观测。
"""
    (output / "readout.md").write_text(readout)
    for relative, expected in hashes.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError("Source changed during collection")
    summary = {"dataset": "MerRec_Action", "model": "HyFormer", "seed": 20262028,
               "examples": 512, "unique_users": int(metadata.user_id.nunique()), "sites": list(SITES),
               "observation_rows_per_variant": 3072, "variants": sources,
               "row_identity_sha256": identity_hash({key: metadata[key].to_numpy() for key in IDS}),
               "prediction_bitwise_equal_recomputed": True, "selection_only": True,
               "no_eager_or_am_reconstruction_statistics_used": True,
               "sample_weighting": "equal interaction examples within each site; invalid history queries excluded by source",
               "length_bins": list(LENGTH_GROUPS), "math_tolerance": TOLERANCE,
               "collector_sha256": sha256(Path(__file__)), "collected_utc": datetime.now(timezone.utc).isoformat(),
               "population_representative": False, "independent_training_replicates": 1}
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "complete.json", {"accepted": True, "examples": 512,
                "source_whole_job_states": {variant: sources[variant]["scheduler"]["state"] for variant in sources},
                "b_failed_job_substage_explicitly_preserved": True,
                "artifact_sha256": {path.name: sha256(path) for path in output.iterdir() if path.is_file()}})
    print(json.dumps({"accepted": True, "output": str(output), "examples": 512,
                      "unique_users": summary["unique_users"], "observations": len(combined)}, indent=2))


if __name__ == "__main__":
    main()
