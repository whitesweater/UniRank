# Report contract and QA notes

Audience: technical. Delivery: portable HTML. Markdown is the same narrative and table source for copying, not a second runtime.

Question: Why is HiFormer on QK-Video the only negative AUC cell in the accepted 20-cell RISE matrix?
Answer: position controls locate a major negative source in field interaction; detailed order/decay/phase causes remain unresolved.

Required structure mapping: title and technical summary; scope/metric definitions moved ahead of evidence; results and experimental design are interleaved in the placement section; uncertainty and limitations have a dedicated section; recommendations and remaining questions are merged; a paper-ready paragraph repeats the validated conclusion for direct reuse.

Visual contract: exact lookup and distinctions among comparison baselines are the purpose. Five full-width source-backed native tables have adjacent interpretation, signed absolute AUC differences, neutral numeric colors and explicit default sorting. The canonical report validator rejects table-only reports, so a single native horizontal bar comparison shows the four placement configurations against the same zero reference. It uses a diverging palette and explicit configuration labels, retains baseline AUC and LogLoss for inspection, and is followed by the exact table. Raw unrounded values are preserved in original CSV sources; displayed tables use eight decimal places for AUC/LogLoss.

No significance assertion is made from three heterogeneous RISE runs or one-seed position controls. Group intervals, if inspected in source CSV, concern fixed-model paired test predictions and are not evidence for training-seed stability. No claim of universal subgroup decline, proven overfitting, causal sparsity, or isolated decay/phase causality is made.

Additional current-turn audits: artifacts/hiformer_qk_failure_20260924/group_audit.json and artifacts/hiformer_qk_failure_20260924/site_audit.json.

Sources retain their historical filenames. All reader-facing method names are RISE. Existing reports, models and training jobs are unchanged.

Reproduce from repository root with the repository virtual environment:
`.venv/bin/python experiments/hiformer_qk_diagnostic/failure_analysis_20260924/build_report.py`

Package through the Data Analytics plugin's canonical `npm run report:deliver -- --input <artifact.json> --output <report.html>` command. Keep its final receipt as `delivery_receipt.json`. Do not manually edit generated HTML.
