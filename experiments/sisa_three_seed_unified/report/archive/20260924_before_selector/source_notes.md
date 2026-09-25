# Report source notes

- Audience: technical.
- Delivery mode: portable HTML from the canonical artifact contract.
- Required structure mapping: title; technical summary; key visual findings; scope/definitions; experimental design; methodology; limitations; reporting rules; further questions.
- Paper source: UniRank arXiv:2607.19987 Table 2, four-decimal values, materialized in sources/paper_table.csv. TAAC-25 maps to TencentGR_10M_Action.
- Baseline source: local non-SISA seed20262027 results selected from strict and expansion archives.
- SISA sources: seed20262027 is the matching strict+expansion union; seed20262028 and seed20262029 are complete 2×H100 studies.
- Statistical boundary: descriptive three-point mean/sample-SD/range only because seed20262027 differs in hardware and per-GPU batch.
- Exact lookup additions: a 4×4 AUC matrix shows three-seed mean first and raw-scale delta vs baseline second; a 16-row long table shows all three seed values, mean, sample SD, baseline, and delta.
- Reader pagination: the portable delivery wrapper uses a 16-row table page so the four complete model groups remain visible together.
- Omitted visual: logloss remains a table because label base rates create heterogeneous absolute scales; a cross-label chart would overstate comparability.
- Repeated chart-family audit: no repetition; one comparison bar and one relationship scatter.

## 2026-09-24 integrated 20-cell section

- The accepted report section now contains five models and four datasets (20 cells, 85 labels). The retained prior sections and charts are marked as a historical 16-cell snapshot.
- Primary sources: 16-cell original `results/cell_summary.csv`, OneTrans selected `results/selected_macro_summary.csv`, and the ten-run audited `results/final_comparison.csv`. The reproducible join is `report/build_integrated_evidence.py`; derived exact values are in `results/integrated_cells.csv`.
- Each 20262027/28/29 latest value selects the newest completed, audited run for that logical seed, regardless of whether AUC improved. The original three-seed mean remains visible. Seed 20262030 exists for only two MerRec cells and is shown separately, excluded from every comparable three-seed mean. The latest 20-cell aggregate is descriptive because the eight repeats were selected after observing low results.
- AUC movement and LogLoss improvement encode favorable direction as green, unfavorable as red, exact zero as neutral; fourth seed values are blue. Exact lookup tables replace charts for the new section because individual seed values and eight selected repeats are the question.
- Three added static fallback tables were repaired from canonical display strings because the packaged fallback rounds numeric-like values to two decimals and limits rows to fifteen. `integrated_qa.json` confirms 5/5/10 visible rows, seven-digit display, both color surfaces, browser verification at 1440/390 widths, and source-dialog interaction.
- `archive/20260924_before_20cell_merge/` contains the complete unchanged prior artifact, HTML, receipt, chart map, and source notes.

## Matrix layout revision

- The two primary integrated tables are transposed to five model rows by four dataset columns. Each primary matrix cell shows Baseline, the latest three-seed mean, and original and repeated values when that seed was rerun; seed30 appears where available. The underlying 20-row source dataset preserves all three per-seed values, and prior sections are unchanged.

## Visual width repair

- A rendered desktop screenshot exposed a 1,000,000px enhanced table width despite structural browser verification passing. The forced `width:100%` on the native table conflicted with its intrinsic scroll content width. Removed that override and visually confirmed all five columns at 1440px; the 390px view scrolls horizontally and now states that interaction.
