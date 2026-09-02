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
