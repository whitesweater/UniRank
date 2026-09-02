#!/usr/bin/env node

import { existsSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const inputPath = resolve(scriptDirectory, "artifact.json");
const outputPath = resolve(scriptDirectory, "report.html");

function findReportBuilderDirectory() {
  const cacheRoot = resolve(
    homedir(),
    ".codex/plugins/cache/openai-curated-remote/data-analytics",
  );
  if (!existsSync(cacheRoot)) {
    throw new Error(`Data Analytics plugin cache not found: ${cacheRoot}`);
  }

  const candidates = readdirSync(cacheRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => resolve(cacheRoot, entry.name, "skills/build-report/scripts"))
    .filter((candidate) => existsSync(resolve(candidate, "build_portable_artifact.mjs")))
    .sort()
    .reverse();

  if (!candidates.length) {
    throw new Error("No installed Data Analytics portable report builder was found.");
  }
  return candidates[0];
}

const reportBuilderDirectory = findReportBuilderDirectory();
const buildModule = await import(
  pathToFileURL(resolve(reportBuilderDirectory, "build_portable_artifact.mjs"))
);
const deliveryModule = await import(
  pathToFileURL(resolve(reportBuilderDirectory, "deliver_portable_artifact.mjs"))
);

const { buildPortableArtifact, readPackagedReaderRuntime } = buildModule;
const { deliverPortableArtifact } = deliveryModule;

const reportTheme = String.raw`
<style id="unirank-report-theme">
  :root,
  :root[data-theme="light"],
  :root[data-theme="dark"] {
    color-scheme: light;
    --unirank-navy: #102b57;
    --unirank-navy-soft: #315078;
    --unirank-ink: #253653;
    --unirank-muted: #64748b;
    --unirank-rule: #d5dee9;
    --unirank-rule-strong: #183864;
    --unirank-green: #087a45;
    --unirank-red: #b43a35;
    --unirank-hover: #f4f7f6;
    --unirank-serif: "Noto Serif CJK SC", "Source Han Serif SC", "Songti SC", STSong, Georgia, serif;
    --unirank-sans: "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
    --ds-bg: #ffffff;
    --ds-surface: #ffffff;
    --ds-surface-raised: #ffffff;
    --ds-border-subtle: var(--unirank-rule);
    --ds-border-strong: var(--unirank-rule-strong);
    --ds-text-primary: var(--unirank-ink);
    --ds-text-secondary: var(--unirank-muted);
    --ds-green: var(--unirank-green);
    --ds-red: var(--unirank-red);
    --ds-chart-series-green: var(--unirank-green);
    --ds-chart-series-red: var(--unirank-red);
    --ds-chart-series-blue: var(--unirank-navy);
    --ds-chart-series-neutral: #9aa7b8;
    --ds-font: var(--unirank-sans);
    --ds-font-heading: var(--unirank-serif);
    --ds-report-content-max-width: 1080px;
    --ds-chart-body-height: 340px;
  }

  html { background: #ffffff; }
  body {
    background: #ffffff;
    color: var(--unirank-ink);
    font-family: var(--unirank-sans);
    -webkit-font-smoothing: antialiased;
  }

  .dashboard-shell.report-shell {
    background: #ffffff;
    min-height: 100vh;
  }

  .dashboard-shell.report-shell > :not(.analytics-top-bar):not(.copy-toast):not(dialog):not(.unified-chart-detail-page) {
    width: min(1080px, calc(100% - 64px));
  }

  .analytics-top-bar {
    position: sticky;
    top: 0;
    z-index: 20;
    width: 100%;
    margin: 0;
    padding: 0 max(32px, calc((100% - 1080px) / 2));
    border-bottom: 1px solid var(--unirank-rule);
    background: rgba(255, 255, 255, 0.96);
    backdrop-filter: blur(12px);
    box-sizing: border-box;
  }

  .analytics-top-bar .page-title-edit-target h1,
  .analytics-top-bar-title {
    color: var(--unirank-navy);
    font-family: var(--unirank-serif);
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 0.01em;
  }

  .analytics-top-bar-freshness,
  .snapshot-status {
    color: var(--unirank-muted);
    font-size: 12px;
    letter-spacing: 0.04em;
  }

  .report-block-stack {
    gap: 0;
    padding-top: 54px;
    padding-bottom: 88px;
  }

  .report-stack-item,
  .analytics-layout-item {
    margin-bottom: 34px;
  }

  .report-markdown-editor.markdown-render {
    color: var(--unirank-ink);
    font-size: 15px;
    line-height: 1.78;
  }

  .report-markdown-editor.markdown-render h1,
  .report-markdown-editor.markdown-render h2,
  .report-markdown-editor.markdown-render h3,
  .panel h2,
  .panel-header h2 {
    color: var(--unirank-navy);
    font-family: var(--unirank-serif);
    font-weight: 600;
    letter-spacing: -0.02em;
  }

  [data-layout-block-id="title"] {
    max-width: 960px;
    margin-bottom: 18px;
  }

  [data-layout-block-id="title"] .report-markdown-editor.markdown-render h1 {
    max-width: 920px;
    margin: 0 0 18px;
    font-size: clamp(38px, 4.1vw, 54px);
    line-height: 1.14;
    letter-spacing: -0.035em;
  }

  [data-layout-block-id="title"] .report-markdown-editor.markdown-render p {
    max-width: 820px;
    color: var(--unirank-navy-soft);
    font-family: var(--unirank-serif);
    font-size: 18px;
    line-height: 1.7;
  }

  [data-layout-block-id="technical_summary"] {
    max-width: 850px;
    margin-bottom: 38px;
  }

  [data-layout-block-id="technical_summary"] p {
    color: var(--unirank-muted);
  }

  .metric-card-layout {
    padding-top: 4px;
  }

  .report-stack-item-metric-card {
    margin-bottom: 40px;
    border-top: 2px solid var(--unirank-rule-strong);
    border-bottom: 2px solid var(--unirank-rule-strong);
  }

  .report-stack-item-metric-card + .report-stack-item-metric-card {
    border-left: 1px solid var(--unirank-rule);
  }

  .analytics-layout-item-shell > .report-metric-card,
  .report-metric-card {
    min-height: 112px;
    padding: 20px 26px;
    border: 0;
    border-radius: 0;
    background: #ffffff;
    box-shadow: none;
  }

  .report-metric-card .kpi-label {
    color: var(--unirank-muted);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.04em;
    line-height: 1.45;
  }

  .report-metric-card .kpi-value {
    color: var(--unirank-green);
    font-family: var(--unirank-serif);
    font-size: clamp(29px, 3vw, 40px);
    font-variant-numeric: tabular-nums lining-nums;
    font-weight: 600;
    letter-spacing: -0.035em;
  }

  [data-layout-block-id="baseline_agreement"],
  [data-layout-block-id="expansion_takeaway"],
  [data-layout-block-id="strict_takeaway"],
  [data-layout-block-id="weight_boundary"] {
    padding: 4px 0 4px 28px;
    border-left: 3px solid var(--unirank-navy);
  }

  [data-layout-block-id="expansion_takeaway"] {
    border-left-color: var(--unirank-green);
  }

  .report-markdown-editor.markdown-render h2 {
    margin: 0 0 14px;
    font-size: 24px;
    line-height: 1.3;
  }

  .report-markdown-editor.markdown-render h3 {
    margin: 24px 0 10px;
    font-size: 18px;
  }

  .report-markdown-editor.markdown-render p,
  .report-markdown-editor.markdown-render li {
    max-width: 900px;
  }

  .report-markdown-editor.markdown-render code {
    padding: 0.12em 0.35em;
    border: 1px solid #e0e6ed;
    border-radius: 3px;
    background: #f7f9fb;
    color: var(--unirank-navy-soft);
  }

  .chart-panel,
  .table-panel,
  .viz-card {
    overflow: hidden;
    border: 0;
    border-top: 2px solid var(--unirank-rule-strong);
    border-bottom: 1px solid var(--unirank-rule-strong);
    border-radius: 0;
    background: #ffffff;
    box-shadow: none;
  }

  .panel-header,
  .chart-panel .panel-header,
  .table-panel .panel-header {
    padding: 22px 0 16px;
    border-bottom: 1px solid var(--unirank-rule);
  }

  .panel-header h2,
  .panel h2 {
    font-size: 22px;
    line-height: 1.35;
  }

  .panel-header p,
  .panel > p,
  .viz-card-subtitle {
    color: var(--unirank-muted);
    font-size: 13px;
    line-height: 1.55;
  }

  .chart-frame {
    padding: 22px 0 12px;
  }

  .report-block-stack .chart-frame {
    min-height: 340px;
  }

  .recharts-cartesian-grid line {
    stroke: #e1e7ee;
  }

  .recharts-text,
  .recharts-label {
    fill: var(--unirank-muted);
    font-family: var(--unirank-sans);
    font-size: 11px;
  }

  .heatmap-grid-panel {
    flex-direction: column;
    padding: 22px 0 12px;
    box-sizing: border-box;
  }

  .report-block-stack .viz-card .chart-frame--heatmap {
    flex: 0 0 390px !important;
    height: 390px !important;
    min-height: 390px !important;
    max-height: 390px !important;
  }

  .report-block-stack .viz-card:has(.chart-frame--heatmap) .chart-body-measure {
    flex: 0 0 390px !important;
    height: 390px !important;
    min-height: 390px !important;
    max-height: 390px !important;
  }

  .heatmap-grid {
    flex: 1 1 auto;
    gap: 5px;
    min-height: 0;
  }

  .heatmap-grid > [hidden] {
    display: none !important;
  }

  .unirank-heatmap-controls {
    flex: 0 0 auto;
    display: flex;
    gap: 8px 14px;
    align-items: center;
    flex-wrap: wrap;
    margin: 0 0 12px;
    padding: 8px 10px;
    border: 1px solid var(--unirank-rule);
    background: #f8fafc;
    color: var(--unirank-muted);
    font-size: 11px;
  }

  .unirank-collapse-help {
    color: var(--unirank-ink);
    font-weight: 600;
  }

  .unirank-collapse-status {
    font-variant-numeric: tabular-nums lining-nums;
  }

  .unirank-collapse-reset,
  .unirank-restore-chip,
  .heatmap-axis-collapse {
    border: 1px solid var(--unirank-rule);
    background: #ffffff;
    color: var(--unirank-navy-soft);
    font: inherit;
    cursor: pointer;
  }

  .unirank-collapse-reset {
    margin-left: auto;
    padding: 4px 8px;
  }

  .unirank-collapse-reset:hover,
  .unirank-collapse-reset:focus-visible,
  .unirank-restore-chip:hover,
  .unirank-restore-chip:focus-visible,
  .heatmap-axis-collapse:hover,
  .heatmap-axis-collapse:focus-visible {
    border-color: var(--unirank-navy-soft);
    color: var(--unirank-navy);
    outline: 2px solid rgba(49, 80, 120, 0.18);
    outline-offset: 1px;
  }

  .unirank-hidden-items {
    flex: 1 0 100%;
    display: flex;
    gap: 6px;
    align-items: center;
    flex-wrap: wrap;
  }

  .unirank-hidden-items[hidden] {
    display: none !important;
  }

  .unirank-restore-chip {
    padding: 3px 7px;
    border-radius: 999px;
  }

  .heatmap-grid > strong,
  .heatmap-grid > b {
    display: flex;
    gap: 4px;
    align-items: center;
    justify-content: space-between;
    color: var(--unirank-navy-soft);
    font-family: var(--unirank-sans);
    font-size: 12px;
    font-weight: 600;
  }

  .heatmap-axis-collapse {
    display: inline-grid;
    width: 18px;
    height: 18px;
    flex: 0 0 18px;
    padding: 0;
    place-items: center;
    border-radius: 50%;
    line-height: 1;
    opacity: 0.42;
  }

  .heatmap-grid > strong:hover .heatmap-axis-collapse,
  .heatmap-grid > b:hover .heatmap-axis-collapse,
  .heatmap-axis-collapse:focus-visible {
    opacity: 1;
  }

  .heatmap-axis-collapse:disabled {
    cursor: not-allowed;
    opacity: 0.18;
  }

  .heatmap-cell {
    position: relative;
    display: grid;
    min-height: 38px;
    place-items: center;
    border: 1px solid rgba(16, 43, 87, 0.12);
    border-radius: 2px;
    box-sizing: border-box;
  }

  .heatmap-cell::after {
    content: attr(data-unirank-label);
    color: var(--unirank-cell-text, #ffffff);
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    font-variant-numeric: tabular-nums lining-nums;
    font-weight: 700;
    letter-spacing: -0.02em;
  }

  .heatmap-cell[data-unirank-missing="true"] {
    border-color: var(--unirank-rule);
    background: #f1f4f7 !important;
    pointer-events: none;
  }

  .heatmap-cell[data-unirank-missing="true"]::after {
    color: #9aa7b8;
  }

  .heatmap-cell[data-unirank-source="digai"] {
    box-shadow: inset 0 0 0 2px rgba(16, 43, 87, 0.72);
  }

  .unirank-heatmap-legend {
    flex: 0 0 auto;
    display: flex;
    gap: 10px;
    align-items: center;
    justify-content: center;
    margin-top: 16px;
    color: var(--unirank-muted);
    font-size: 11px;
  }

  .unirank-heatmap-legend-scale {
    display: inline-grid;
    grid-template-columns: repeat(9, 18px);
    gap: 2px;
  }

  .unirank-heatmap-legend-scale i {
    width: 18px;
    height: 10px;
    border: 1px solid rgba(16, 43, 87, 0.08);
    box-sizing: border-box;
  }

  .unirank-heatmap-legend-missing {
    display: inline-flex;
    gap: 5px;
    align-items: center;
  }

  .unirank-heatmap-legend-missing::before {
    width: 18px;
    height: 10px;
    border: 1px solid var(--unirank-rule);
    background: #f1f4f7;
    content: "";
    box-sizing: border-box;
  }

  .unirank-heatmap-legend-source {
    display: inline-flex;
    gap: 5px;
    align-items: center;
  }

  .unirank-heatmap-legend-source::before {
    width: 18px;
    height: 10px;
    border: 2px solid rgba(16, 43, 87, 0.72);
    background: #ffffff;
    content: "";
    box-sizing: border-box;
  }

  .heatmap-tooltip .chart-tooltip-marker {
    display: none;
  }

  .data-table {
    border-collapse: collapse;
    font-size: 13px;
    font-variant-numeric: tabular-nums lining-nums;
  }

  .data-table th {
    padding-top: 14px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--unirank-rule-strong);
    background: #ffffff;
    color: var(--unirank-navy);
    font-family: var(--unirank-serif);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.015em;
  }

  .data-table td {
    height: 42px;
    border-bottom: 1px solid var(--unirank-rule);
    color: var(--unirank-ink);
  }

  .data-table tbody tr:hover td {
    background: var(--unirank-hover);
  }

  .table-cell-movement-positive {
    color: var(--unirank-green) !important;
    font-weight: 650;
  }

  .table-cell-movement-negative {
    color: var(--unirank-red) !important;
    font-weight: 650;
  }

  button,
  .viz-card-menu-button {
    border-radius: 4px;
  }

  button:focus-visible,
  [tabindex]:focus-visible {
    outline: 2px solid var(--unirank-green);
    outline-offset: 3px;
  }

  dialog {
    border: 1px solid var(--unirank-rule-strong);
    border-radius: 4px;
    box-shadow: 0 22px 60px rgba(16, 43, 87, 0.16);
  }

  @media (max-width: 760px) {
    :root,
    :root[data-theme="light"],
    :root[data-theme="dark"] {
      --ds-chart-body-height: 300px;
    }

    .dashboard-shell.report-shell > :not(.analytics-top-bar):not(.copy-toast):not(dialog):not(.unified-chart-detail-page) {
      width: min(100% - 32px, 1080px);
    }

    .analytics-top-bar {
      padding: 0 16px;
    }

    .analytics-top-bar-freshness {
      display: none;
    }

    .report-block-stack {
      padding-top: 34px;
      padding-bottom: 56px;
    }

    [data-layout-block-id="title"] .report-markdown-editor.markdown-render h1 {
      font-size: 34px;
      line-height: 1.17;
    }

    [data-layout-block-id="title"] .report-markdown-editor.markdown-render p {
      font-size: 16px;
    }

    .report-stack-item-metric-card {
      margin-bottom: 0;
      border-top: 1px solid var(--unirank-rule);
      border-bottom: 0;
      border-left: 0 !important;
    }

    .report-stack-item-metric-card:first-of-type {
      border-top: 2px solid var(--unirank-rule-strong);
    }

    .report-stack-item-metric-card:last-of-type {
      margin-bottom: 34px;
      border-bottom: 2px solid var(--unirank-rule-strong);
    }

    .report-metric-card {
      min-height: 92px;
      padding: 16px 4px;
    }

    .report-metric-card .kpi-value {
      font-size: 32px;
    }

    [data-layout-block-id="baseline_agreement"],
    [data-layout-block-id="expansion_takeaway"],
    [data-layout-block-id="strict_takeaway"],
    [data-layout-block-id="weight_boundary"] {
      padding-left: 18px;
    }

    .report-markdown-editor.markdown-render h2 {
      font-size: 21px;
    }

    .panel-header,
    .chart-panel .panel-header,
    .table-panel .panel-header {
      padding-top: 18px;
    }

    .data-table {
      font-size: 12px;
    }

    .heatmap-grid-panel {
      overflow-x: auto;
      padding-bottom: 18px;
    }

    .heatmap-grid {
      min-width: 720px;
    }

    .heatmap-cell {
      min-height: 38px;
    }

    .unirank-heatmap-legend {
      min-width: 720px;
      justify-content: flex-start;
    }

    .unirank-heatmap-controls {
      min-width: 698px;
    }

    .heatmap-axis-collapse {
      opacity: 0.72;
    }
  }

  @media print {
    .analytics-top-bar { position: static; }
    .viz-card-menu-button { display: none !important; }
    .report-block-stack { padding-top: 24px; }
  }
</style>`;

const heatmapEnhancement = String.raw`
<script id="unirank-diverging-heatmap-enhancement">
(() => {
  const chartId = "combined_auc_matrix";
  const positive = [8, 122, 69];
  const negative = [180, 58, 53];
  const hiddenRows = new Set();
  const hiddenColumns = new Set();

  function artifactPayload() {
    const raw = window.__DATA_ANALYTICS_PORTABLE_ARTIFACT__;
    return raw && typeof raw === "object" ? (raw.artifact_payload || raw) : null;
  }

  function mixWithWhite(root, strength) {
    const channel = (value) => Math.round(255 + (value - 255) * strength);
    return "rgb(" + channel(root[0]) + ", " + channel(root[1]) + ", " + channel(root[2]) + ")";
  }

  function addLegend(panel) {
    if (panel.querySelector(".unirank-heatmap-legend")) return;
    const strengths = [0.95, 0.72, 0.48, 0.26];
    const colors = [
      ...strengths.map((strength) => mixWithWhite(negative, strength)),
      "#f7f9fb",
      ...strengths.slice().reverse().map((strength) => mixWithWhite(positive, strength)),
    ];
    const legend = document.createElement("div");
    legend.className = "unirank-heatmap-legend";
    legend.setAttribute("aria-label", "颜色图例：深红表示较大下降，深绿表示较大提升，蓝色内框表示 DIGAI Lab 迁移记录");
    legend.innerHTML = [
      "<span>下降</span>",
      '<span class="unirank-heatmap-legend-scale" aria-hidden="true">' + colors.map((color) => '<i style="background:' + color + '"></i>').join("") + "</span>",
      "<span>提升</span>",
      '<span class="unirank-heatmap-legend-source">DIGAI Lab 迁移</span>',
    ].join("");
    panel.appendChild(legend);
  }

  function directChildren(grid, tagName) {
    return Array.from(grid.children).filter((element) => element.tagName === tagName);
  }

  function matrixParts(grid) {
    return {
      cells: Array.from(grid.children).filter((element) => element.classList.contains("heatmap-cell")),
      columnLabels: directChildren(grid, "B"),
      corner: Array.from(grid.children).find((element) => element.classList.contains("heatmap-axis-corner")),
      rowLabels: directChildren(grid, "STRONG"),
    };
  }

  function addAxisCollapseButton(label, kind, key, applyVisibility) {
    if (label.dataset.unirankCollapseReady === "true") return;
    label.dataset.unirankCollapseReady = "true";
    label.dataset.unirankAxisKind = kind;
    label.dataset.unirankAxisKey = key;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "heatmap-axis-collapse";
    button.textContent = "−";
    button.title = kind === "row" ? "折叠这一行" : "折叠这一列";
    button.setAttribute("aria-label", (kind === "row" ? "折叠数据集行：" : "折叠模型列：") + key);
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const target = kind === "row" ? hiddenRows : hiddenColumns;
      target.add(key);
      applyVisibility();
    });
    label.appendChild(button);
  }

  function ensureControls(panel, grid, rows, series, applyVisibility) {
    let controls = panel.querySelector(".unirank-heatmap-controls");
    if (!controls) {
      controls = document.createElement("div");
      controls.className = "unirank-heatmap-controls";
      controls.setAttribute("aria-label", "矩阵行列折叠控制");
      controls.innerHTML = [
        '<span class="unirank-collapse-help">点击行名或列名旁的 − 折叠</span>',
        '<span class="unirank-collapse-status" aria-live="polite"></span>',
        '<button class="unirank-collapse-reset" type="button">恢复全部</button>',
        '<div class="unirank-hidden-items" hidden></div>',
      ].join("");
      controls.querySelector(".unirank-collapse-reset").addEventListener("click", () => {
        hiddenRows.clear();
        hiddenColumns.clear();
        applyVisibility();
      });
      panel.insertBefore(controls, grid);
    }

    const status = controls.querySelector(".unirank-collapse-status");
    const reset = controls.querySelector(".unirank-collapse-reset");
    const tray = controls.querySelector(".unirank-hidden-items");
    const visibleRowCount = rows.length - hiddenRows.size;
    const visibleColumnCount = series.length - hiddenColumns.size;
    const statusText = "当前显示 " + visibleRowCount + "/" + rows.length + " 行、" + visibleColumnCount + "/" + series.length + " 列";
    if (status.textContent !== statusText) status.textContent = statusText;
    reset.hidden = hiddenRows.size === 0 && hiddenColumns.size === 0;

    const signature = Array.from(hiddenRows).sort().join("|") + "::" + Array.from(hiddenColumns).sort().join("|");
    if (tray.dataset.signature === signature) return;
    tray.dataset.signature = signature;
    tray.replaceChildren();
    tray.hidden = hiddenRows.size === 0 && hiddenColumns.size === 0;
    if (tray.hidden) return;

    const label = document.createElement("span");
    label.textContent = "已折叠：";
    tray.appendChild(label);
    rows.forEach((row) => {
      if (!hiddenRows.has(row.dataset)) return;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "unirank-restore-chip";
      button.textContent = "行 · " + row.dataset + "  ×";
      button.setAttribute("aria-label", "恢复数据集行：" + row.dataset);
      button.addEventListener("click", () => {
        hiddenRows.delete(row.dataset);
        applyVisibility();
      });
      tray.appendChild(button);
    });
    series.forEach((item) => {
      if (!hiddenColumns.has(item.field)) return;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "unirank-restore-chip";
      button.textContent = "列 · " + (item.label || item.field) + "  ×";
      button.setAttribute("aria-label", "恢复模型列：" + (item.label || item.field));
      button.addEventListener("click", () => {
        hiddenColumns.delete(item.field);
        applyVisibility();
      });
      tray.appendChild(button);
    });
  }

  function addCollapsing(grid, panel, rows, series) {
    const applyVisibility = () => {
      const parts = matrixParts(grid);
      const visibleRows = rows.filter((row) => !hiddenRows.has(row.dataset));
      const visibleColumns = series.filter((item) => !hiddenColumns.has(item.field));

      parts.rowLabels.forEach((label, index) => {
        const row = rows[index];
        label.hidden = !row || hiddenRows.has(row.dataset);
        const button = label.querySelector(".heatmap-axis-collapse");
        if (button) button.disabled = visibleRows.length <= 1;
      });
      parts.columnLabels.forEach((label, index) => {
        const item = series[index];
        label.hidden = !item || hiddenColumns.has(item.field);
        const button = label.querySelector(".heatmap-axis-collapse");
        if (button) button.disabled = visibleColumns.length <= 1;
      });
      parts.cells.forEach((cell, index) => {
        const row = rows[Math.floor(index / series.length)];
        const item = series[index % series.length];
        cell.hidden = !row || !item || hiddenRows.has(row.dataset) || hiddenColumns.has(item.field);
      });
      if (parts.corner) parts.corner.hidden = false;

      grid.style.gridTemplateColumns = "minmax(112px, max-content) repeat(" + visibleColumns.length + ", minmax(64px, 1fr))";
      grid.style.gridTemplateRows = "repeat(" + visibleRows.length + ", minmax(var(--heatmap-row-min-height), 1fr)) minmax(var(--heatmap-header-row-height), auto)";
      ensureControls(panel, grid, rows, series, applyVisibility);
    };

    const parts = matrixParts(grid);
    parts.rowLabels.forEach((label, index) => {
      const row = rows[index];
      if (row) addAxisCollapseButton(label, "row", row.dataset, applyVisibility);
    });
    parts.columnLabels.forEach((label, index) => {
      const item = series[index];
      if (item) addAxisCollapseButton(label, "column", item.field, applyVisibility);
    });
    applyVisibility();
  }

  function enhance() {
    const artifact = artifactPayload();
    const chart = artifact?.manifest?.charts?.find((candidate) => candidate.id === chartId);
    const rows = chart ? artifact?.snapshot?.datasets?.[chart.dataset] : null;
    const series = Array.isArray(chart?.series)
      ? chart.series
      : (chart?.encodings?.y?.fields || []).map((field) => ({ field, label: field }));
    if (!chart || !Array.isArray(rows) || !series.length) return;

    const numericValues = rows.flatMap((row) => series
      .map((series) => row?.[series.field])
      .filter((value) => typeof value === "number" && Number.isFinite(value)));
    const maxAbs = Math.max(...numericValues.map((value) => Math.abs(value)), 1);
    const cells = document.querySelectorAll(".heatmap-grid .heatmap-cell");
    const expectedCount = rows.length * series.length;
    if (cells.length !== expectedCount) return;

    cells.forEach((cell, index) => {
      const rowIndex = Math.floor(index / series.length);
      const seriesIndex = index % series.length;
      const row = rows[rowIndex];
      const seriesSpec = series[seriesIndex];
      const value = row?.[seriesSpec.field];
      const provenance = row?.["_source_" + seriesSpec.field] || "unknown";
      const missing = typeof value !== "number" || !Number.isFinite(value);
      cell.dataset.unirankMissing = missing ? "true" : "false";
      cell.dataset.unirankSource = provenance;

      if (missing) {
        cell.dataset.unirankLabel = "—";
        cell.style.removeProperty("--unirank-cell-text");
        cell.setAttribute("aria-label", (row?.dataset || "") + " " + (seriesSpec.label || seriesSpec.field) + ": 未覆盖");
        cell.tabIndex = -1;
        return;
      }

      const strength = 0.16 + 0.79 * Math.pow(Math.abs(value) / maxAbs, 0.72);
      const root = value >= 0 ? positive : negative;
      const sign = value >= 0 ? "+" : "−";
      const label = sign + Math.abs(value).toFixed(3);
      cell.dataset.unirankLabel = label;
      cell.style.background = mixWithWhite(root, strength);
      cell.style.setProperty("--unirank-cell-text", strength > 0.58 ? "#ffffff" : (value >= 0 ? "#075b36" : "#7f2925"));
      const sourceLabel = provenance === "digai" ? "DIGAI Lab 迁移记录" : "HPC3 expansion";
      cell.setAttribute("aria-label", row.dataset + " " + (seriesSpec.label || seriesSpec.field) + ": " + label + " ×10⁻³，" + sourceLabel);
    });

    const panel = cells[0]?.closest(".heatmap-grid-panel");
    const grid = cells[0]?.closest(".heatmap-grid");
    if (panel && grid) {
      addLegend(panel);
      addCollapsing(grid, panel, rows, series);
    }
  }

  const observer = new MutationObserver(enhance);
  observer.observe(document.documentElement, { childList: true, subtree: true });
  window.addEventListener("data-analytics-portable-reader-ready", enhance);
  document.addEventListener("DOMContentLoaded", enhance, { once: true });
})();
</script>`;

const fallbackTheme = String.raw`
<style id="unirank-fallback-theme">
  body { margin: 0; background: #fff; color: #253653; font-family: "Noto Sans CJK SC", sans-serif; }
  .portable-content-card { border-color: #d5dee9 !important; border-radius: 0 !important; box-shadow: none !important; }
  .portable-content-card h1, .portable-content-card h2, .portable-content-card h3 {
    color: #102b57; font-family: "Noto Serif CJK SC", Georgia, serif;
  }
  .portable-content-card table { font-variant-numeric: tabular-nums; }
  .portable-content-card th { color: #102b57; }
</style>`;

function injectBeforeHeadEnd(html, css) {
  const markerIndex = html.lastIndexOf("</head>");
  if (markerIndex < 0) throw new Error("Portable reader HTML has no </head> marker.");
  return `${html.slice(0, markerIndex)}${css}\n${html.slice(markerIndex)}`;
}

function injectBeforeBodyEnd(html, content) {
  const markerIndex = html.lastIndexOf("</body>");
  if (markerIndex < 0) throw new Error("Portable reader HTML has no </body> marker.");
  return `${html.slice(0, markerIndex)}${content}\n${html.slice(markerIndex)}`;
}

const packagedRuntime = readPackagedReaderRuntime().html;
const themedRuntime = injectBeforeBodyEnd(
  injectBeforeHeadEnd(packagedRuntime, reportTheme),
  heatmapEnhancement,
);

function themedBuild(input, options = {}) {
  const html = buildPortableArtifact(input, {
    ...options,
    runtimeHtml: themedRuntime,
  });
  return injectBeforeHeadEnd(html, fallbackTheme);
}

try {
  const result = await deliverPortableArtifact(
    {
      actionTimeoutMs: 5000,
      inputPath,
      outputPath,
      readyTimeoutMs: 10000,
      timeoutMs: 30000,
    },
    { build: themedBuild },
  );
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
} catch (error) {
  const result = error?.deliveryResult ?? {
    ok: false,
    error: error?.message ?? String(error),
  };
  if (error?.details) result.details = error.details;
  if (error?.cause?.message) result.cause = error.cause.message;
  process.stderr.write(`${JSON.stringify(result, null, 2)}\n`);
  process.exitCode = 1;
}
