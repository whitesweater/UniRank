#!/usr/bin/env node

import { existsSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const reportDirectory = dirname(fileURLToPath(import.meta.url));
const inputPath = resolve(reportDirectory, "artifact.json");
const outputPath = resolve(reportDirectory, "report.html");

function findReportBuilderDirectory() {
  const cacheRoot = resolve(
    homedir(),
    ".codex/plugins/cache/openai-curated-remote/data-analytics",
  );
  const candidates = readdirSync(cacheRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => resolve(cacheRoot, entry.name, "skills/build-report/scripts"))
    .filter((candidate) =>
      existsSync(resolve(candidate, "deliver_portable_artifact.mjs")),
    )
    .sort()
    .reverse();
  if (!candidates.length) {
    throw new Error("No installed Data Analytics portable report builder was found.");
  }
  return candidates[0];
}

const builderDirectory = findReportBuilderDirectory();
const buildModule = await import(
  pathToFileURL(resolve(builderDirectory, "build_portable_artifact.mjs"))
);
const deliveryModule = await import(
  pathToFileURL(resolve(builderDirectory, "deliver_portable_artifact.mjs"))
);

const { buildPortableArtifact, readPackagedReaderRuntime } = buildModule;
const { deliverPortableArtifact } = deliveryModule;

const overflowFix = String.raw`
<style id="unirank-portable-overflow-fix">
  .analytics-top-bar {
    width: 100%;
    margin-right: 0;
    margin-left: 0;
  }
</style>`;

function injectBeforeHeadEnd(html, content) {
  const markerIndex = html.lastIndexOf("</head>");
  if (markerIndex < 0) throw new Error("Portable reader HTML has no </head> marker.");
  return `${html.slice(0, markerIndex)}${content}\n${html.slice(markerIndex)}`;
}

const packagedRuntime = readPackagedReaderRuntime().html;
const fixedRuntime = injectBeforeHeadEnd(packagedRuntime, overflowFix);

function fixedBuild(input, options = {}) {
  return buildPortableArtifact(input, {
    ...options,
    runtimeHtml: fixedRuntime,
  });
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
    { build: fixedBuild },
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
