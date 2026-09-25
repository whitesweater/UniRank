#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const directory = dirname(fileURLToPath(import.meta.url));
const repository = resolve(directory, "../../..");
execFileSync(resolve(repository, ".venv/bin/python"),
  [resolve(directory, "build_selector.py")],
  { cwd: repository, stdio: "inherit" });
