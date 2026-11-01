#!/usr/bin/env python3
"""Benchmark regression tracking for mllee_bench.

Builds ``mllee_bench`` in Release mode, runs it, parses the per-op
ticks/element (cycles per op on x86) and compares every hot-path op against
the stored baseline in ``tools/benchmarks/baseline.json``.

The script exits non-zero when any op regresses by more than 5% against the
baseline. With ``--update-baseline`` the baseline is rewritten from the
current run instead.

Baseline format::

    {
        "LINEAR 128x128 i8 raw": {"cycles": 2.43, "unit": "ticks/element"},
        ...
    }
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = REPO_ROOT / "tools" / "benchmarks"
BASELINE_PATH = BASELINE_DIR / "baseline.json"
BUILD_DIR = REPO_ROOT / "cmake-build-bench"
BENCH_BIN = BUILD_DIR / "mllee_bench"

REGRESSION_TOLERANCE = 0.05  # 5%

# mllee_bench prints a markdown table:
#   | LINEAR 128x128 i8 raw        |      16384 |     2.43 |            16816 |
_TABLE_RE = re.compile(
    r"^\s*\|\s*(?P<name>.+?)\s*\|\s*(?P<elements>\d+)\s*\|\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*\|\s*(?P<checksum>\d+)\s*\|\s*$"
)

# Fallback for a flat "name: value unit/op" line.
_FLAT_RE = re.compile(
    r"^(?P<name>\S.+?):\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>\w+)/op\s*$"
)

_SKIP_NAMES = {"op case"}


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                          text=True, capture_output=True)


def _nproc() -> int:
    try:
        return max(1, os.cpu_count() or 2)
    except Exception:  # pragma: no cover - defensive
        return 2


def build_bench() -> str:
    """Configure (Release) and build mllee_bench. Returns the binary path."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    generator = "Ninja" if shutil.which("ninja") else "Unix Makefiles"

    configure = [
        "cmake", "-S", str(REPO_ROOT), "-B", str(BUILD_DIR),
        "-G", generator,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DOPENZZS_BUILD_TESTS=OFF",
    ]
    proc = _run(configure)
    if proc.returncode != 0:
        sys.stderr.write("cmake configure failed:\n")
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(1)

    build = ["cmake", "--build", str(BUILD_DIR), "-j", str(_nproc()),
             "--target", "mllee_bench"]
    proc = _run(build)
    if proc.returncode != 0:
        sys.stderr.write("cmake build of mllee_bench failed:\n")
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(1)

    if not BENCH_BIN.exists():
        sys.stderr.write(f"mllee_bench binary not found at {BENCH_BIN}\n")
        raise SystemExit(1)

    return str(BENCH_BIN)


def run_bench(binary: str) -> str:
    proc = _run([binary], cwd=REPO_ROOT)
    if proc.returncode != 0:
        sys.stderr.write(f"mllee_bench exited {proc.returncode}:\n")
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(1)
    return proc.stdout


def parse_output(text: str) -> dict[str, dict]:
    """Parse bench output into {op_name: {"cycles": float, "unit": str}}."""
    result: dict[str, dict] = {}
    unit = "ticks/element"  # unit used by mllee_bench's ticks/element column

    for line in text.splitlines():
        match = _TABLE_RE.match(line)
        if match:
            name = match.group("name").strip()
            if name in _SKIP_NAMES:
                continue
            result[name] = {"cycles": float(match.group("value")), "unit": unit}
            continue

        match = _FLAT_RE.match(line)
        if match:
            name = match.group("name").strip()
            result[name] = {"cycles": float(match.group("value")),
                            "unit": match.group("unit")}

    return result


def load_baseline() -> dict[str, dict]:
    if not BASELINE_PATH.exists():
        return {}
    try:
        with open(BASELINE_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"warning: could not read baseline {BASELINE_PATH}: {exc}\n")
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_baseline(data: dict[str, dict]) -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = BASELINE_DIR / "baseline.json.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(BASELINE_PATH)


def compare(current: dict[str, dict], baseline: dict[str, dict],
            update: bool) -> int:
    failed = False

    for name in sorted(current):
        cur = current[name]["cycles"]
        cur_unit = current[name]["unit"]

        if name not in baseline:
            status = "PASS (new op, no baseline)"
            print(f"{status:<30} {name:<32} {cur:9.3f} {cur_unit}")
            continue

        base_entry = baseline[name]
        base = float(base_entry.get("cycles", 0.0))
        base_unit = base_entry.get("unit", cur_unit)

        if base <= 0.0:
            print(f"{'PASS (zero baseline)':<30} {name:<32} {cur:9.3f} {cur_unit}")
            continue

        change = (cur - base) / base
        if change > REGRESSION_TOLERANCE:
            failed = True
            status = "FAIL (regression)"
            print(f"{status:<30} {name:<32} {cur:9.3f} {cur_unit}"
                  f" (baseline {base:.3f} {base_unit}, {change:+.1%} > +5.0%)")
        else:
            direction = "faster" if change < 0 else "same"
            print(f"{'PASS':<30} {name:<32} {cur:9.3f} {cur_unit}"
                  f" (baseline {base:.3f} {base_unit}, {change:+.1%}, {direction})")

    if update:
        save_baseline(current)
        print(f"\nUpdated baseline with {len(current)} op(s) at {BASELINE_PATH}")

    if failed:
        print("\nBenchmark regression detected: hot-path op(s) regressed >5%.")
        return 1
    print("\nNo benchmark regressions.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Track mllee_bench hot-path regressions against a stored baseline.")
    parser.add_argument(
        "--update-baseline", action="store_true",
        help="Rewrite tools/benchmarks/baseline.json from the current run "
             "instead of comparing against it.")
    args = parser.parse_args(argv)

    binary = build_bench()
    text = run_bench(binary)
    current = parse_output(text)
    if not current:
        sys.stderr.write("No op measurements parsed from mllee_bench output.\n")
        return 1

    baseline = load_baseline()
    return compare(current, baseline, update=args.update_baseline)


if __name__ == "__main__":
    sys.exit(main())
