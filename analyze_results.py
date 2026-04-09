#!/usr/bin/env python3
"""Parse experiment logs and generate a structured result summary.

This script is designed for the planned runs in `logs/` and extracts:
- run_id
- val_loss
- val_bpb
- final_roundtrip_bpb
- artifact_size
- notes

It also writes a JSON and CSV summary file.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LOG_PATTERN = re.compile(r"final_.*?_roundtrip(?!_exact)\s+.*?val_loss:([0-9]+\.[0-9]+)\s+val_bpb:([0-9]+\.[0-9]+)")
FALLBACK_ROUNDTRIP_PATTERN = re.compile(r"final_.*?_roundtrip\s+.*?val_loss:([0-9]+\.[0-9]+)\s+val_bpb:([0-9]+\.[0-9]+)")
ARTIFACT_PATTERN = re.compile(r"Total submission size int8\+zlib:\s*([0-9]+)\s*bytes")
VAL_LOSS_PATTERN = re.compile(r"val_loss:([0-9]+\.[0-9]+)")


@dataclass
class RunResult:
    run_id: str
    val_loss: float | None = None
    val_bpb: float | None = None
    final_roundtrip_bpb: float | None = None
    artifact_size: int | None = None
    notes: str = ""
    raw_path: str = ""
    baseline_gap: float | None = None
    artifact_headroom: int | None = None
    improvement_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["artifact_size"] = self.artifact_size
        data["baseline_gap"] = self.baseline_gap
        data["artifact_headroom"] = self.artifact_headroom
        data["improvement_pct"] = self.improvement_pct
        return data


def parse_log_file(path: Path) -> RunResult:
    text = path.read_text(errors="ignore")
    run_id = path.stem

    result = RunResult(run_id=run_id, raw_path=str(path))

    match = LOG_PATTERN.search(text)
    if match:
        result.val_loss = float(match.group(1))
        result.final_roundtrip_bpb = float(match.group(2))
        result.val_bpb = result.final_roundtrip_bpb
    else:
        fallback = FALLBACK_ROUNDTRIP_PATTERN.search(text)
        if fallback:
            result.val_loss = float(fallback.group(1))
            result.final_roundtrip_bpb = float(fallback.group(2))
            result.val_bpb = result.final_roundtrip_bpb

    artifact_match = ARTIFACT_PATTERN.search(text)
    if artifact_match:
        result.artifact_size = int(artifact_match.group(1))

    if result.val_loss is None and result.final_roundtrip_bpb is None:
        result.notes += "No final roundtrip line found. "

    if result.artifact_size is None:
        result.notes += "No artifact size line found. "

    return result


def load_results(log_dir: Path) -> list[RunResult]:
    results: list[RunResult] = []
    for path in sorted(log_dir.glob("*.log")):
        results.append(parse_log_file(path))
    return results


def annotate_results(results: list[RunResult], baseline_run_id: str = "baseline") -> None:
    baseline = next((r for r in results if r.run_id == baseline_run_id), None)
    if baseline and baseline.final_roundtrip_bpb is not None:
        for result in results:
            if result.final_roundtrip_bpb is not None:
                result.baseline_gap = result.final_roundtrip_bpb - baseline.final_roundtrip_bpb
                result.improvement_pct = (
                    (baseline.final_roundtrip_bpb - result.final_roundtrip_bpb) / baseline.final_roundtrip_bpb * 100
                )
            if result.artifact_size is not None:
                result.artifact_headroom = 16_000_000 - result.artifact_size
    else:
        for result in results:
            if result.artifact_size is not None:
                result.artifact_headroom = 16_000_000 - result.artifact_size


def write_json(results: list[RunResult], output: Path) -> None:
    with output.open("w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)


def write_csv(results: list[RunResult], output: Path) -> None:
    fieldnames = [
        "run_id",
        "val_loss",
        "val_bpb",
        "final_roundtrip_bpb",
        "artifact_size",
        "artifact_headroom",
        "baseline_gap",
        "improvement_pct",
        "notes",
        "raw_path",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: getattr(r, k) for k in fieldnames})


def table_text(results: list[RunResult]) -> str:
    header = [
        "run_id",
        "final_bpb",
        "val_loss",
        "artifact_size",
        "artifact_headroom",
        "baseline_gap",
        "improvement_pct",
        "notes",
    ]
    lines = ["\t".join(header)]
    for r in results:
        row = [
            r.run_id,
            f"{r.final_roundtrip_bpb:.4f}" if r.final_roundtrip_bpb is not None else "N/A",
            f"{r.val_loss:.4f}" if r.val_loss is not None else "N/A",
            str(r.artifact_size) if r.artifact_size is not None else "N/A",
            str(r.artifact_headroom) if r.artifact_headroom is not None else "N/A",
            f"{r.baseline_gap:.4f}" if r.baseline_gap is not None else "N/A",
            f"{r.improvement_pct:.2f}%" if r.improvement_pct is not None else "N/A",
            r.notes,
        ]
        lines.append("\t".join(row))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse experiment logs and generate structured summaries.")
    parser.add_argument("--log-dir", default="logs", help="Directory containing run logs (*.log)")
    parser.add_argument("--output-json", default="results.json", help="JSON output path")
    parser.add_argument("--output-csv", default="results.csv", help="CSV output path")
    parser.add_argument("--baseline", default="baseline", help="Baseline run_id for comparisons")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists() or not log_dir.is_dir():
        raise SystemExit(f"Log directory not found: {log_dir}")

    results = load_results(log_dir)
    annotate_results(results, args.baseline)

    results_sorted = sorted(
        results,
        key=lambda r: (r.final_roundtrip_bpb is None, r.final_roundtrip_bpb),
    )

    write_json(results_sorted, Path(args.output_json))
    write_csv(results_sorted, Path(args.output_csv))

    print("Parsed results:")
    print(table_text(results_sorted))
    print()
    print(f"Wrote JSON: {args.output_json}")
    print(f"Wrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
