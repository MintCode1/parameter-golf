#!/usr/bin/env python3
"""Recommend next experiment step based on parsed results from analyze_results.py."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Configuration constants
ARTIFACT_SAFE_HEADROOM = 500_000  # bytes; if below this, avoid combinations
ARTIFACT_WARNING_HEADROOM = 100_000  # bytes; if below this, warn about tight artifact pressure
ARTIFACT_SIZE_LIMIT = 16_000_000  # bytes


@dataclass
class Result:
    run_id: str
    final_roundtrip_bpb: float | None
    artifact_size: int | None
    artifact_headroom: int | None
    baseline_gap: float | None
    improvement_pct: float | None
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Result:
        return cls(
            run_id=data["run_id"],
            final_roundtrip_bpb=data.get("final_roundtrip_bpb"),
            artifact_size=data.get("artifact_size"),
            artifact_headroom=data.get("artifact_headroom"),
            baseline_gap=data.get("baseline_gap"),
            improvement_pct=data.get("improvement_pct"),
            notes=data.get("notes", ""),
        )

    def status(self, baseline: Result | None) -> str:
        """Classify the result as WIN, NEUTRAL, LOSE, or INVALID_ARTIFACT."""
        if self.artifact_headroom is not None and self.artifact_headroom < 0:
            return "INVALID_ARTIFACT"

        if self.final_roundtrip_bpb is None:
            return "INVALID_SCORE"

        if baseline is None or baseline.final_roundtrip_bpb is None:
            return "UNKNOWN"

        if self.final_roundtrip_bpb < baseline.final_roundtrip_bpb:
            return "WIN"
        elif self.final_roundtrip_bpb > baseline.final_roundtrip_bpb:
            return "LOSE"
        else:
            return "NEUTRAL"


def load_results(path: Path) -> list[Result]:
    """Load results from JSON file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [Result.from_dict(d) for d in data]


def format_table(results: list[Result], baseline: Result | None) -> str:
    """Format results as a summary table."""
    lines = []
    lines.append(
        "{:<20} {:<12} {:<12} {:<15} {:<12} {:<15} {:<15}".format(
            "run_id", "status", "final_bpb", "artifact_mb", "headroom_mb", "baseline_gap", "improvement_%"
        )
    )
    lines.append("-" * 115)

    for result in results:
        status = result.status(baseline)
        bpb_str = f"{result.final_roundtrip_bpb:.4f}" if result.final_roundtrip_bpb else "N/A"
        artifact_mb = f"{result.artifact_size / 1e6:.1f}" if result.artifact_size else "N/A"
        headroom_mb = f"{result.artifact_headroom / 1e6:.1f}" if result.artifact_headroom else "N/A"
        gap_str = f"{result.baseline_gap:+.4f}" if result.baseline_gap else "N/A"
        imp_str = f"{result.improvement_pct:+.2f}%" if result.improvement_pct else "N/A"

        lines.append(
            "{:<20} {:<12} {:<12} {:<15} {:<12} {:<15} {:<15}".format(
                result.run_id, status, bpb_str, artifact_mb, headroom_mb, gap_str, imp_str
            )
        )
    return "\n".join(lines)


def recommend_next_step(results: list[Result], baseline_id: str = "baseline") -> dict[str, Any]:
    """Apply decision logic to recommend the next step."""
    baseline = next((r for r in results if r.run_id == baseline_id), None)

    if baseline is None:
        return {
            "status": "error",
            "message": f"Baseline run '{baseline_id}' not found.",
            "best_candidate": None,
            "next_step": None,
            "runs_to_avoid": [],
            "reasoning": "Cannot proceed without baseline.",
        }

    wins = [r for r in results if r.run_id != baseline_id and r.status(baseline) == "WIN"]
    loses = [r for r in results if r.run_id != baseline_id and r.status(baseline) == "LOSE"]
    invalid = [r for r in results if r.run_id != baseline_id and r.status(baseline) == "INVALID_ARTIFACT"]

    quant_only = next((r for r in results if r.run_id == "quant_only"), None)
    bigram_4096 = next((r for r in results if r.run_id == "bigram_4096"), None)
    bigram_8192 = next((r for r in results if r.run_id == "bigram_8192"), None)
    mlp3x_only = next((r for r in results if r.run_id == "mlp3x_only"), None)

    quant_wins = quant_only and quant_only.status(baseline) == "WIN"
    bigram_4k_wins = bigram_4096 and bigram_4096.status(baseline) == "WIN"
    bigram_8k_wins = bigram_8192 and bigram_8192.status(baseline) == "WIN"
    mlp3x_wins = mlp3x_only and mlp3x_only.status(baseline) == "WIN"

    if not wins and not loses and not invalid:
        return {
            "status": "no_experiments_run",
            "best_candidate": None,
            "next_step": "Run the planned experiments (baseline first, then quant_only, bigram_4096, etc.)",
            "runs_to_avoid": [],
            "reasoning": "No non-baseline experiments have been run yet.",
        }

    if len(wins) == 0 and len(invalid) > 0:
        return {
            "status": "all_artifacts_exceed_limit",
            "best_candidate": baseline.run_id,
            "next_step": "STOP. All experiments exceeded 16MB artifact limit. Debug model weight compression or reduce model size.",
            "runs_to_avoid": [r.run_id for r in invalid],
            "reasoning": f"Every run produced artifacts > 16MB ({len(invalid)} runs). "
            f"This is not a feature problem; the base model+quantization strategy is too large. "
            f"Inspect train_gpt.py for weight savings opportunities (reduce layers, dim, embedding size, etc).",
        }

    if len(wins) == 0:
        return {
            "status": "all_lose",
            "best_candidate": baseline.run_id,
            "next_step": "STOP. Do not combine features. Inspect logs for failure modes.",
            "runs_to_avoid": [r.run_id for r in loses] + [r.run_id for r in invalid],
            "reasoning": f"No single feature improved over baseline. "
            f"{len(loses)} runs made it worse, {len(invalid)} exceeded artifact size. "
            f"Before combining features, debug training stability and hyperparameters.",
        }

    if quant_wins and bigram_4k_wins:
        tight_headroom = []
        if quant_only.artifact_headroom and quant_only.artifact_headroom < ARTIFACT_WARNING_HEADROOM:
            tight_headroom.append(f"quant_only ({quant_only.artifact_headroom:,} bytes)")
        if bigram_4096.artifact_headroom and bigram_4096.artifact_headroom < ARTIFACT_WARNING_HEADROOM:
            tight_headroom.append(f"bigram_4096 ({bigram_4096.artifact_headroom:,} bytes)")

        warning = ""
        if tight_headroom:
            warning = f" ⚠️ WARNING: {', '.join(tight_headroom)} have tight artifact headroom; combination may exceed 16MB."

        return {
            "status": "both_win",
            "best_candidate": "quant_only + bigram_4096",
            "next_step": f"Test combination: USE_NEW_QUANT=1 USE_BIGRAM_FEATURE=1 BIGRAM_HASH_SIZE=4096{warning}",
            "runs_to_avoid": [r.run_id for r in loses] + [r.run_id for r in invalid],
            "reasoning": "Both quant_only and bigram_4096 individually improve BPB. "
            "Test the combination next. If that also wins, bigram_4096 is your go-to variant. "
            "Do NOT try bigram_8192 or mlp3x_only until you confirm the combined model also improves.",
        }

    if quant_wins:
        next_step = "Run bigram_4096 (using quant_only as new baseline reference)"
        return {
            "status": "quant_wins",
            "best_candidate": "quant_only",
            "next_step": next_step,
            "runs_to_avoid": [r.run_id for r in loses] + [r.run_id for r in invalid],
            "reasoning": "quant_only improves BPB. This is the cleanest win. "
            "Next, check if bigram_4096 also improves. If so, test the combination. "
            "Avoid mlp3x_only and larger bigram variants until you confirm both features can coexist safely.",
        }

    if bigram_4k_wins:
        if bigram_4096.artifact_headroom and bigram_4096.artifact_headroom > ARTIFACT_SAFE_HEADROOM:
            next_step = "Run bigram_8192 to test hash-size sensitivity"
        else:
            next_step = "Keep bigram_4096. Do NOT try bigram_8192 (artifact headroom too low)"

        return {
            "status": "bigram_4k_wins",
            "best_candidate": "bigram_4096",
            "next_step": next_step,
            "runs_to_avoid": [r.run_id for r in loses] + [r.run_id for r in invalid],
            "reasoning": "bigram_4096 improves BPB. "
            "If artifact headroom is safe (>500KB), try bigram_8192 next. "
            "Still investigate quant_only separately, then decide whether to combine them.",
        }

    if bigram_8k_wins:
        return {
            "status": "bigram_8k_wins",
            "best_candidate": "bigram_8192",
            "next_step": "Consider quant_only next, then test quant_only + bigram_8192 if safe",
            "runs_to_avoid": [r.run_id for r in loses] + [r.run_id for r in invalid],
            "reasoning": "bigram_8192 wins but this is a riskier variant. "
            "Confirm artifact remains comfortably under 16MB before pursuing combinations.",
        }

    if mlp3x_wins:
        if mlp3x_only.artifact_headroom and mlp3x_only.artifact_headroom > ARTIFACT_SAFE_HEADROOM:
            next_step = "Test mlp3x_only + quant_only (if both show promise separately)"
        else:
            next_step = "Keep mlp3x_only isolated. Do NOT combine with other features (artifact too tight)"

        return {
            "status": "mlp3x_wins",
            "best_candidate": "mlp3x_only",
            "next_step": next_step,
            "runs_to_avoid": [r.run_id for r in loses] + [r.run_id for r in invalid],
            "reasoning": "mlp3x_only wins but it is a high-risk variant due to artifact pressure. "
            "Only combine it with quant if you have confirmed artifact headroom. "
            "Avoid combining with bigram features until you fully understand the size/performance tradeoff.",
        }

    best = min([r for r in wins], key=lambda r: r.final_roundtrip_bpb if r.final_roundtrip_bpb else float("inf"))
    return {
        "status": "multiple_wins",
        "best_candidate": best.run_id,
        "next_step": f"Best so far is {best.run_id}. Check artifact headroom before combining with other features.",
        "runs_to_avoid": [r.run_id for r in loses] + [r.run_id for r in invalid],
        "reasoning": f"Multiple runs improved: {[r.run_id for r in wins]}. "
        f"Best candidate is {best.run_id} with BPB={best.final_roundtrip_bpb:.4f}.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend next experiment step based on parsed results.")
    parser.add_argument("--results", default="results.json", help="Path to results.json from analyze_results.py")
    parser.add_argument("--baseline", default="baseline", help="Baseline run ID")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run `python3 analyze_results.py` first to generate results.json")
        return

    results = load_results(results_path)

    print("=" * 115)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 115)
    print(format_table(results, next((r for r in results if r.run_id == args.baseline), None)))
    print()

    recommendation = recommend_next_step(results, args.baseline)

    print("=" * 115)
    print("RECOMMENDATION")
    print("=" * 115)
    print(f"Status: {recommendation['status']}")
    print(f"Best Current Candidate: {recommendation['best_candidate']}")
    print(f"Next Step: {recommendation['next_step']}")
    if recommendation["runs_to_avoid"]:
        print(f"Runs to Avoid: {', '.join(recommendation['runs_to_avoid'])}")
    print(f"Why: {recommendation['reasoning']}")
    print("=" * 115)


if __name__ == "__main__":
    main()
