#!/usr/bin/env python3
"""
Parameter Golf Experimentation Harness

Minimal, reproducible runner for training and evaluating parameter golf submissions.
"""

import os
import sys
import json
import time
import subprocess
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse


class ExperimentRunner:
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.logs_dir = workspace_root / "experiment_logs"
        self.logs_dir.mkdir(exist_ok=True)

    def run_experiment(
        self,
        script_path: Path,
        env_vars: Dict[str, str],
        timeout_seconds: Optional[int] = None,
        seed: int = 42,
        nproc_per_node: int = 8,
        dry_run: bool = False,
        cleanup: bool = False
    ) -> Dict[str, Any]:
        """
        Run a single experiment.

        Args:
            script_path: Path to the training script
            env_vars: Environment variables to set
            timeout_seconds: Maximum runtime in seconds
            seed: Random seed
            nproc_per_node: Number of processes per node for torchrun
            dry_run: If True, print command without executing
            cleanup: If True, delete run directory after completion

        Returns:
            Dict with results: success, wallclock_time, artifact_size, final_bpb, logs_path
        """
        run_id = f"run_{int(time.time())}_seed{seed}"
        run_dir = self.workspace_root / f"run_{run_id}"
        run_dir.mkdir(exist_ok=True)
        
        log_file = self.logs_dir / f"{run_id}.log"

        # Copy script to run directory
        script_name = script_path.name
        run_script_path = run_dir / script_name
        import shutil
        shutil.copy2(script_path, run_script_path)

        # Set environment variables
        env = os.environ.copy()
        env.update(env_vars)
        env["SEED"] = str(seed)
        env["RUN_ID"] = run_id

        # Prepare command - use torchrun for distributed execution
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={nproc_per_node}",
            str(run_script_path)
        ]

        print(f"Starting run {run_id} with seed {seed}")
        print(f"Run directory: {run_dir}")

        if dry_run:
            print("DRY RUN - not executing")
            return {
                "run_id": run_id,
                "success": True,
                "wallclock_time": 0.0,
                "artifact_size": 0,
                "final_bpb": None,
                "logs_path": str(log_file),
                "seed": seed,
                "config": env_vars.copy(),
                "dry_run": True
            }

        start_time = time.time()
        success = False
        stdout = ""
        stderr = ""

        try:
            result = subprocess.run(
                cmd,
                cwd=run_dir,  # Run in the isolated run directory
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
            success = result.returncode == 0
            stdout = result.stdout
            stderr = result.stderr

        except subprocess.TimeoutExpired:
            print(f"Run {run_id} timed out after {timeout_seconds} seconds")
            success = False
            stderr = f"Process timed out after {timeout_seconds} seconds"
        except Exception as e:
            print(f"Run {run_id} failed with exception: {e}")
            success = False
            stderr = str(e)

        wallclock_time = time.time() - start_time

        # Save logs
        with open(log_file, 'w') as f:
            f.write(f"=== STDOUT ===\n{stdout}\n")
            f.write(f"=== STDERR ===\n{stderr}\n")

        # Parse results
        final_bpb = self._parse_final_bpb(stdout + stderr)
        artifact_size, file_breakdown = self._get_artifact_size(run_dir)

        result = {
            "run_id": run_id,
            "success": success,
            "wallclock_time": wallclock_time,
            "artifact_size": artifact_size,
            "final_bpb": final_bpb,
            "logs_path": str(log_file),
            "seed": seed,
            "config": env_vars.copy(),
            "run_dir": str(run_dir)
        }

        # Parse and display first roundtrip line
        roundtrip_logs = stdout + stderr
        roundtrip_pattern = r"(final_.*?_roundtrip.*)"
        roundtrip_match = re.search(roundtrip_pattern, roundtrip_logs)
        
        # Debug output
        print(f"\nArtifacts in run directory:")
        for file_path, size in file_breakdown.items():
            print(f"  {file_path}: {size:,} bytes")
        print(f"Total artifact size: {artifact_size:,} bytes ({artifact_size/1_000_000:.1f}MB)")
        
        if roundtrip_match:
            print(f"\nFirst detected final_*_roundtrip line:")
            print(f"  {roundtrip_match.group(1)[:120]}...") if len(roundtrip_match.group(1)) > 120 else print(f"  {roundtrip_match.group(1)}")

        # Validate constraints
        if success and artifact_size > 16_000_000:
            print(f"FAIL: Artifact size {artifact_size} exceeds 16MB limit")
            result["success"] = False

        if final_bpb is None and success:
            print(f"FAIL: No valid final BPB score found")
            result["success"] = False
        
        # If failed, show stderr
        if not success and stderr:
            print(f"\nSTDERR from run:")
            stderr_lines = stderr.split("\n")[-50:]
            for line in stderr_lines:
                if line.strip():
                    print(f"  {line}")

        # Print summary
        status = "✓" if result["success"] else "✗"
        bpb_str = f"{result['final_bpb']:.4f}" if result["final_bpb"] else "N/A"
        size_mb = result["artifact_size"] / 1_000_000
        print(f"{status} Run {run_id}: BPB={bpb_str}, Size={size_mb:.1f}MB, Time={result['wallclock_time']:.0f}s")

        # Cleanup if requested
        if cleanup and run_dir.exists():
            import shutil
            shutil.rmtree(run_dir)
            print(f"Cleaned up run directory: {run_dir}")

        return result

    def _parse_final_bpb(self, logs: str) -> Optional[float]:
        """
        Extract the final roundtrip val_bpb from logs.

        Looks for any line containing: final_*_roundtrip ... val_bpb:X
        Prefers the non-exact (rounded) version if both exist.
        """
        # Look for the rounded version first (not _exact)
        # Match lines like: final_int8_zlib_roundtrip val_loss:2.0727 val_bpb:1.2244
        # The pattern needs to find "final_", then anything, then "_roundtrip" (but not "_roundtrip_exact")
        rounded_pattern = r"final_.*?_roundtrip(?!_exact)\s+.*?val_bpb:([\d.]+)"
        rounded_match = re.search(rounded_pattern, logs)
        if rounded_match:
            return float(rounded_match.group(1))

        # Fallback to any final roundtrip score (including exact)
        pattern = r"final_.*?_roundtrip\s+.*?val_bpb:([\d.]+)"
        matches = re.findall(pattern, logs)
        if matches:
            scores = [float(m) for m in matches]
            return min(scores)  # Lower BPB is better

        return None

    def _get_artifact_size(self, run_dir: Path) -> tuple[int, Dict[str, int]]:
        """Get the size of the submission artifact from files in the run directory."""
        model_patterns = ["*.pt", "*.ptz", "*.pth", "*.bin", "final_model*"]
        file_breakdown = {}
        total_size = 0

        # Scan for model files in run directory
        for pattern in model_patterns:
            for file_path in run_dir.glob(f"**/{pattern}"):
                if file_path.is_file():
                    rel_path = str(file_path.relative_to(run_dir))
                    if rel_path not in file_breakdown:  # Avoid double-counting
                        size = file_path.stat().st_size
                        file_breakdown[rel_path] = size
                        total_size += size

        # Add script file size
        script_file = run_dir / "train_gpt.py"
        if script_file.exists():
            rel_path = "train_gpt.py"
            if rel_path not in file_breakdown:  # Avoid double-counting
                size = script_file.stat().st_size
                file_breakdown[rel_path] = size
                total_size += size

        return total_size, file_breakdown


class ExperimentManager:
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.runner = ExperimentRunner(workspace_root)
        self.results: List[Dict[str, Any]] = []

    def run_grid(
        self,
        script_path: Path,
        configs: List[Dict[str, str]],
        seeds: List[int],
        timeout_seconds: Optional[int] = None,
        nproc_per_node: int = 8,
        dry_run: bool = False,
        cleanup: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Run experiments across a grid of configs and seeds.
        """
        for config in configs:
            for seed in seeds:
                result = self.runner.run_experiment(
                    script_path=script_path,
                    env_vars=config,
                    timeout_seconds=timeout_seconds,
                    seed=seed,
                    nproc_per_node=nproc_per_node,
                    dry_run=dry_run,
                    cleanup=cleanup
                )
                self.results.append(result)

        return self.results

    def save_results(self, output_path: Path):
        """Save results as JSONL."""
        with open(output_path, 'w') as f:
            for result in self.results:
                json.dump(result, f)
                f.write('\n')

    def generate_leaderboard(self) -> str:
        """Generate a markdown leaderboard ranked by median BPB."""
        # Group by config
        config_groups = {}
        for result in self.results:
            if not result["success"]:
                continue
            config_key = json.dumps(result["config"], sort_keys=True)
            if config_key not in config_groups:
                config_groups[config_key] = []
            config_groups[config_key].append(result["final_bpb"])

        # Calculate medians
        leaderboard = []
        for config_key, bpbs in config_groups.items():
            if len(bpbs) >= 1:  # At least one successful run
                median_bpb = sorted(bpbs)[len(bpbs)//2]
                config = json.loads(config_key)
                leaderboard.append({
                    "config": config,
                    "median_bpb": median_bpb,
                    "num_runs": len(bpbs),
                    "bpbs": bpbs
                })

        # Sort by median BPB (lower is better)
        leaderboard.sort(key=lambda x: x["median_bpb"])

        # Generate markdown
        lines = ["# Experiment Leaderboard", ""]
        lines.append("| Rank | Config | Median BPB | Runs | All BPB Scores |")
        lines.append("|------|--------|------------|-------|----------------|")

        for i, entry in enumerate(leaderboard, 1):
            config_str = ", ".join(f"{k}={v}" for k, v in entry["config"].items())
            bpbs_str = ", ".join(f"{b:.4f}" for b in sorted(entry["bpbs"]))
            lines.append(f"| {i} | {config_str} | {entry['median_bpb']:.4f} | {entry['num_runs']} | {bpbs_str} |")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Parameter Golf Experiment Harness")
    parser.add_argument("--script", type=Path,
                       help="Path to training script")
    parser.add_argument("--configs", type=str,
                       help="JSON file with list of config dicts")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                       help="Random seeds to use")
    parser.add_argument("--timeout", type=int,
                       help="Timeout per run in seconds")
    parser.add_argument("--nproc", type=int, default=8,
                       help="Number of processes per node for torchrun")
    parser.add_argument("--output", type=Path, default=Path("experiment_results.jsonl"),
                       help="Output file for results")
    parser.add_argument("--reproduce", action="store_true",
                       help="Reproduction mode: run one config and print results")
    parser.add_argument("--dry_run", action="store_true",
                       help="Dry run: print commands without executing")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up run directories after completion")
    parser.add_argument("--local_test", action="store_true",
                       help="Local test mode: use 1 process, 60s timeout for quick validation")

    args = parser.parse_args()

    workspace_root = Path.cwd()
    manager = ExperimentManager(workspace_root)

    # Reproduction mode
    if args.reproduce:
        if not args.script:
            parser.error("--reproduce requires --script")
        
        # Override settings for local test mode
        if args.local_test:
            nproc = 1
            timeout = 60
        else:
            nproc = args.nproc
            timeout = args.timeout
        
        # Default config for reproduction
        default_config = {
            "DATA_PATH": str(workspace_root / "data/datasets/fineweb10B_sp1024"),
            "TOKENIZER_PATH": str(workspace_root / "data/tokenizers/fineweb_1024_bpe.model"),
            "VOCAB_SIZE": "1024",
            "MAX_WALLCLOCK_SECONDS": str(timeout if args.local_test else 600),
            "TRAIN_LOG_EVERY": "50",
            "VAL_LOSS_EVERY": "200"
        }
        
        # Validate data dependencies
        data_path = Path(default_config["DATA_PATH"])
        tokenizer_path = Path(default_config["TOKENIZER_PATH"])
        
        if not data_path.exists():
            print(f"ERROR: Data path does not exist: {data_path}")
            print(f"Create it or update DATA_PATH in the config")
            return
        
        if not tokenizer_path.exists():
            print(f"ERROR: Tokenizer path does not exist: {tokenizer_path}")
            print(f"Create it or update TOKENIZER_PATH in the config")
            return
        
        if args.local_test:
            print("[LOCAL TEST MODE] Using: nproc=1, MAX_WALLCLOCK_SECONDS=60")
        
        result = manager.runner.run_experiment(
            script_path=args.script,
            env_vars=default_config,
            timeout_seconds=timeout,
            seed=args.seeds[0] if args.seeds else 42,
            nproc_per_node=nproc,
            dry_run=args.dry_run,
            cleanup=args.cleanup
        )
        
        if not args.dry_run:
            print("\n" + "="*60)
            print("REPRODUCTION RESULTS")
            print("="*60)
            if result["success"]:
                print(f"✓ Status: SUCCESS")
                print(f"  Final BPB: {result['final_bpb']:.4f}")
                print(f"  Artifact Size: {result['artifact_size']:,} bytes ({result['artifact_size']/1_000_000:.1f}MB)")
                print(f"  Runtime: {result['wallclock_time']:.0f} seconds")
                print(f"  Run Directory: {result['run_dir']}")
            else:
                print(f"✗ Status: FAILED")
                print(f"  Final BPB: {result['final_bpb'] if result['final_bpb'] else 'N/A'}")
                print(f"  Runtime: {result['wallclock_time']:.0f} seconds")
                print(f"  Run Directory: {result['run_dir']}")
                print(f"\n[DIAGNOSTICS] Last 50 lines of log:")
                log_file = Path(result['logs_path'])
                if log_file.exists():
                    with open(log_file) as f:
                        lines = f.readlines()[-50:]
                        for line in lines:
                            print(f"  {line.rstrip()}")
            print("="*60)
        
        return

    # Normal experiment mode
    if not args.script or not args.configs:
        parser.error("Normal mode requires --script and --configs")

    # Load configs
    with open(args.configs) as f:
        configs = json.load(f)

    # Run experiments
    results = manager.run_grid(
        script_path=args.script,
        configs=configs,
        seeds=args.seeds,
        timeout_seconds=args.timeout,
        nproc_per_node=args.nproc,
        dry_run=args.dry_run,
        cleanup=args.cleanup
    )

    if args.dry_run:
        return

    # Save results
    manager.save_results(args.output)

    # Generate and print leaderboard
    leaderboard = manager.generate_leaderboard()
    print("\n" + leaderboard)

    # Save leaderboard
    leaderboard_path = args.output.with_suffix(".md")
    with open(leaderboard_path, 'w') as f:
        f.write(leaderboard)


if __name__ == "__main__":
    main()