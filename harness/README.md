# Parameter Golf Experimentation Harness

Minimal, reproducible experimentation harness for the OpenAI Parameter Golf challenge.

## Features

- **Runner**: Launches training scripts with configurable environment variables
- **Logging**: Captures full stdout/stderr to timestamped log files
- **Score Parsing**: Robustly extracts final roundtrip val_bpb from logs
- **Metrics Tracking**: Records per-run metrics (time, size, score, seed, config)
- **Hard Constraints**: Enforces 16MB artifact limit and score validation
- **Experiment Manager**: Runs grids of configs/seeds, ranks by median BPB
- **Correctness Tests**: Validates parsing and failure cases

## Quick Start

1. **Setup data** (see main repo README):
   ```bash
   python3 data/cached_challenge_fineweb.py --variant sp1024
   ```

2. **Run a single experiment**:
   ```bash
   cd harness
   python3 runner.py --script ../train_gpt.py --configs example_configs.json --seeds 42
   ```

3. **Run multiple configs/seeds**:
   ```bash
   python3 runner.py --script ../train_gpt.py --configs my_configs.json --seeds 42 1337 2024 --timeout 1800
   ```

## Output

- **Logs**: Saved to `experiment_logs/` with timestamps
- **Results**: JSONL file with per-run metrics
- **Leaderboard**: Markdown ranking by median BPB

## Testing

Run correctness tests:
```bash
python3 -m pytest test_harness.py -v
```

## Configuration

Configs are JSON arrays of environment variable dicts:

```json
[
  {
    "VOCAB_SIZE": "1024",
    "NUM_LAYERS": "9",
    "MAX_WALLCLOCK_SECONDS": "600"
  }
]
```

## Constraints Enforced

- Artifact size ≤ 16,000,000 bytes
- Valid final BPB score must be found
- Run must complete without crash
- Never silently continues on failures

## Score Parsing

Looks for lines like:
```
final_int8_zlib_roundtrip val_loss:2.0727 val_bpb:1.2244
```

Supports int8, int6, mixed precision formats.