# Local Test Mode

Local test mode validates the harness end-to-end **without GPUs** by using minimal resources.

## What It Does

- Uses `nproc_per_node=1` (single process)
- Sets `MAX_WALLCLOCK_SECONDS=60` (1 minute timeout)
- Validates all data dependencies before running
- Captures and displays real outputs
- Provides clear failure diagnostics

## Quick Start

### Command (No GPU Required)

```bash
cd /Users/mahinnaveen/Documents/GitHub/parameter-golf/harness

python3 runner.py \
  --script ../records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py \
  --reproduce \
  --local_test
```

### Expected Output (Success Case)

```
[LOCAL TEST MODE] Using: nproc=1, MAX_WALLCLOCK_SECONDS=60
Starting run run_1775586752_seed42 with seed 42
Run directory: /Users/mahinnaveen/Documents/GitHub/parameter-golf/harness/run_run_1775586752_seed42

Artifacts in run directory:
  train_gpt.py: 2,048 bytes
  final_model.int8.ptz: 1,024,000 bytes
Total artifact size: 1,026,048 bytes (1.0MB)

First detected final_*_roundtrip line:
  final_int8_zlib_roundtrip val_loss:2.0727 val_bpb:1.2244

✓ Run run_1775586752_seed42: BPB=1.2244, Size=1.0MB, Time=45s

============================================================
REPRODUCTION RESULTS
============================================================
✓ Status: SUCCESS
  Final BPB: 1.2244
  Artifact Size: 1,026,048 bytes (1.0MB)
  Runtime: 45 seconds
  Run Directory: /Users/mahinnaveen/Documents/GitHub/parameter-golf/harness/run_run_1775586752_seed42
============================================================
```

---

## Common Scenarios

### Success (Score is valid and <16MB)
```
✓ Status: SUCCESS
  Final BPB: 1.2244
  Artifact Size: 1,026,048 bytes (1.0MB)
```
**Action:** Harness is working correctly.

---

### Missing Data Path
```
ERROR: Data path does not exist: /Users/mahinnaveen/Documents/GitHub/parameter-golf/data/datasets/fineweb10B_sp1024
Create it or update DATA_PATH in the config
```
**Action:** Create the data directory or update the path in runner.py.

---

### Missing Tokenizer
```
ERROR: Tokenizer path does not exist: /Users/mahinnaveen/Documents/GitHub/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
Create it or update TOKENIZER_PATH in the config
```
**Action:** Create the tokenizer or update the path.

---

### Timeout (Script exceeded 60 seconds)
```
Run run_1775586752_seed42 timed out after 60 seconds

[DIAGNOSTICS] Last 50 lines of log:
  step:1000/20000 val_loss:2.0606 val_bpb:1.2172
  step:1050/20000 val_loss:2.0615 val_bpb:1.2178
  ...

✗ Status: FAILED
  Runtime: 60 seconds
```
**Action:** Local test mode is just for validation. Full runs may need longer timeouts.

---

### Script Crash (No BPB Found)
```
[DIAGNOSTICS] Last 50 lines of log:
  Traceback (most recent call last):
    File "/path/to/train_gpt.py", line 42, in <module>
      ...ImportError: No module named 'torch'

✗ Status: FAILED
  Final BPB: N/A
```
**Action:** Check stderr for the error. Install missing dependencies.

---

### Artifact Size Exceeded
```
✗ Status: FAILED
  FAIL: Artifact size 17000000 exceeds 16MB limit
```
**Action:** The quantization scheme is creating artifacts that are too large. Adjust the script.

---

## Diagnostics When Things Go Wrong

### Show last 50 lines of logs:
```bash
tail -50 /Users/mahinnaveen/Documents/GitHub/parameter-golf/harness/experiment_logs/run_*.log
```

### Show stderr from specific run:
```bash
grep "=== STDERR ===" -A 100 /Users/mahinnaveen/Documents/GitHub/parameter-golf/harness/experiment_logs/run_*.log
```

### Inspect run directory:
```bash
ls -lh /Users/mahinnaveen/Documents/GitHub/parameter-golf/harness/run_run_*
```

---

## Next Steps After Validation

### 1. Verify on Real GPU (8xH100)
```bash
python3 runner.py \
  --script ../records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py \
  --reproduce \
  --nproc 8 \
  --timeout 600
```

### 2. Run Full Baseline
```bash
python3 runner.py \
  --script train_gpt.py \
  --configs baseline_config.json \
  --seeds 42 43 44 \
  --nproc 8 \
  --timeout 600
```

### 3. Experiment Grid
Update configs and run multiple configurations.

---

## Cleanup

Remove run directories after validation:
```bash
python3 runner.py \
  --script ../records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py \
  --reproduce \
  --local_test \
  --cleanup
```

Or manually:
```bash
rm -rf /Users/mahinnaveen/Documents/GitHub/parameter-golf/harness/run_run_*
```
