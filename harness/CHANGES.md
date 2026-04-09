# Harness Enhancements: Local Test Mode & Diagnostics

## Changes Made

### 1. Local Test Mode (`--local_test` flag)
- Uses `nproc_per_node=1` (single process)
- Sets `MAX_WALLCLOCK_SECONDS=60` (1 minute timeout)
- Ideal for validating harness correctness without GPUs

### 2. Data Dependency Validation
- Checks if `DATA_PATH` exists before running
- Checks if `TOKENIZER_PATH` exists before running
- Prints clear error messages with paths if missing
- Exits gracefully without running script

### 3. Enhanced Debug Output
```
Artifacts in run directory:
  train_gpt.py: 2,048 bytes
  final_model.int8.ptz: 1,024,000 bytes
Total artifact size: 1,026,048 bytes (1.0MB)

First detected final_*_roundtrip line:
  final_int8_zlib_roundtrip val_loss:2.0727 val_bpb:1.2244
```

### 4. Improved Failure Diagnostics
When a run fails:
- Shows last 50 lines of logs
- Displays stderr clearly
- Points to run directory for investigation
- Summary output format even on failure

### 5. Better Result Summary
```
============================================================
REPRODUCTION RESULTS
============================================================
✓ Status: SUCCESS
  Final BPB: 1.2244
  Artifact Size: 1,026,048 bytes (1.0MB)
  Runtime: 45 seconds
  Run Directory: /path/to/run_directory
============================================================
```

---

## Code Changes Summary

### File: `runner.py`

**Added:**
- `--local_test` argument flag
- Data dependency checks (DATA_PATH, TOKENIZER_PATH)
- First `final_*_roundtrip` line detection and display
- Improved success/failure result display
- Last 50 lines of logs on failure
- stderr capture on exceptions

**Modified:**
- `run_experiment()`: Now handles `local_test` mode
- `main()`: Added validation logic and enhanced output
- Error handling: Better exception messages

**No breaking changes** - all existing functionality preserved.

---

## Commands

### Minimal Test (Syntax Check, No Run)

```bash
cd /Users/mahinnaveen/Documents/GitHub/parameter-golf/harness
python3 runner.py --help
```

### Dry Run (Verify Command Construction)

```bash
python3 runner.py \
  --script ../records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py \
  --reproduce \
  --local_test \
  --dry_run
```

**Expected Output:**
```
[LOCAL TEST MODE] Using: nproc=1, MAX_WALLCLOCK_SECONDS=60
Starting run run_XXXXXXX_seed42 with seed 42
Run directory: /path/to/run_XXXXXXX_seed42
DRY RUN - not executing
```

---

## What Gets Validated

| Aspect | Check | Action if Fails |
|--------|-------|-----------------|
| DATA_PATH | Exists? | Exit with clear error |
| TOKENIZER_PATH | Exists? | Exit with clear error |
| Script runs | Executes? | Show last 50 log lines |
| BPB parsed | Found in output? | Fail with N/A |
| Artifacts <16MB | Size valid? | Fail with size |
| Roundtrip line | Detected in logs? | Display it |

---

## Real Run (When Data Exists)

```bash
python3 runner.py \
  --script ../records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py \
  --reproduce \
  --local_test
```

This will:
1. Create isolated run directory: `run_run_XXXXXXX_seed42/`
2. Copy script to run directory
3. Execute with torchrun (1 process, 60s timeout)
4. Capture stdout/stderr to logs
5. Parse BPB score
6. Measure artifact size
7. Print detailed results

---

## Error Messages

### Missing Data Path
```
ERROR: Data path does not exist: /path/to/data
Create it or update DATA_PATH in the config
```

### Missing Tokenizer
```
ERROR: Tokenizer path does not exist: /path/to/tokenizer
Create it or update TOKENIZER_PATH in the config
```

### Timeout
```
Run run_1775586752_seed42 timed out after 60 seconds
[DIAGNOSTICS] Last 50 lines of log:
  ... output ...
✗ Status: FAILED
  Runtime: 60 seconds
```

### Script Error
```
[DIAGNOSTICS] Last 50 lines of log:
  Traceback (most recent call last):
    ...
✗ Status: FAILED
  Final BPB: N/A
```

---

## File Structure After Run

```
harness/
├── runner.py
├── LOCAL_TEST.md
├── experiment_logs/
│   └── run_1775586752_seed42.log      ← Stdout/stderr
└── run_run_1775586752_seed42/         ← Isolated run dir
    ├── train_gpt.py                   ← Script copy
    └── final_model.int8.ptz           ← Model artifacts
```

---

## Next Steps

1. **Verify data exists:**
   ```bash
   ls -la /Users/mahinnaveen/Documents/GitHub/parameter-golf/data/datasets/
   ls -la /Users/mahinnaveen/Documents/GitHub/parameter-golf/data/tokenizers/
   ```

2. **Run local test:**
   ```bash
   python3 runner.py --script ../records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py --reproduce --local_test
   ```

3. **Check results:**
   - If SUCCESS: Harness is working. Move to full baseline.
   - If FAILED with timeout: Data loading issue, see logs.
   - If FAILED with error: Check stderr, fix script/data.

---

## Minimal Changes Made

- Added 1 flag (`--local_test`)
- Added 2 validation checks (DATA_PATH, TOKENIZER_PATH)
- Enhanced output formatting (+50 lines)
- Better error messages (+30 lines)
- **Total: ~80 lines added, 0 lines removed**

No breaking changes to existing API or behavior.

---

## Files Modified

1. `/Users/mahinnaveen/Documents/GitHub/parameter-golf/harness/runner.py`
   - Added argument parsing for `--local_test`
   - Added data path validation
   - Enhanced result display
   - Better error diagnostics

2. Created `/Users/mahinnaveen/Documents/GitHub/parameter-golf/harness/LOCAL_TEST.md`
   - User-facing guide for local testing
   - Common error scenarios
   - Troubleshooting steps
