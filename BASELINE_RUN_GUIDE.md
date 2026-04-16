# Baseline Run: Complete Execution & Validation Guide

**Goal:** Run baseline successfully, validate metrics, decide on next step.  
**GPU:** 1×H100  
**Expected Duration:** ~10 min active training + 2 min validation = 12 min total  
**Cost:** ~$0.50 USD

---

## STEP 1: Pre-Flight Checks (Before You Start)

Run these 5 commands to confirm everything is ready:

```bash
# Check GPU visibility
nvidia-smi
```
**Expected:** Single H100 with 80GB free memory. Example output:
```
+-----+ GPU Memory +-----+
| 0   H100-80GB  | 80000 MiB free |
+-----+----------+-----+
```
✅ **Pass if:** GPU shows and free memory > 70GB

---

```bash
# Check directory structure
ls -la | grep -E "train_gpt|data|logs"
```
**Expected:** All present
```
-rw-r--r--  train_gpt.py
drwxr-xr-x  data/
drwxr-xr-x  logs/
```
✅ **Pass if:** train_gpt.py exists, data/ exists, logs/ exists

---

```bash
# Check data paths
ls -lh data/datasets/fineweb10B_sp1024/ | head -5
```
**Expected:** Multiple shard files (~500 MB each)
```
total 49380992
-rw-r--r-- shard_000.tar
-rw-r--r-- shard_001.tar
...
```
✅ **Pass if:** Shows 80+ shard files, ~50 GB total

---

```bash
# Check tokenizer
ls -lh data/tokenizers/
```
**Expected:**
```
-rw-r--r--  250K  fineweb_1024_bpe.model
```
✅ **Pass if:** File > 100KB

---

```bash
# Test Python imports
python3 -c "import torch; import sentencepiece; print('✓ All imports OK')"
```
**Expected:**
```
✓ All imports OK
```
✅ **Pass if:** No errors printed

---

## STEP 2: Set Environment Variables

Run this exactly as shown (copy-paste):

```bash
cd /workspace/parameter-golf

# Core paths (one-time setup)
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export MAX_WALLCLOCK_SECONDS=600
export CUDA_VISIBLE_DEVICES=0

# Verify all set
echo "✓ DATA_PATH=$DATA_PATH"
echo "✓ TOKENIZER_PATH=$TOKENIZER_PATH"
echo "✓ MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS"
echo "✓ CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
```

**Expected output:**
```
✓ DATA_PATH=./data/datasets/fineweb10B_sp1024
✓ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
✓ MAX_WALLCLOCK_SECONDS=600
✓ CUDA_VISIBLE_DEVICES=0
```

✅ **Pass if:** All 4 lines print with correct values

---

## STEP 3: Execute Baseline Command

**COPY THIS EXACTLY** (all feature flags OFF):

```bash
export RUN_ID=baseline
export USE_BIGRAM_FEATURE=0
export USE_MLP_3X=0
export USE_NEW_QUANT=0

echo "========== BASELINE RUN START =========="
echo "Time: $(date)"
echo "RUN_ID: $RUN_ID"
echo "Features: bigram=$USE_BIGRAM_FEATURE, mlp3x=$USE_MLP_3X, quant=$USE_NEW_QUANT"
echo "======================================="

stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log

echo "========== BASELINE RUN END =========="
echo "Time: $(date)"
```

---

## STEP 4: Monitor Real-Time Output

You will see output in this sequence. **Watch for these signs:**

### Sign 1: Initialization (First 30 seconds)
You should see:
```
Rank 0 initialized on worker-0 (rank=0 out of 1)
Loading training data...
Tokenizer initialized
Model initialized
```

✅ **Good:** Initialization completed without error  
❌ **Bad:** 
- "out of memory" → GPU has no memory; restart RunPod
- "No such file" → Data or tokenizer path wrong; check Step 2
- Anything else → Copy error message to FAILURE DIAGNOSIS section below

---

### Sign 2: Training Loop (Minutes 1–9)

Every few seconds you'll see:
```
[Iteration 5] loss=1.2540 val_loss=1.2541 val_bpb=1.2451
[Iteration 10] loss=1.2100 val_loss=1.2050 val_bpb=1.2100
[Iteration 15] loss=1.1950 val_loss=1.1890 val_bpb=1.1950
...
```

✅ **Good signs:**
- Loss **decreases gradually** (1.25 → 1.20 → 1.19...)
- val_bpb **gradually improves** (1.25 → 1.20...)
- GPU memory stable (~60–70 GB used)
- New iteration every 30–60 seconds

❌ **Bad signs:**
- Loss is NaN or Inf: `[Iteration 5] loss=nan`
- Loss **increases** dramatically or oscillates wildly
- No output **for >2 minutes** (GPU stuck)
- Memory keeps increasing (memory leak)

**If bad sign appears:** Stop training immediately (see FAILURE DIAGNOSIS)

---

### Sign 3: Training Completion (Minute 10)

Near the end you'll see:
```
[Iteration 100] loss=1.0812 val_loss=1.0745 val_bpb=1.0823
[Iteration 101] loss=1.0798 val_loss=1.0731 val_bpb=1.0809
Wrapping up training...
Validating final model...
Computing compression metrics...
```

Then **the critical lines:**
```
final_int8_zlib_roundtrip val_loss:1.0745 val_bpb:1.0823
Total submission size int8+zlib: 15912045 bytes
Computing final metrics...
Serializing final model...
Final metrics logged, finishing
Exiting cleanly
```

✅ **Success:** Both of these lines appeared:
```
final_int8_zlib_roundtrip val_loss:X.XXXX val_bpb:Y.YYYY
Total submission size int8+zlib: Z,ZZZ,ZZZ bytes
```

❌ **Failure:** Run crashed, log shows error, or one/both metrics lines missing

---

## STEP 5: Immediate Post-Run Validation

After the command returns (10 min), run this:

```bash
echo "=== BASELINE RUN VALIDATION ==="

# Check 1: Log file exists and is large
echo -e "\n[CHECK 1] Log file size:"
ls -lh logs/baseline.log

# Check 2: Extract final metrics
echo -e "\n[CHECK 2] Final metrics:"
grep "final_int8_zlib_roundtrip\|Total submission" logs/baseline.log | tail -2

# Check 3: Check for errors
echo -e "\n[CHECK 3] Error check (should be empty):"
grep -i "error\|exception\|traceback" logs/baseline.log | head -5

# Check 4: Check for NaN
echo -e "\n[CHECK 4] NaN check (should be empty):"
grep "nan\|inf" logs/baseline.log | head -5

# Check 5: Training iterations completed
echo -e "\n[CHECK 5] Training progress:"
grep "Iteration" logs/baseline.log | tail -5
```

**Expected output:**

```
[CHECK 1] Log file size:
-rw-r--r--  2.5M  logs/baseline.log

[CHECK 2] Final metrics:
final_int8_zlib_roundtrip val_loss:1.0745 val_bpb:1.0823
Total submission size int8+zlib: 15912045 bytes

[CHECK 3] Error check (should be empty):
(empty output is good)

[CHECK 4] NaN check (should be empty):
(empty output is good)

[CHECK 5] Training progress:
[Iteration 98] loss=1.0821 val_loss=1.0751 val_bpb=1.0831
[Iteration 99] loss=1.0816 val_loss=1.0748 val_bpb=1.0826
[Iteration 100] loss=1.0812 val_loss=1.0745 val_bpb=1.0823
[Iteration 101] loss=1.0798 val_loss=1.0731 val_bpb=1.0809
Wrapping up training...
```

---

## STEP 6: Extract & Verify Metrics

Run this to extract exact values:

```bash
# Extract BPB value
BPB=$(grep "final_int8_zlib_roundtrip" logs/baseline.log | grep -oP 'val_bpb:\K[0-9.]+' | tail -1)
echo "BPB: $BPB"

# Extract artifact size
ARTIFACT=$(grep "Total submission" logs/baseline.log | grep -oP 'size:[^:]*: \K[0-9,]+' | tail -1 | tr -d ',')
echo "Artifact size: $ARTIFACT bytes ($(echo "scale=1; $ARTIFACT / 1000000" | bc) MB)"

# Calculate headroom
HEADROOM=$((16000000 - ARTIFACT))
echo "Artifact headroom: $HEADROOM bytes ($(echo "scale=1; $HEADROOM / 1000000" | bc) MB)"
```

**Expected output:**
```
BPB: 1.0823
Artifact size: 15912045 bytes (15.9 MB)
Artifact headroom: 87955 bytes (0.1 MB)
```

---

## STEP 7: Validate Metrics Are In Range

**Criterion 1: BPB (bits-per-byte)**

```bash
grep "final_int8_zlib_roundtrip" logs/baseline.log | tail -1
```

**Valid range:** 1.00–1.30 (for baseline)

✅ **Pass if:** BPB between 1.00 and 1.30  
❌ **Fail if:** 
- BPB < 0.95 (unrealistic; something wrong)
- BPB > 1.50 (model not converging)

---

**Criterion 2: Artifact Size**

```bash
grep "Total submission" logs/baseline.log | tail -1
```

**Valid range:** 14,000,000–16,000,000 bytes (14–16 MB)

✅ **Pass if:** Between 14 MB and 16 MB  
❌ **Fail if:** 
- > 16,000,000 bytes (EXCEEDS LIMIT; invalid submission)
- < 5,000,000 bytes (artifact packing failed)

---

**Criterion 3: Artifact Headroom** (not critical for baseline, but useful)

```bash
ARTIFACT=$(grep "Total submission" logs/baseline.log | grep -oP 'size:[^:]*: \K[0-9,]+' | tail -1 | tr -d ',')
HEADROOM=$((16000000 - ARTIFACT))
echo "Headroom: $HEADROOM bytes"
```

**Interpretation:**
- > 500KB → Plenty of headroom for combinations
- 100KB–500KB → Safe, but watch for combinations
- < 100KB → Tight; be careful with feature combinations
- Negative → **INVALID** (run failed; do not use)

---

## STEP 8: Automated Validation

Run this Python check:

```bash
python3 -c "
import json

# Parse metrics manually
with open('logs/baseline.log', 'r') as f:
    log_text = f.read()

# Extract BPB
import re
bpb_match = re.search(r'final_int8_zlib_roundtrip.*val_bpb:([0-9.]+)', log_text)
bpb = float(bpb_match.group(1)) if bpb_match else None

# Extract artifact size
artifact_match = re.search(r'Total submission size.*: ([0-9,]+) bytes', log_text)
artifact_size = int(artifact_match.group(1).replace(',', '')) if artifact_match else None

print('=== BASELINE VALIDATION ===')
print(f'BPB: {bpb}')
print(f'Artifact size: {artifact_size} bytes ({artifact_size/1e6:.1f} MB)')
print(f'Headroom: {16000000 - artifact_size} bytes')

# Validation checks
valid = True
if bpb is None:
    print('❌ BPB not found')
    valid = False
elif not (1.0 <= bpb <= 1.3):
    print(f'❌ BPB out of range: {bpb}')
    valid = False
else:
    print(f'✅ BPB in range: {bpb}')

if artifact_size is None:
    print('❌ Artifact size not found')
    valid = False
elif artifact_size > 16000000:
    print(f'❌ Artifact exceeds 16MB: {artifact_size/1e6:.1f} MB')
    valid = False
elif artifact_size < 5000000:
    print(f'❌ Artifact too small (packing failed?): {artifact_size/1e6:.1f} MB')
    valid = False
else:
    print(f'✅ Artifact size valid: {artifact_size/1e6:.1f} MB')

if valid:
    print('\n✅ BASELINE RUN VALID - PROCEED TO NEXT STEP')
else:
    print('\n❌ BASELINE RUN INVALID - DEBUG REQUIRED')
"
```

**Expected output (success):**
```
=== BASELINE VALIDATION ===
BPB: 1.0823
Artifact size: 15912045 bytes (15.9 MB)
Headroom: 87955 bytes
✅ BPB in range: 1.0823
✅ Artifact size valid: 15.9 MB

✅ BASELINE RUN VALID - PROCEED TO NEXT STEP
```

---

## STEP 9: Parse Baseline Results

Now that baseline is valid, extract structured metrics:

```bash
python3 analyze_results.py \
  --log-dir logs \
  --output-json results.json \
  --output-csv results.csv \
  --baseline baseline
```

**Expected output:**
```
Parsed 1 run(s) from logs/
Wrote results.json
Wrote results.csv
```

**Verify results file:**
```bash
cat results.json | python3 -m json.tool
```

**Expected output:**
```json
[
  {
    "run_id": "baseline",
    "final_roundtrip_bpb": 1.0823,
    "artifact_size": 15912045,
    "artifact_headroom": 87955,
    "baseline_gap": 0.0,
    "improvement_pct": 0.0,
    "notes": ""
  }
]
```

✅ **Valid if:** All fields populated (no nulls), artifact_headroom ≥ 0

---

## What Could Go Wrong: Failure Cases

### Failure 1: CUDA Out of Memory (OOM)

**Symptom in logs:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB...
```

**Root cause:** Batch size too large OR GPU memory pressure from other processes

**Quick fix:**
```bash
# Check GPU memory
nvidia-smi

# If low (<10GB free), restart RunPod
# If >10GB free, reduce batch size in train_gpt.py:
#   Find: BATCH_SIZE = 32
#   Change to: BATCH_SIZE = 16
# Then restart run
```

---

### Failure 2: Loss → NaN (Divergence)

**Symptom in logs:**
```
[Iteration 15] loss=nan val_loss=nan
```

**Root cause:** Learning rate too high, weight initialization unstable, or corrupted data

**Quick fix:**
```bash
# Check if it's a one-time glitch (restart and see)
# If repeats, lower learning rate in train_gpt.py:
#   Find: lr = 3e-4
#   Change to: lr = 1e-4
# Restart baseline
```

---

### Failure 3: Training Stuck / No Progress

**Symptom:** No output for 2+ minutes, but no error message

**Root cause:** Data loading hang, GPU hang, or network timeout

**Quick fix:**
```bash
# Kill current run (Ctrl+C)
# Check disk space
df -h

# Check if data is readable
ls -lh data/datasets/fineweb10B_sp1024/ | head

# Restart training
```

---

### Failure 4: Wrong Feature Flags

**Symptom in logs:** Config shows unexpected values
```
USE_BIGRAM_FEATURE=1 (should be 0 for baseline)
```

**Root cause:** Env var not exported, or shell session lost

**Quick fix:**
```bash
# Re-export all env vars
export RUN_ID=baseline
export USE_BIGRAM_FEATURE=0
export USE_MLP_3X=0
export USE_NEW_QUANT=0

# Verify they're set
echo $USE_BIGRAM_FEATURE $USE_MLP_3X $USE_NEW_QUANT

# Restart training
```

---

### Failure 5: Artifact Size Exceeds 16MB

**Symptom in logs:**
```
Total submission size int8+zlib: 16,245,000 bytes
```

**Root cause:** Model + compression overhead too large; unlikely for baseline

**Quick fix:**
- This is NOT expected for baseline (should be ~15.9 MB)
- If it happens, check if features accidentally enabled
- Otherwise, model size may need reduction; rare for baseline

---

## Decision Criteria: Proceed or Debug?

After baseline completes, check this table:

| Outcome | Action |
|---------|--------|
| ✅ All validation checks pass | **PROCEED to quant_only** (Section: PROCEED DECISION) |
| ❌ BPB out of range (>1.3 or <1.0) | Debug and restart baseline |
| ❌ Artifact >16MB or <5MB | Debug and restart baseline |
| ❌ Training iterations <50 | Training cut short; restart |
| ❌ Error in logs (NaN, OOM, etc.) | Follow failure case above; restart |
| ⚠️ Artifact headroom <0 (negative) | **INVALID RUN** - restart from scratch |

---

## PROCEED DECISION: When to Run Quant-Only

**Check this before proceeding:**

```bash
# 1. Baseline log has both required lines
grep "final_int8_zlib_roundtrip" logs/baseline.log && \
grep "Total submission" logs/baseline.log && \
echo "✅ Both required lines present"

# 2. Results.json has baseline entry
grep "baseline" results.json && echo "✅ Baseline in results.json"

# 3. BPB in valid range (automate check)
python3 -c "
import json
with open('results.json') as f:
    data = json.load(f)
    bpb = data[0]['final_roundtrip_bpb']
    if 1.0 <= bpb <= 1.3:
        print('✅ BPB valid:', bpb)
    else:
        print('❌ BPB invalid:', bpb)
"
```

**Expected output:**
```
✅ Both required lines present
✅ Baseline in results.json
✅ BPB valid: 1.0823
```

**If all 3 checks pass:** ✅ **SAFE TO PROCEED TO QUANT_ONLY**

---

## Summary Checklist

- [ ] Pre-flight checks pass (GPU, data, tokenizer, imports)
- [ ] Environment variables set correctly
- [ ] Baseline command executed successfully
- [ ] Training iterations completed (>50)
- [ ] Both final metric lines appear in logs
- [ ] BPB in range 1.0–1.3
- [ ] Artifact size 14–16 MB
- [ ] No errors, NaNs, or OOMs in logs
- [ ] analyze_results.py runs without error
- [ ] results.json valid with baseline entry
- [ ] All validation checks pass

**If all boxes checked:** Ready for next step. Open **H100_EXECUTION_GUIDE.md SECTION 5 STEP 5** to run quant_only.

---

## Quick Reference: What to Save

After baseline succeeds, note these values:
```
Baseline BPB: _____________
Baseline Artifact Size (MB): _____________
Artifact Headroom (MB): _____________
Start Time: _____________
End Time: _____________
Total Duration: _____________
```

These become your anchor for all future comparisons.

---

**Created:** April 14, 2026  
**Purpose:** Execute baseline correctly and validate before proceeding  
**Next:** [H100_EXECUTION_GUIDE.md SECTION 5 STEP 5](H100_EXECUTION_GUIDE.md) → quant_only run
