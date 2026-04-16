# 1×H100 Execution Guide: First Iteration Loop

**Target:** Execute 3 runs (baseline, quant_only, bigram_4096) safely, cheaply, and correctly.  
**GPU:** 1× H100 80GB (not 8× setup)  
**Total Expected Time:** ~90 min wall-clock  
**Total Expected Cost:** ~$3.50–4.00 USD (at typical RunPod rates ~$2.50/hr)

---

## SECTION 1: Setup Differences (1×H100 vs 8×H100)

### Key Differences

| Aspect | 8×H100 Setup | 1×H100 Setup | Action |
|--------|-------------|------------|--------|
| `torchrun` command | `--nproc_per_node=8` | `--nproc_per_node=1` | **Use 1** |
| DDP (distributed) | Required | Not needed | **Remove DDP flags** |
| Memory per GPU | 80GB / 8 = 10GB per process | Full 80GB | **No change** |
| CUDA device mapping | `CUDA_VISIBLE_DEVICES=0-7` | `CUDA_VISIBLE_DEVICES=0` | **Use 0 only** |
| Batch scaling | Smaller batches × 8 GPUs | Larger single batch | **No code change; auto-scales** |
| Wall-clock time | ~10 min per run | ~10 min per run | **Same** |

### Required Changes

**Only two things differ from multi-GPU setup:**

1. **torchrun command** (already in DAY1_GPU_EXECUTION_CHECKLIST.md)
   ```bash
   # 1×H100 (use this)
   torchrun --standalone --nproc_per_node=1 train_gpt.py
   
   # NOT this (8×H100 format)
   torchrun --nproc_per_node=8 train_gpt.py
   ```

2. **Environment variable** (optional but safe)
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   ```

**No other changes needed.** The training script auto-detects single-GPU mode and adjusts batch sizes, learning rates, and accumulation steps correctly.

---

## SECTION 2: Exact Commands for Each Run

### Environment Setup (One-Time)
```bash
cd /workspace/parameter-golf

# Set paths (persistent for all runs)
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export MAX_WALLCLOCK_SECONDS=600
export CUDA_VISIBLE_DEVICES=0

# Create logs directory
mkdir -p logs

# Verify GPU is visible
nvidia-smi
```

**Expected from nvidia-smi:**
```
+-----+ GPU Memory +-----+
| 0   H100-80GB  | 80000 MiB free |
+-----+----------+-----+
```

---

### Run 1: BASELINE
**Duration:** ~10 min  
**Purpose:** Establish anchor metrics  
**Features:** All OFF

```bash
export RUN_ID=baseline
export USE_BIGRAM_FEATURE=0
export USE_MLP_3X=0
export USE_NEW_QUANT=0

stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log
```

**Run starts when you see:**
```
Rank 0 initialized on worker-0 (rank=0 out of 1)
Loading training data...
```

**Successful completion when you see:**
```
Final metrics logged, finishing
Exiting cleanly
```

**Save to memory:**
- Time started: `_________`
- Time ended: `_________`

---

### Run 2: QUANT_ONLY
**Duration:** ~10 min  
**Purpose:** Test quantization  
**Features:** USE_NEW_QUANT=1

```bash
export RUN_ID=quant_only
export USE_BIGRAM_FEATURE=0
export USE_MLP_3X=0
export USE_NEW_QUANT=1

stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log
```

**Run starts when you see:**
```
Rank 0 initialized on worker-0 (rank=0 out of 1)
Loading training data...
```

**Successful completion when you see:**
```
Final metrics logged, finishing
Exiting cleanly
```

---

### Run 3: BIGRAM_4096
**Duration:** ~10 min  
**Purpose:** Test bigram feature  
**Features:** USE_BIGRAM_FEATURE=1, BIGRAM_HASH_SIZE=4096

```bash
export RUN_ID=bigram_4096
export USE_BIGRAM_FEATURE=1
export USE_MLP_3X=0
export USE_NEW_QUANT=0
export BIGRAM_HASH_SIZE=4096
export BIGRAM_FEATURE_DIM=32
export BIGRAM_GATE_INIT=0.1

stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log
```

**Run starts when you see:**
```
Rank 0 initialized on worker-0 (rank=0 out of 1)
Loading training data...
```

**Successful completion when you see:**
```
Final metrics logged, finishing
Exiting cleanly
```

---

## SECTION 3: Strict Validation Checks

### Must-Have Log Lines (Check After Each Run)

```bash
tail -100 logs/${RUN_ID}.log | grep -E "final_int8_zlib_roundtrip|Total submission"
```

**REQUIRED OUTPUT (both lines must appear):**
```
final_int8_zlib_roundtrip val_loss:X.XXXX val_bpb:Y.YYYY
Total submission size int8+zlib: ZZZ,ZZZ bytes
```

**If either is missing:** Run failed → Proceed to "FAILURE DIAGNOSIS" below

---

### Success Indicators (Per Run)

| Check | Expected | Pass/Fail |
|-------|----------|-----------|
| **Log file exists** | `ls -lh logs/<RUN_ID>.log` shows > 1 MB | ✅ |
| **BPB value present** | `grep "final_int8_zlib_roundtrip" logs/<RUN_ID>.log` | ✅ |
| **Artifact size present** | `grep "Total submission size" logs/<RUN_ID>.log` | ✅ |
| **BPB in range** | 1.15–1.35 (baseline, quant, bigram) | ✅ |
| **Artifact < 16MB** | Size < 16,000,000 bytes | ✅ |
| **No NaN in loss** | `grep "loss=nan\|loss=inf" logs/<RUN_ID>.log` returns empty | ✅ |
| **Training continued** | >50 iterations completed (> 5 min of output) | ✅ |

---

### Failure Indicators (When to Stop Early)

#### Indicator 1: CUDA Out of Memory (OOM)
```bash
grep "RuntimeError.*out of memory\|CUDA out of memory" logs/${RUN_ID}.log
```
**Action:** 
- Kill immediately (Ctrl+C)
- Reduce batch size in train_gpt.py
- **Cost impact:** ~$0.40 per failed run
- **Decision:** After 1 OOM, reduce batch and restart from baseline

#### Indicator 2: Loss → NaN (Divergence)
```bash
tail -50 logs/${RUN_ID}.log | grep "loss="
```
**Expected:** Loss gradually decreases (1.2 → 1.0 → stable)  
**Failure:** Loss = NaN or loss = inf or loss > 10

**Action:**
- Kill immediately
- Check feature implementation (likely gate initialization issue)
- **Cost impact:** ~$0.40 per diverged run
- **Decision:** After 1 divergence, lower gate init for bigram features

#### Indicator 3: Stuck / No Progress
**Sign:** No new iteration output for >2 min AND no error message

```bash
tail -5 logs/${RUN_ID}.log  # Shows nothing new for 2+ min
```

**Action:**
- Kill (Ctrl+C)
- Check GPU (run `nvidia-smi`)
- Restart run
- **Cost impact:** ~$0.30 per restart
- **Decision:** If happens twice, investigate data loading

#### Indicator 4: Wrong Feature Flags
**Sign:** Config mismatch in logs

```bash
grep "USE_NEW_QUANT\|USE_BIGRAM_FEATURE\|BIGRAM_HASH_SIZE" logs/${RUN_ID}.log | head -5
```

**Expected (Run 1):**
```
USE_BIGRAM_FEATURE=0
USE_MLP_3X=0
USE_NEW_QUANT=0
```

**Action:**
- Kill immediately
- Verify export statements
- Restart with correct flags
- **Cost impact:** ~$0.40 per restart
- **Decision:** Use copy-paste from commands above; verify before running

---

## SECTION 4: Early-Stop Rules (Preserve Budget)

### STOP & KILL Immediately If:

- [ ] **CUDA OOM error** → Kill (Ctrl+C)
  - Saves: ~$0.40 per run
  - Impact: Batch size too large; investigate in train_gpt.py

- [ ] **Loss → NaN within first 2 min** → Kill
  - Saves: ~$0.40 per run
  - Impact: Feature gate initialization unstable; lower BIGRAM_GATE_INIT

- [ ] **No output for >2 min** (stuck) → Kill
  - Saves: ~$0.40+ per hang
  - Impact: Data loading or GPU hang; check filesystem

- [ ] **BPB outside range (>1.5 or <1.0) after 5 min** → Kill
  - Saves: ~$0.20 per stuck run
  - Impact: Feature breaking baseline; likely needs revert

- [ ] **Artifact size > 16,100,000 bytes** → Kill after 2 min
  - Saves: ~$0.20+ per bloated run
  - Impact: Model too large; compression fails differently

---

### DON'T kill (let run complete even if slow):

- ✅ Loss bounces or fluctuates (normal)
- ✅ GPU util drops temporarily (data loading intermission)
- ✅ BPB slowly improves (good feature, just slow training)
- ✅ Artifact size 15.8–16.0 MB (within margin; finalize check happens at end)

---

## SECTION 5: Exact Execution Loop

### Complete Loop (Copy-Paste Path)

```
STEP 1: Start Run 1 (Baseline)
        ↓
STEP 2: Monitor for early-stop conditions
        ↓
STEP 3: Verify completion (both log lines present)
        ↓
STEP 4: Run analyze_results.py
        ↓
STEP 5: Start Run 2 (Quant-Only)
        ↓
STEP 6: Monitor for early-stop conditions
        ↓
STEP 7: Verify completion
        ↓
STEP 8: Run analyze_results.py (now 2 runs parsed)
        ↓
STEP 9: Get recommendation from recommend_next_step.py
        ↓
STEP 10: Decide: Continue to Run 3 or STOP?
        ↓
STEP 11: If continue, start Run 3 (Bigram-4096)
        ↓
STEP 12: Monitor for early-stop conditions
        ↓
STEP 13: Verify completion
        ↓
STEP 14: Run analyze_results.py (all 3 runs parsed)
        ↓
STEP 15: Get final recommendation from recommend_next_step.py
        ↓
STEP 16: Record decision for Iteration 2
```

---

### Step-by-Step Commands

#### STEP 1: Start Run 1 Baseline
```bash
export RUN_ID=baseline
export USE_BIGRAM_FEATURE=0
export USE_MLP_3X=0
export USE_NEW_QUANT=0

echo "=== Starting $RUN_ID at $(date) ===" 
stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log
echo "=== Completed $RUN_ID at $(date) ==="
```

**Monitor output:** Watch for "Rank 0 initialized" and training iterations

#### STEP 2+3: Verify Baseline
```bash
# Wait for command to return (10 min)
# Then verify:
tail -50 logs/baseline.log | grep -E "final_int8_zlib_roundtrip|Total submission"
```

**Expected output (MUST have both):**
```
final_int8_zlib_roundtrip val_loss:1.2244 val_bpb:1.2244
Total submission size int8+zlib: 15912045 bytes
```

**If missing:** 
```bash
# Review full tail for errors
tail -200 logs/baseline.log
# Diagnose using "FAILURE DIAGNOSIS" section below
```

#### STEP 4: Parse Results After Run 1
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

#### STEP 5: Start Run 2 Quant-Only
```bash
export RUN_ID=quant_only
export USE_BIGRAM_FEATURE=0
export USE_MLP_3X=0
export USE_NEW_QUANT=1

echo "=== Starting $RUN_ID at $(date) ===" 
stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log
echo "=== Completed $RUN_ID at $(date) ==="
```

#### STEP 7: Verify Quant-Only
```bash
tail -50 logs/quant_only.log | grep -E "final_int8_zlib_roundtrip|Total submission"
```

**Expected:** Both lines present, BPB within 1.15–1.35

#### STEP 8: Parse Results After Run 2
```bash
python3 analyze_results.py \
  --log-dir logs \
  --output-json results.json \
  --output-csv results.csv \
  --baseline baseline
```

**Expected:** 2 runs now in results.json

#### STEP 9: Get Recommendation
```bash
python3 recommend_next_step.py --results results.json --baseline baseline
```

**Output will show:**
```
Status: <status>
Best Current Candidate: <run>
Next Step: <recommendation>
```

#### STEP 10: Decision Point
```
Does output say "quant_wins" OR "bigram_4k_wins" OR "both_win"?
  YES → Continue to Step 11 (Run 3)
  NO  → Go to STOP & ANALYZE section below
```

#### STEP 11: Start Run 3 Bigram-4096
```bash
export RUN_ID=bigram_4096
export USE_BIGRAM_FEATURE=1
export USE_MLP_3X=0
export USE_NEW_QUANT=0
export BIGRAM_HASH_SIZE=4096
export BIGRAM_FEATURE_DIM=32
export BIGRAM_GATE_INIT=0.1

echo "=== Starting $RUN_ID at $(date) ===" 
stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log
echo "=== Completed $RUN_ID at $(date) ==="
```

#### STEP 13: Verify Bigram-4096
```bash
tail -50 logs/bigram_4096.log | grep -E "final_int8_zlib_roundtrip|Total submission"
```

**Expected:** Both lines, BPB close to baseline or better

#### STEP 14: Final Parse
```bash
python3 analyze_results.py \
  --log-dir logs \
  --output-json results.json \
  --output-csv results.csv \
  --baseline baseline
```

**Expected:** 3 runs in results.json

#### STEP 15: Final Recommendation
```bash
python3 recommend_next_step.py --results results.json --baseline baseline
```

**Output:** Case A, B, C, or D assigned; Iteration 2 direction clear

#### STEP 16: Record Decision
```bash
# Save recommendation to file for Iteration 2
python3 recommend_next_step.py --results results.json --baseline baseline > ITERATION1_RESULT.txt
```

---

## SECTION 6: Failure Diagnosis

### If Baseline Fails

```bash
# Check for OOM
grep "out of memory\|RuntimeError" logs/baseline.log

# Check for NaN
grep "loss=nan\|loss=inf" logs/baseline.log

# Check for stuck
tail -100 logs/baseline.log | grep "loss="
```

**Root Causes:**
- **OOM:** Batch size too large. Edit train_gpt.py, reduce `BATCH_SIZE`, restart.
- **NaN:** Unusual learning rate or weight initialization. Check training logs carefully; likely config issue.
- **Stuck:** Data loading or GPU hang. Check disk space: `df -h`

**Action:** Fix root cause, restart baseline run.

---

### If Quant-Only Fails

```bash
# Verify quant flag set
grep "USE_NEW_QUANT" logs/quant_only.log

# Check loss progression
tail -100 logs/quant_only.log | grep "loss="
```

**Root Causes:**
- **Quantization diverges:** Try lowering learning rate in train_gpt.py by 10%.
- **Artifact too large:** Quantization scheme is inefficient; this is expected. Proceed anyway; recommend_next_step will flag.
- **OOM during quant:** Quantization adds overhead; reduce batch or check GPU memory.

**Action:** If OOM, reduce batch. Otherwise, proceed; quant divergence is information.

---

### If Bigram-4096 Fails

```bash
# Verify bigram flags
grep "USE_BIGRAM_FEATURE\|BIGRAM_HASH_SIZE\|BIGRAM_GATE_INIT" logs/bigram_4096.log

# Check loss progression
tail -100 logs/bigram_4096.log | grep "loss="
```

**Root Causes:**
- **Gate initialization too high** (0.1 default) → outputs unstable. Lower to 0.01 or 0.05.
- **Bigram feature adds artifact overhead** → Model too large. Check if artifact > 16MB.
- **Hash collision cost high** → Bigram hashing less effective than expected. Proceed; recommend_next_step will note.

**Action:** 
- If loss diverges immediately: Lower `BIGRAM_GATE_INIT=0.01` and restart.
- Otherwise: Proceed; feature validation will show if it's useful.

---

## SECTION 7: Cost & Runtime Estimates

### Per-Run Costs

| Run | Duration | Cost @ $2.50/hr | Notes |
|-----|----------|-----------------|-------|
| baseline | 10 min | $0.42 | Minimal overhead |
| quant_only | 10 min | $0.42 | Quantization adds ~30% memory, same time |
| bigram_4096 | 11 min | $0.46 | Feature adds embedding overhead |
| **Total 3 Runs** | ~32 min | **$1.30** | Excluding parsing, prep |

### Full Iteration 1 Cost Breakdown

| Phase | Time | Cost | Notes |
|-------|------|------|-------|
| Setup & verify (Phase 0–3) | 25 min | $0.00 | No GPU active |
| Run 1 (baseline) | 10 min | $0.42 | |
| Parse + recommend | 3 min | $0.13 | GPU idle (optional keep warm) |
| Run 2 (quant_only) | 10 min | $0.42 | |
| Parse + recommend | 3 min | $0.13 | |
| Run 3 (bigram_4096) | 11 min | $0.46 | |
| Parse + final recommend | 3 min | $0.13 | |
| **Total Iteration 1** | **~65 min** | **$1.69** | Minimal setup; efficient loop |

### Cost With Early Stops (Worst Case)

| Scenario | Early Stops | Total Runs | Total Cost |
|----------|------------|-----------|-----------|
| All pass | 0 | 3 | $1.30 |
| 1 OOM restart | 1 | 4 | $1.72 |
| 2 OOM restarts | 2 | 5 | $2.14 |
| Complete failure | (rare) | 3 + debug | ~$2.50 |

**Budget recommendation:** Budget $3.50–4.00 for full Iteration 1 with safety margin.

---

## SECTION 8: Quick Reference Checklist

### Pre-Run Checklist
- [ ] `nvidia-smi` shows 1× H100 with 80GB free
- [ ] `echo $DATA_PATH` shows correct path
- [ ] `echo $TOKENIZER_PATH` shows correct path
- [ ] `ls logs/` exists and is writable
- [ ] Copy exact commands from sections above (don't improvise)

### During Run Checklist
- [ ] Monitor first 2 min for OOM or NaN errors
- [ ] Let run complete unless early-stop conditions hit
- [ ] Check `tail -50 logs/${RUN_ID}.log | grep "loss="` every 2 min
- [ ] If stuck >2 min with no new output: Kill and diagnose

### After Run Checklist
- [ ] `grep "final_int8_zlib_roundtrip" logs/${RUN_ID}.log` must find line
- [ ] `grep "Total submission" logs/${RUN_ID}.log` must find line
- [ ] BPB value in 1.15–1.35 range
- [ ] Artifact size < 16,100,000 bytes
- [ ] No "nan" or "inf" in logs

### After Each Parse Checklist
- [ ] `cat results.json | python3 -m json.tool` validates JSON
- [ ] Run count in results.json increments (1 → 2 → 3)
- [ ] All fields populated (baseline_gap, artifact_headroom, etc.)

### After Final Recommend Checklist
- [ ] Status is one of: quant_wins, bigram_4k_wins, both_win, all_lose
- [ ] Next step is clear and actionable
- [ ] Ready to proceed to Iteration 2 or stop

---

## Final Notes

**Total GPU uptime for Iteration 1:** ~32 min of active training + ~10 min parsing = 42 min wall-clock.  
**Total cost estimate:** $1.70–2.00 USD (plus RunPod container overhead, typically <$0.50).

**Success metric:** All 3 baseline, quant_only, bigram_4096 runs complete with valid metrics → Iteration 2 direction determined → Stop or pivot based on recommendation.

**If anything unclear:** Refer back to DAY1_GPU_EXECUTION_CHECKLIST.md for Phase-by-Phase detail.

---

**Created:** April 14, 2026  
**Purpose:** Efficient, cost-aware execution on 1×H100  
**Target Task:** Execute Iteration 1 (3 runs) with zero wasted GPU time
