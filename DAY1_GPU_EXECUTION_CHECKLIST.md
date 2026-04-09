# Day 1 GPU Execution Checklist

**Purpose:** Launch GPU experiments immediately when credits arrive.  
**Duration:** ~45 min for full Day 1 (baseline + 2 feature runs).  
**Success Criteria:** 3 runs complete → results.json generated → recommendation output.

---

## PHASE 0: RunPod Instance Launch (5 min)

### 0.1 Create RunPod Instance
- [ ] Go to https://www.runpod.io/console/pods
- [ ] Click **"+ Deploy New Pod"**
- [ ] Select **GPU Pod** (not Serverless)
- [ ] Choose GPU: **1x A100 (80GB)** or **1x H100** (cheapest available)
- [ ] Select **PyTorch 2.0+** template (prebuilt CUDA environment)
- [ ] Set Volume: **100 GB** (for dataset + logs)
- [ ] Click **"Deploy Pod"**
- [ ] Wait ~2 min for instance to start
- [ ] Note: **Pod ID** and **SSH connection string**

### 0.2 Connect via SSH
```bash
# Paste the SSH command from RunPod console (looks like):
ssh -i <key> root@<ip> -p <port>
```
- [ ] Confirm SSH connection successful (you should see `root@` prompt)

---

## PHASE 1: Environment Setup (10 min)

### 1.1 Verify CUDA and GPU
```bash
nvidia-smi
```
**Expected output:**
```
NVIDIA-SMI X.XX.XX    Driver Version: XXX.XX
GPU Name             Compute SM
A100-80GB            8.0
used: 0 MiB, free: 80000+ MiB
```
- [ ] GPU visible and free memory > 70 GB

### 1.2 Verify Python 3.10+
```bash
python3 --version
```
**Expected:** `Python 3.10.X` or higher
- [ ] Python 3.10+ installed

### 1.3 Create Working Directory
```bash
cd /workspace
pwd
```
**Expected:** `/workspace` (or your chosen path)
- [ ] Working directory confirmed

---

## PHASE 2: Repository & Dependencies (15 min)

### 2.1 Clone Repository
```bash
git clone https://github.com/<your-username>/parameter-golf.git
cd parameter-golf
```
- [ ] Repo cloned successfully
- [ ] Confirm files present: `ls -la | grep -E "train_gpt|requirements|data"`

### 2.2 Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
**Expected packages:**
- `torch >= 2.0`
- `torchvision`
- `torchdata`
- `sentencepiece`
- `numpy`
- `tensorboard`

**Watch for:** No errors marked `ERROR` or `FAILED`
- [ ] All dependencies installed without errors

### 2.3 Verify Key Imports
```bash
python3 -c "import torch; import sentencepiece; print('✓ Imports OK')"
```
**Expected output:** `✓ Imports OK`
- [ ] All imports successful

---

## PHASE 3: Data & Tokenizer Verification (10 min)

### 3.1 Create Directories
```bash
mkdir -p data/datasets data/tokenizers logs
ls -la data/
```
**Expected:** Three directories visible: `datasets/`, `tokenizers/`, and root `logs/` visible
- [ ] Directories created

### 3.2 Download or Verify Dataset
**Option A: Download FineWeb (if not on instance)**
```bash
cd data/datasets
# Download FineWeb 1024-token vocabulary, 80 shards (~8B tokens)
# (Exact download command depends on your data source)
# Expected result: fineweb10B_sp1024/ folder with 80 shards
ls -lh fineweb10B_sp1024/ | head -20
```

**Option B: Verify Existing Dataset**
```bash
ls -lh data/datasets/fineweb10B_sp1024/ | wc -l
```
**Expected:** 80+ files (shards), ~500 MB each
- [ ] Dataset folder has 80 shards, ~40 GB total

### 3.3 Verify Tokenizer
```bash
ls -lh data/tokenizers/
```
**Expected:** `fineweb_1024_bpe.model` (~200-300 KB)
- [ ] Tokenizer file present and > 100 KB

### 3.4 Set Environment Variables (for all runs)
```bash
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export MAX_WALLCLOCK_SECONDS=600

# Verify
echo "DATA_PATH=$DATA_PATH"
echo "TOKENIZER_PATH=$TOKENIZER_PATH"
echo "MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS"
```
**Expected:** All three variables printed correctly
- [ ] Environment variables set

---

## PHASE 4: Run 1 — BASELINE (10 min execution + wait)

**Purpose:** Establish anchor metric (BPB + artifact size).  
**Feature Flags:** All OFF (plain model, no quantization, no bigram)  
**Expected BPB:** ~1.22–1.25 (depends on architecture)  
**Expected Artifact Size:** ~15–16 MB

### 4.1 Execute Baseline Run
```bash
cd /workspace/parameter-golf
export RUN_ID=baseline
export USE_BIGRAM_FEATURE=0
export USE_MLP_3X=0
export USE_NEW_QUANT=0

stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log
```

**Do NOT interrupt.** This will run for ~10 min.

- [ ] Run started (you see training iterations printed to terminal)

### 4.2 Monitor Real-Time Output
While running, look for:
```
...
[Iteration N/M] loss=X.XXX ...
[Iteration N/M] val_loss=X.XXX val_bpb=X.XXX
...
Final metrics logged, finishing
```

**Early stop conditions (interrupt with `Ctrl+C` if you see):**
- [ ] **STOP if:** CUDA out of memory (OOM error) → reduce batch size in train_gpt.py
- [ ] **STOP if:** `RuntimeError: ...` that repeats → kill and check logs
- [ ] **STOP if:** No output for >2 min → kill, check GPU (run `nvidia-smi`)

### 4.3 Verify Baseline Completion
```bash
# Wait for prompt to return, then:
tail -50 logs/baseline.log | grep -E "final_int8_zlib_roundtrip|Total submission"
```

**Expected output MUST contain both lines:**
```
final_int8_zlib_roundtrip val_loss:X.XXXX val_bpb:Y.YYYY
Total submission size int8+zlib: ZZZ,ZZZ bytes
```

**If missing:** Run failed → review logs with `tail -200 logs/baseline.log`
- [ ] Both lines present in logs/baseline.log
- [ ] BPB in reasonable range (1.20–1.30)

### 4.4 Record Baseline Metrics (for reference)
```bash
grep "final_int8_zlib_roundtrip\|Total submission" logs/baseline.log | tail -2
```
**Write down:**
- Baseline BPB: `___________`
- Baseline artifact size (MB): `___________`

---

## PHASE 5: Parse Baseline Results (2 min)

### 5.1 Run Analysis Tool
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
Wrote results.json (1 record)
Wrote results.csv (1 record)
```

- [ ] No errors printed
- [ ] Both `results.json` and `results.csv` created

### 5.2 Verify Results File
```bash
cat results.json | python3 -m json.tool | head -30
```

**Expected:**
```json
[
  {
    "run_id": "baseline",
    "final_roundtrip_bpb": 1.224,
    "artifact_size": 15912000,
    "artifact_headroom": 88000,
    "baseline_gap": 0,
    "improvement_pct": 0,
    "notes": ""
  }
]
```

- [ ] JSON is valid (no syntax errors)
- [ ] `final_roundtrip_bpb` and `artifact_size` present

---

## PHASE 6: Run 2 — QUANT_ONLY (10 min execution + wait)

**Purpose:** Test quantization alone (most probable improvement).  
**Feature Flags:** `USE_NEW_QUANT=1`, others OFF  
**Expected:** BPB improvement 0.5–2% if it works  
**Risk:** Low (quantization is mature)

### 6.1 Execute Quant-Only Run
```bash
export RUN_ID=quant_only
export USE_BIGRAM_FEATURE=0
export USE_MLP_3X=0
export USE_NEW_QUANT=1

stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log
```

- [ ] Run started

### 6.2 Verify Completion
```bash
tail -50 logs/quant_only.log | grep -E "final_int8_zlib_roundtrip|Total submission"
```

**Both lines must be present:**
- [ ] Lines present
- [ ] BPB close to baseline (within ±5%)

---

## PHASE 7: Parse & Get Recommendation (2 min)

### 7.1 Update Analysis
```bash
python3 analyze_results.py \
  --log-dir logs \
  --output-json results.json \
  --output-csv results.csv \
  --baseline baseline
```

**Expected:** 2 runs parsed
- [ ] results.json now has 2 records

### 7.2 Get Automated Recommendation
```bash
python3 recommend_next_step.py --results results.json --baseline baseline
```

**Will print decision table and recommendation:**

| Status | Next Step | Decision |
|--------|-----------|----------|
| `quant_wins` | Run bigram_4096 next | Continue to Run 3 ✅ |
| `quant_loses` | STOP; debug logs | ❌ **STOP HERE** |
| `all_lose` | STOP; inspect logs | ❌ **STOP HERE** |

- [ ] Read `Status` field in output
- [ ] If status is `quant_wins` (or similar WIN): **Continue to Run 3**
- [ ] If status is `all_lose` or `all_artifacts_exceed_limit`: **STOP and review logs**

**Decision Point:**
```
Is Status == "quant_wins" or similar (not "all_lose")?
  YES → Continue to Run 3
  NO  → STOP; run: tail -100 logs/quant_only.log
```

---

## PHASE 8: Run 3 — BIGRAM_4096 (10 min execution + wait)

**Purpose:** Test bigram feature (second highest probability).  
**Feature Flags:** `USE_BIGRAM_FEATURE=1`, `BIGRAM_HASH_SIZE=4096`, others OFF  
**Expected:** BPB improvement 0.3–1.5% if it works  
**Risk:** Low–Medium (new feature, but independent)

### 8.1 Execute Bigram-4096 Run
```bash
export RUN_ID=bigram_4096
export USE_BIGRAM_FEATURE=1
export USE_MLP_3X=0
export USE_NEW_QUANT=0
export BIGRAM_HASH_SIZE=4096

stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log
```

- [ ] Run started

### 8.2 Verify Completion
```bash
tail -50 logs/bigram_4096.log | grep -E "final_int8_zlib_roundtrip|Total submission"
```

**Both lines must be present:**
- [ ] Lines present
- [ ] BPB close to baseline (within ±5%)

---

## PHASE 9: Final Parse & Next Decision (2 min)

### 9.1 Final Analysis Update
```bash
python3 analyze_results.py \
  --log-dir logs \
  --output-json results.json \
  --output-csv results.csv \
  --baseline baseline
```

**Expected:** 3 runs parsed
- [ ] results.json now has 3 records

### 9.2 Get Final Recommendation
```bash
python3 recommend_next_step.py --results results.json --baseline baseline
```

**Possible outcomes:**

| Status | Next Step |
|--------|-----------|
| `both_win` | Run combination: `USE_NEW_QUANT=1 USE_BIGRAM_FEATURE=1` |
| `quant_wins` | Keep quant only; investigate bigram separately later |
| `bigram_4k_wins` | Keep bigram; investigate quant separately later |
| `all_lose` | STOP; debug model or hyperparameters |
| `all_artifacts_exceed_limit` | STOP; model too large for submission |

- [ ] Record the recommendation status
- [ ] If `both_win` with safe artifact headroom: **Proceed to Run 4 (combination)**
- [ ] Otherwise: **Document results and stop for today**

---

## QUICK REFERENCE: Command Sequence

```bash
# Setup (one-time)
ssh root@<runpod-ip> -p <port>
cd /workspace/parameter-golf
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export MAX_WALLCLOCK_SECONDS=600

# Run Baseline
export RUN_ID=baseline USE_BIGRAM_FEATURE=0 USE_MLP_3X=0 USE_NEW_QUANT=0
stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log

# Parse & Recommend
python3 analyze_results.py --log-dir logs --output-json results.json --output-csv results.csv --baseline baseline
python3 recommend_next_step.py --results results.json --baseline baseline

# Run Quant-Only (if recommended)
export RUN_ID=quant_only USE_BIGRAM_FEATURE=0 USE_MLP_3X=0 USE_NEW_QUANT=1
stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log

# Parse & Recommend again
python3 analyze_results.py --log-dir logs --output-json results.json --output-csv results.csv --baseline baseline
python3 recommend_next_step.py --results results.json --baseline baseline

# Run Bigram-4096 (if recommended)
export RUN_ID=bigram_4096 USE_BIGRAM_FEATURE=1 USE_MLP_3X=0 USE_NEW_QUANT=0 BIGRAM_HASH_SIZE=4096
stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log

# Final Parse & Recommend
python3 analyze_results.py --log-dir logs --output-json results.json --output-csv results.csv --baseline baseline
python3 recommend_next_step.py --results results.json --baseline baseline
```

---

## LOGS & DEBUGGING

### Where Logs Are Stored
```bash
logs/baseline.log       # Baseline run
logs/quant_only.log     # Quant-only run
logs/bigram_4096.log    # Bigram-4096 run
results.json            # Parsed metrics (JSON)
results.csv             # Parsed metrics (CSV)
```

### Quick Debug: Check Latest Metrics
```bash
tail -5 logs/<RUN_ID>.log
```

### If a Run Fails: Review Full Log
```bash
tail -200 logs/<RUN_ID>.log | less
```

### If Training Diverges (loss goes NaN or explodes)
```bash
grep "loss=" logs/<RUN_ID>.log | tail -50
```

---

## SUCCESS CHECKLIST (End of Day 1)

- [ ] Baseline run complete with valid metrics
- [ ] Quant-only run complete with valid metrics
- [ ] Bigram-4096 run complete with valid metrics
- [ ] results.json contains all 3 runs
- [ ] Final recommendation output generated
- [ ] Next action (run combination, stop, or iterate) identified

**If all boxes are checked:** Day 1 is complete. Save results, document findings, and proceed with next iteration based on recommendation.

---

## COMMON ISSUES & FIXES

| Issue | Check | Fix |
|-------|-------|-----|
| GPU out of memory (OOM) | Run `nvidia-smi` during training | Reduce `BATCH_SIZE` in train_gpt.py |
| Training diverges (loss → NaN) | `grep "loss=" logs/<run>.log` | Lower learning rate or check data |
| Log file empty | `ls -lh logs/` | Check if run actually started; SSH may have dropped |
| Tokenizer not found | `ls -l data/tokenizers/` | Download tokenizer or fix `TOKENIZER_PATH` |
| Dataset shards not loaded | `ls -l data/datasets/fineweb10B_sp1024/ \| wc -l` | Verify 80 shards present and readable |
| analyze_results.py fails | `python3 analyze_results.py --log-dir logs ...` | Ensure log files have expected format; check tail of log |

---

**Created:** April 9, 2026  
**Purpose:** Day 1 GPU execution without thinking  
**Next Review:** After first 3 runs complete
