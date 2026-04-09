# Full 2-Iteration Experiment Strategy

**Purpose:** Define all decision pathways and next experiments before GPU execution.  
**Scope:** Iteration 1 (Day 1) → Iteration 2 (Day 2 based on outcomes)  
**Objective:** Move immediately after results with zero decision latency.

---

## ITERATION 1: Day 1 (Initial 3 Runs)

### Runs in Iteration 1

| Run | Feature Flags | Hypothesis | Risk |
|-----|--------------|-----------|------|
| **baseline** | All OFF | Establish anchor BPB + artifact size | Minimal |
| **quant_only** | `USE_NEW_QUANT=1` | Quantization improves BPB without adding overhead | Low |
| **bigram_4096** | `USE_BIGRAM_FEATURE=1, BIGRAM_HASH_SIZE=4096` | Bigram lexical embeddings improve BPB without exceeding artifact limit | Low–Medium |

### Expected Outcomes (Iteration 1 → Iteration 2 Decision)

After running all 3, recommendation system will output **one of 4 cases:**

| Case | Condition | Status | Next Action |
|------|-----------|--------|------------|
| **A** | Only `quant_only` wins | `quant_wins` | Run feature combinations (quant + bigram) |
| **B** | Only `bigram_4096` wins | `bigram_4k_wins` | Test bigram scalability (bigram_8192) or quant separately |
| **C** | Both `quant_only` AND `bigram_4096` win | `both_win` | Test combination first; if safe, proceed to secondary features |
| **D** | Neither wins (or artifacts exceed limit) | `all_lose` or `all_artifacts_exceed_limit` | **STOP & PIVOT**: Debug hyperparameters or model size |

---

## ITERATION 2: Day 2 (Next ~3–5 Runs Based on Case)

### Case A: `quant_only` Wins (Only Quantization Improves)

**Insight:** Quantization works; bigram feature didn't help.

**Iteration 2 Experiments:**

#### Run 2.A.1: Test Bigram + Quant Combination
```bash
export RUN_ID=combo_quant_bigram4k
export USE_BIGRAM_FEATURE=1
export USE_MLP_3X=0
export USE_NEW_QUANT=1
export BIGRAM_HASH_SIZE=4096
export BIGRAM_FEATURE_DIM=32
export BIGRAM_GATE_INIT=0.1

stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log
```

| Hypothesis | Test | Success Criteria |
|-----------|------|------------------|
| Features don't interfere; combination is safe | Do quant + bigram improve BPB together? | BPB ≤ quant_only BPB AND artifact_headroom > 100KB |
| Reason: Quant alone works, so add bigram to see if synergy exists. | Can we get both improvements? | If YES → run mlp3x next; if NO → explore bigram hyperparameters |

**Decision After 2.A.1:**
- **Improve?** → Proceed to Run 2.A.2 (test mlp3x variant)
- **Worsen?** → Revert bigram; pivot to testing alternative quantization or MLP scaling

---

#### Run 2.A.2: Test Quant + MLP3x Combination (if 2.A.1 succeeds)
```bash
export RUN_ID=combo_quant_mlp3x
export USE_BIGRAM_FEATURE=0
export USE_MLP_3X=1
export USE_NEW_QUANT=1

stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log
```

| Hypothesis | Test | Success Criteria |
|-----------|------|------------------|
| MLP expansion + quantization safe | Does wider MLP + quant help BPB? | BPB < quant_only BPB AND artifact still safe |
| Reason: Quant works; test architectural width scaling. | Does model capacity × quant yield synergy? | Artifact_headroom > 500KB (safe for final composition) |

**Decision After 2.A.2:**
- **Both 2.A.1 & 2.A.2 improve?** → Final run: test all three together (Run 2.A.3)
- **Only one improves?** → Focus on that; stop other branch
- **Neither improves?** → Finalize with `quant_only`; stop iterating

---

#### Run 2.A.3: Test All Three Together (if both 2.A.1 and 2.A.2 improve)
```bash
export RUN_ID=combo_quant_bigram4k_mlp3x
export USE_BIGRAM_FEATURE=1
export USE_MLP_3X=1
export USE_NEW_QUANT=1
export BIGRAM_HASH_SIZE=4096

stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log
```

| Hypothesis | Test | Success Criteria |
|-----------|------|------------------|
| All three features compose safely | Final triple-feature model improves BPB? | BPB < best_single AND artifact_headroom > 100KB |
| Reason: Both individual combos worked; test triple synergy. | Best composition of quant + bigram + mlp? | Record final artifact size; ensure < 16MB |

**Decision After 2.A.3:**
- **Improves?** → **FINALIZE**: Use this as submission model
- **Worsens?** → **FINALIZE**: Revert to best combo from 2.A.1 or 2.A.2
- **Stop** iterating; prioritize refining top-3 models for final submission

---

**Case A Summary:**
- **Goal:** Find best combination with quant base
- **Sequence:** quant+bigram → quant+mlp3x → (maybe) all three
- **Stop Condition:** Combination worsens BPB or artifact headroom drops below 100KB
- **Expected Duration:** 2–3 runs; ~30 min

---

### Case B: `bigram_4096` Wins (Only Bigram Improves)

**Insight:** Bigram works; quantization didn't help (or hurt).

**Iteration 2 Experiments:**

#### Run 2.B.1: Test Bigram Scalability (bigram_8192)
```bash
export RUN_ID=bigram_8192
export USE_BIGRAM_FEATURE=1
export USE_MLP_3X=0
export USE_NEW_QUANT=0
export BIGRAM_HASH_SIZE=8192
export BIGRAM_FEATURE_DIM=32
export BIGRAM_GATE_INIT=0.1

stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log
```

| Hypothesis | Test | Success Criteria |
|-----------|------|------------------|
| Larger hash table captures more bigram diversity | Does 8192-hash improve over 4096? | BPB ≤ bigram_4096 BPB AND artifact_headroom > 500KB |
| Reason: Bigram works; test if model is undercapacity for bigram hashing. | Can we squeeze more from hash expansion? | If YES → test 16K next; if NO → 4K is optimal |

**Decision After 2.B.1:**
- **2.B.1 improves significantly?** → Test bigram_16384 (Run 2.B.2)
- **2.B.1 improves marginally (<0.2%)?** → Stop scaling; 4K is likely optimal
- **2.B.1 worsens?** → Artifact too tight; stick with 4K

---

#### Run 2.B.2: Test Further Bigram Scaling (if 2.B.1 improves >0.2%)
```bash
export RUN_ID=bigram_16384
export USE_BIGRAM_FEATURE=1
export USE_MLP_3X=0
export USE_NEW_QUANT=0
export BIGRAM_HASH_SIZE=16384
export BIGRAM_FEATURE_DIM=32
export BIGRAM_GATE_INIT=0.1

stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log
```

| Hypothesis | Test | Success Criteria |
|-----------|------|------------------|
| Even larger hash table | Does 16K improve over 8K? | BPB improves AND artifact_headroom > 300KB |
| Reason: Bigram scalability; find saturation point. | Where does hash scaling saturate? | Record: 4K vs 8K vs 16K BPB trend |

**Decision After 2.B.2:**
- **16K improves over 8K?** → Use 16K; stop scaling (costs too much artifact)
- **16K marginal or worse?** → Revert to best (4K or 8K)
- **Proceed to Run 2.B.3** (test quant + best bigram)

---

#### Run 2.B.3: Test Quant + Best Bigram Variant
```bash
export RUN_ID=combo_bigram_best_quant
export USE_BIGRAM_FEATURE=1
export USE_MLP_3X=0
export USE_NEW_QUANT=1
export BIGRAM_HASH_SIZE=4096  # or 8192 if that was best; adjust below

stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log
```

**Note:** Adjust `BIGRAM_HASH_SIZE` to whichever was best from 2.B.1/2.B.2.

| Hypothesis | Test | Success Criteria |
|-----------|------|------------------|
| Can we add quant to best bigram? | Does bigram + quant improve both? | BPB < best_bigram AND artifact_headroom > 100KB |
| Reason: Quantization failed alone; test if it helps bigram model. | Does quant synergize with bigram? | If YES → final candidate; if NO → stick with bigram |

**Decision After 2.B.3:**
- **Improves?** → **FINALIZE**: Use bigram + quant combo
- **Worsens?** → **FINALIZE**: Use best bigram alone (no quant)
- **Stop iterating**

---

**Case B Summary:**
- **Goal:** Find optimal bigram variant + test quant compatibility
- **Sequence:** bigram_8K → (maybe) bigram_16K → bigram+quant
- **Stop Condition:** Hash-size scaling saturates or quant + bigram artifact limit exceeded
- **Expected Duration:** 2–3 runs; ~30 min

---

### Case C: Both `quant_only` AND `bigram_4096` Win

**Insight:** Both features improve independently. Test safe combination.

**Iteration 2 Experiments:**

#### Run 2.C.1: Test Combination First (Most Likely Path)
```bash
export RUN_ID=combo_quant_bigram4k
export USE_BIGRAM_FEATURE=1
export USE_MLP_3X=0
export USE_NEW_QUANT=1
export BIGRAM_HASH_SIZE=4096

stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log
```

| Hypothesis | Test | Success Criteria |
|-----------|------|------------------|
| Safe synergy of independent wins | Does quant + bigram compose? | BPB < both individual runs AND artifact safe (>100KB) |
| Reason: Both won separately; 85% chance combo also wins. | Do independent features multiply gains? | Expect 0.5–1.5% additional improvement |

**Decision After 2.C.1:**
- **Improves? (BPB < min(quant, bigram))** → Proceed to Run 2.C.2 (test scaling)
- **Worsens or artifact tight?** → Decide which base to keep; try alternative scaling
- **Marginal?** → Likely good enough; consider stopping to preserve GPU time

---

#### Run 2.C.2: Test Bigram Scaling with Quant (if 2.C.1 succeeds)
```bash
export RUN_ID=combo_quant_bigram8k
export USE_BIGRAM_FEATURE=1
export USE_MLP_3X=0
export USE_NEW_QUANT=1
export BIGRAM_HASH_SIZE=8192

stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log
```

| Hypothesis | Test | Success Criteria |
|-----------|------|------------------|
| Can we scale bigram hash while keeping quant? | Does bigram_8K + quant improve over 4K + quant? | BPB improves AND artifact_headroom stays > 300KB |
| Reason: 4K+quant worked; test if bigger hash helps. | What's the capacity sweet spot? | Record: 4K vs 8K + quant trade-off |

**Decision After 2.C.2:**
- **8K improves?** → Test MLP addition (Run 2.C.3)
- **8K marginal or artifacts tight?** → Stick with 4K+quant; done
- **Any path tight on artifact?** → Final model is best so far; stop iterating

---

#### Run 2.C.3: Test Triple Feature (if 2.C.2 succeeds and artifact safe)
```bash
export RUN_ID=combo_quant_bigram_mlp3x
export USE_BIGRAM_FEATURE=1
export USE_MLP_3X=1
export USE_NEW_QUANT=1
export BIGRAM_HASH_SIZE=4096  # adjust if 8K was better

stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log
```

| Hypothesis | Test | Success Criteria |
|-----------|------|------------------|
| Triple-feature synergy | Does quant + bigram + mlp3x compose? | BPB < quant_bigram AND artifact > 50KB headroom |
| Reason: Both pairs worked; test full triple. | Maximum composition of all three | Record improvement vs baseline |

**Decision After 2.C.3:**
- **Improves?** → **FINALIZE**: This is your best model
- **Worsens?** → **FINALIZE**: Revert to best from 2.C.1 or 2.C.2
- **Stop iterating**; prepare for final submission

---

**Case C Summary:**
- **Goal:** Safely combine both winners; find optimal scaling
- **Sequence:** quant+bigram_4K → (maybe) quant+bigram_8K → (maybe) all three
- **Stop Condition:** Artifact limit breached or BPB improvements saturate
- **Expected Duration:** 2–3 runs; ~30 min
- **Most Optimistic Path:** All three work together → immediate final candidate

---

### Case D: None Win (`all_lose` or `all_artifacts_exceed_limit`)

**Insight:** No feature improved. Model/hyperparameter issue.

**Decisions for Case D:**

#### Decision D.1: Diagnose the Problem
```bash
# Check logs for divergence patterns
tail -100 logs/quant_only.log | grep "loss="
tail -100 logs/bigram_4096.log | grep "loss="
```

**Potential Issues:**
1. **Model too large** → "all_artifacts_exceed_limit" (model + quantization > 16MB)
   - Solution: Reduce layers, embedding dim, or hidden dim in train_gpt.py
   
2. **Feature implementation unstable** → Training diverges (loss → NaN)
   - Solution: Lower feature learning rate or gate initialization
   
3. **Hyperparameters suboptimal** → Features don't help baseline even works worse
   - Solution: Sweep learning rate, warmup, or weight decay

#### Decision D.2: Choose Pivot Strategy

| Symptom | Root Cause | Pivot Action | Run Next |
|---------|-----------|--------------|----------|
| All artifacts > 16MB | Model bloat | Reduce model size | Re-run baseline with smaller config |
| Training diverges (loss→NaN) | Feature instability | Lower feature gate init | Re-run with `BIGRAM_GATE_INIT=0.01` or `0.05` |
| BPB worse than baseline | Hyperparams bad | Tune baseline first | Sweep baseline learning rate |
| Artifacts within limit but no improve | Feature not useful | Accept baseline as best | **STOP**: Use baseline for final submission |

---

**Case D Recommendation Logic:**

1. **Inspect logs** (5 min): Is there a clear failure mode (NaN, OOM, etc.)?
2. **Diagnose** (2 min): Map to root cause from table above
3. **Pivot** (1 run): Test root-cause fix once
4. **Re-evaluate** (2 min): Did fix work?
   - YES → Return to Case A/B/C logic with fixed features
   - NO → Accept baseline; prepare for submission

---

**Case D Summary:**
- **Goal:** Diagnose and fix
- **Stop Condition:** After 1 pivot run; don't chase indefinitely
- **Expected Duration:** 1 re-run; ~15 min + analysis
- **Final Fallback:** Use baseline model (reliable, no features)

---

## Summary Decision Tree

```
Day 1 (Iteration 1): Run baseline, quant_only, bigram_4096
                           ↓
                    Get recommendation
                           ↓
     ┌──────────────────────┼──────────────────────┐
     ↓                      ↓                      ↓
  CASE A            CASE B                    CASE C
  quant_wins        bigram_wins               both_win
  (quant only)      (bigram only)             (quant & bigram)
     ↓                      ↓                      ↓
  Run 2 plan:       Run 2 plan:                Run 2 plan:
  combo_quant+      bigram scale +            combo_quant+
  bigram_4k         quant test                bigram + scaling
     ↓                      ↓                      ↓
  → mlp3x combo     → best bigram              → all three
  → all three       → quant+bigram             → finalize
  → finalize             ↓
                    → finalize
                    
                           CASE D (all_lose)
                                  ↓
                          Diagnose issue
                                  ↓
                          Pivot & retest
                                  ↓
                          Accept baseline
```

---

## Iteration 2 Command Reference

### Case A (Quant Wins)
```bash
# 2.A.1: Quant + Bigram
export RUN_ID=combo_quant_bigram4k && export USE_BIGRAM_FEATURE=1 && export USE_MLP_3X=0 && export USE_NEW_QUANT=1 && export BIGRAM_HASH_SIZE=4096
stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log

# 2.A.2: Quant + MLP3x
export RUN_ID=combo_quant_mlp3x && export USE_BIGRAM_FEATURE=0 && export USE_MLP_3X=1 && export USE_NEW_QUANT=1
stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log

# 2.A.3: All Three
export RUN_ID=combo_quant_bigram4k_mlp3x && export USE_BIGRAM_FEATURE=1 && export USE_MLP_3X=1 && export USE_NEW_QUANT=1 && export BIGRAM_HASH_SIZE=4096
stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log
```

### Case B (Bigram Wins)
```bash
# 2.B.1: Bigram 8K
export RUN_ID=bigram_8192 && export USE_BIGRAM_FEATURE=1 && export USE_MLP_3X=0 && export USE_NEW_QUANT=0 && export BIGRAM_HASH_SIZE=8192
stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log

# 2.B.2: Bigram 16K (if 8K improves >0.2%)
export RUN_ID=bigram_16384 && export USE_BIGRAM_FEATURE=1 && export USE_MLP_3X=0 && export USE_NEW_QUANT=0 && export BIGRAM_HASH_SIZE=16384
stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log

# 2.B.3: Best Bigram + Quant
export RUN_ID=combo_bigram_best_quant && export USE_BIGRAM_FEATURE=1 && export USE_MLP_3X=0 && export USE_NEW_QUANT=1 && export BIGRAM_HASH_SIZE=4096
stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log
```

### Case C (Both Win)
```bash
# 2.C.1: Quant + Bigram 4K
export RUN_ID=combo_quant_bigram4k && export USE_BIGRAM_FEATURE=1 && export USE_MLP_3X=0 && export USE_NEW_QUANT=1 && export BIGRAM_HASH_SIZE=4096
stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log

# 2.C.2: Quant + Bigram 8K
export RUN_ID=combo_quant_bigram8k && export USE_BIGRAM_FEATURE=1 && export USE_MLP_3X=0 && export USE_NEW_QUANT=1 && export BIGRAM_HASH_SIZE=8192
stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log

# 2.C.3: All Three
export RUN_ID=combo_quant_bigram_mlp3x && export USE_BIGRAM_FEATURE=1 && export USE_MLP_3X=1 && export USE_NEW_QUANT=1 && export BIGRAM_HASH_SIZE=4096
stdbuf -oL torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/${RUN_ID}.log
```

---

## Stopping Criteria & Pivot Points

### Stop Early If:
- [ ] **Artifact headroom drops below 50KB** → Combination too risky; finalize with best safe model
- [ ] **BPB improvement plateaus** (<0.1% gain per run) → Scaling not worth GPU time
- [ ] **Training diverges** (loss → NaN) → Feature implementation needs fixing, not scaling
- [ ] **Feature adds >2% overhead** → Bad feature; don't try combinations

### Pivot to Different Feature If:
- [ ] **Current feature alone doesn't help** → Move to next feature (Case B/C logic)
- [ ] **Combination worsens >1%** → Revert to best single feature
- [ ] **Artifact limit nearing** (<100KB headroom) → Stop scaling, finalize

### Accept Final Model When:
- [ ] **Best model identified** (lowest BPB with safe artifact)
- [ ] **No scaling or combination improves >0.1%** (diminishing returns)
- [ ] **GPU time spent justifies remaining gains** (1 more run if potential > 0.5%, else stop)

---

## Execution Timeline

| Phase | Iteration | Runs | Time | Decision Point |
|-------|-----------|------|------|----------------|
| Day 1 | Iter 1 | baseline, quant, bigram | 45 min | Case A/B/C/D determined |
| Day 2 | Iter 2 (Case A) | combo+bigram, combo+mlp, all three | 45 min | Choose best; finalize |
| Day 2 | Iter 2 (Case B) | bigram_8K, bigram_16K, quant+bigram | 45 min | Choose best; finalize |
| Day 2 | Iter 2 (Case C) | quant+bigram_4K, quant+bigram_8K, all three | 45 min | Choose best; finalize |
| Day 2 | Iter 2 (Case D) | Pivot run (fix + retest) | 15 min | Diagnose or finalize baseline |

**Total: 90–120 min of GPU time for 2 full iterations + final model selection**

---

## Next Steps After Iteration 2

1. **Identify final model** (lowest BPB, safe artifact)
2. **Extract & validate submission** (run submission packer once)
3. **Ready for final submission** or proceed to Iteration 3 if budget allows

---

**Created:** April 9, 2026  
**Status:** Ready to execute immediately after Day 1 results  
**Next Review:** After Iteration 2 completes
