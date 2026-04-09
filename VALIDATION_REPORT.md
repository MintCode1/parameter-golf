## Workflow Validation Report

### Summary
The full experiment → analysis → decision workflow has been tested with 3 realistic scenarios and 3 edge cases. The system works correctly for the main paths and correctly identifies invalid artifacts. **Minor improvements needed for tighter artifact headroom warnings and edge case messaging.**

---

## Scenarios Tested

### ✅ Scenario A: quant_only WINS
- Status: `quant_wins`
- Recommendation: Run bigram_4096 next
- Verdict: **CORRECT** — cleanest single-feature win, recommend next step appropriately

### ✅ Scenario B: bigram_4096 WINS
- Status: `bigram_4k_wins`
- Recommendation: Keep bigram_4096, do NOT try bigram_8192 (artifact headroom too low)
- Verdict: **CORRECT** — correctly identifies 50KB headroom as insufficient

### ✅ Scenario C: NONE WIN
- Status: `all_lose`
- Recommendation: STOP, inspect logs
- Verdict: **CORRECT** — properly calls out no improvement, prevents blind combination attempts

---

## Edge Cases Tested

### Edge Case 1: Winning run with INVALID_ARTIFACT ✅
- Result: BPB improves but artifact_size > 16MB
- Outcome: Correctly classified as `INVALID_ARTIFACT` (artifact check happens before BPB comparison)
- Verdict: **WORKING CORRECTLY**

### Edge Case 2: All runs INVALID_ARTIFACT ⚠️
- Result: All non-baseline runs exceed 16MB
- Current Output: "no_experiments_run" (misleading)
- Expected Output: "all_artifacts_exceeds_limit" with message to debug artifact pressure
- Verdict: **NEEDS FIX** — messaging is confusing when all runs fail the artifact check

### Edge Case 3: Both win but different artifact pressure ⚠️
- Result: quant_only wins with 10KB headroom, bigram_4096 wins with 200KB headroom
- Current Output: Recommends combination without warning about tight headroom
- Expected Output: Same recommendation but add warning about quant's tight artifact pressure
- Verdict: **NEEDS IMPROVEMENT** — should warn when combining runs with very different headroom values

---

## Identified Issues & Fixes

### Issue 1: Edge Case 2 messaging (MEDIUM PRIORITY)
**Problem:** When all runs are `INVALID_ARTIFACT`, the code returns "no_experiments_run" instead of acknowledging the actual problem.

**Fix:** Add a check in `recommend_next_step()` to detect when all wins + invalid = empty wins list, and distinguish between "no experiments run yet" vs "all experiments exceeded artifact limit".

**Solution:**
```python
# In recommend_next_step(), after counting wins/loses/invalid:
if len(wins) == 0 and len(invalid) > 0 and len(loses) == 0:
    return {
        "status": "all_artifacts_exceed_limit",
        "best_candidate": baseline.run_id,
        "next_step": "STOP. All experiments exceeded 16MB artifact limit. Debug model weight compression or reduce model size.",
        "runs_to_avoid": [r.run_id for r in invalid],
        "reasoning": "Every run produced artifacts > 16MB. This is not a feature problem; the base model+quantization strategy is too large. Inspect train_gpt.py for weight savings opportunities (reduce layers, dim, etc).",
    }
```

### Issue 2: Combination warning for tight headroom (MEDIUM PRIORITY)
**Problem:** When recommending a combination of multiple winning runs, the system doesn't warn if one of them has very tight artifact headroom (e.g., < 100KB), which could make the combination fail.

**Fix:** Add a headroom check in the "both_win" path and similar combination paths.

**Solution:**
```python
# When both quant and bigram win:
tight_headroom = []
if quant_only and quant_only.artifact_headroom and quant_only.artifact_headroom < 100_000:
    tight_headroom.append(f"quant_only ({quant_only.artifact_headroom:,} bytes)")
if bigram_4096 and bigram_4096.artifact_headroom and bigram_4096.artifact_headroom < 100_000:
    tight_headroom.append(f"bigram_4096 ({bigram_4096.artifact_headroom:,} bytes)")

warning = ""
if tight_headroom:
    warning = f" ⚠️ WARNING: {', '.join(tight_headroom)} have tight artifact headroom (<100KB); combination may exceed 16MB."
    
return {
    ...
    "next_step": f"Test combination: USE_NEW_QUANT=1 USE_BIGRAM_FEATURE=1 BIGRAM_HASH_SIZE=4096{warning}",
    ...
}
```

### Issue 3: Hardcoded 500KB threshold (LOW PRIORITY)
**Problem:** The 500KB threshold for "artifact headroom is safe" is hardcoded in multiple places.

**Fix:** Make it a configuration constant at the top of the script.

**Solution:**
```python
# At top of recommend_next_step.py:
ARTIFACT_SAFE_HEADROOM = 500_000  # bytes
ARTIFACT_WARNING_HEADROOM = 100_000  # bytes
```

---

## Validation Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| Main path (single feature wins) | ✅ PASS | Correctly recommends next step |
| Both features win | ✅ PASS | Recommends combination |
| None win | ✅ PASS | Recommends stop |
| Invalid artifacts | ✅ PASS | Correctly classified |
| All invalid artifacts | ⚠️ NEEDS FIX | Confusing "no_experiments_run" message |
| Combination artifact warning | ⚠️ NEEDS IMPROVEMENT | Should warn about tight headroom before combination |
| Hardcoded thresholds | ✅ LOW PRIORITY | Works but should be configurable |

---

## Before GPU Runs: Action Items

### Priority 1 (Must fix before GPU):
- [ ] Add check for "all_artifacts_exceed_limit" scenario

### Priority 2 (Should fix):
- [ ] Add tight headroom warning to combination recommendations
- [ ] Make artifact thresholds configurable

### Priority 3 (Can defer):
- [ ] None identified

---

## Testing Instructions

To test these fixes after implementation:

```bash
# Test main scenarios
python3 recommend_next_step.py --results SCENARIO_A_QUANT_WINS.json
python3 recommend_next_step.py --results SCENARIO_B_BIGRAM_WINS.json
python3 recommend_next_step.py --results SCENARIO_C_NONE_WIN.json

# Verify outputs match expected recommendations above
```

---

## Workflow Readiness

- ✅ `analyze_results.py`: Correctly parses logs and extracts metrics
- ✅ `recommend_next_step.py`: Main logic works, minor messaging improvements needed
- ✅ Experiment commands: Ready to execute
- ✅ Decision strategy: Sound and validated

**Recommendation:** Fix Priority 1 issue (all_artifacts_exceed_limit messaging) before GPU runs. Priority 2 can be applied after first GPU iteration if you prefer.
