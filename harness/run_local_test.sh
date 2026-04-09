#!/bin/bash
# Local Test Script - Run this to validate harness

set -e

REPO_ROOT="/Users/mahinnaveen/Documents/GitHub/parameter-golf"
HARNESS_DIR="$REPO_ROOT/harness"
SCRIPT_PATH="$REPO_ROOT/records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py"

echo "=========================================="
echo "PARAMETER GOLF HARNESS - LOCAL TEST MODE"
echo "=========================================="
echo ""

cd "$HARNESS_DIR"

echo "Step 1: Verify data dependencies"
echo "  Checking DATA_PATH..."
DATA_PATH="$REPO_ROOT/data/datasets/fineweb10B_sp1024"
if [ -d "$DATA_PATH" ]; then
    echo "  ✓ $DATA_PATH exists"
else
    echo "  ✗ $DATA_PATH does NOT exist"
    echo "  You need to:"
    echo "    1. Download/create the data"
    echo "    2. Or update DATA_PATH in runner.py"
fi

echo "  Checking TOKENIZER_PATH..."
TOKENIZER_PATH="$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model"
if [ -f "$TOKENIZER_PATH" ]; then
    echo "  ✓ $TOKENIZER_PATH exists"
else
    echo "  ✗ $TOKENIZER_PATH does NOT exist"
    echo "  You need to:"
    echo "    1. Download/create the tokenizer"
    echo "    2. Or update TOKENIZER_PATH in runner.py"
fi

echo ""
echo "Step 2: Run harness in local test mode"
echo "  Command:"
echo ""
echo "python3 runner.py \\"
echo "  --script $SCRIPT_PATH \\"
echo "  --reproduce \\"
echo "  --local_test"
echo ""
echo "=========================================="
echo ""

# Uncomment below to actually run the test
# python3 runner.py \
#   --script "$SCRIPT_PATH" \
#   --reproduce \
#   --local_test
