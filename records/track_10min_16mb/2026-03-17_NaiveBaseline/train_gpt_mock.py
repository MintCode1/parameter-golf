#!/usr/bin/env python3
"""Mock training script to test the harness without requiring PyTorch."""
import sys
import time
import json
from pathlib import Path

# Simulate training with realistic output
print("Starting training...", file=sys.stderr)
print("Loading data from: /Users/mahinnaveen/Documents/GitHub/parameter-golf/data/datasets/fineweb10B_sp1024")

# Simulate training iterations
for i in range(3):
    print(f"Step {i+1}/3: loss=2.5{i}, acc=0.{i}2")
    time.sleep(0.5)

# Output the final metrics that the harness will parse
print("\nTraining complete!")

# The key line the harness looks for - with final_*_roundtrip format and val_bpb
final_bpb = 1.2543
final_loss = 2.3567

print(f"\n{'='*60}")
print(f"FINAL METRICS:")
print(f"final_mock_model_roundtrip val_loss:{final_loss} val_bpb:{final_bpb}")
print(f"{'='*60}")

# Create a small artifact file for the harness to detect
artifact_path = Path("artifact_model.pt")
with open(artifact_path, "w") as f:
    json.dump({"epoch": 1, "loss": final_loss}, f)
    
print(f"\nModel saved to {artifact_path}")
print("Done!")
