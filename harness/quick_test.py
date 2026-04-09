#!/usr/bin/env python3
from runner import ExperimentRunner
from pathlib import Path
import tempfile

runner = ExperimentRunner(Path(tempfile.mkdtemp()))

# Test int6
logs = "final_int6_zlib_roundtrip val_loss:2.1234 val_bpb:1.5678"
result = runner._parse_final_bpb(logs)
print(f"int6 parse result: {result}")
assert result == 1.5678, f"Expected 1.5678, got {result}"
print("✓ int6 test passed")

# Test mixed
logs2 = "final_mixed_zlib_roundtrip val_loss:2.1234 val_bpb:1.5678"
result2 = runner._parse_final_bpb(logs2)
print(f"mixed parse result: {result2}")
assert result2 == 1.5678, f"Expected 1.5678, got {result2}"
print("✓ mixed test passed")

# Test with exact
logs3 = """
final_int8_zlib_roundtrip val_loss:2.0727 val_bpb:1.2244
final_int8_zlib_roundtrip_exact val_loss:2.07269931 val_bpb:1.2243657
"""
result3 = runner._parse_final_bpb(logs3)
print(f"exact parse result: {result3}")
assert result3 == 1.2244, f"Expected 1.2244, got {result3}"
print("✓ exact test passed")

print("\nAll parsing tests passed!")
