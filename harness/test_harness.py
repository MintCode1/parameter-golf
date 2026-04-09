#!/usr/bin/env python3
"""
Correctness tests for the Parameter Golf harness.
"""

import unittest
import tempfile
import os
from pathlib import Path
from runner import ExperimentRunner


class TestExperimentRunner(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(tempfile.mkdtemp())
        self.runner = ExperimentRunner(self.workspace)

    def tearDown(self):
        # Clean up temp files
        import shutil
        shutil.rmtree(self.workspace)

    def test_parse_final_bpb_success(self):
        """Test parsing final BPB from logs."""
        logs = """
        step:1000/20000 val_loss:2.0606 val_bpb:1.2172
        final_int8_zlib_roundtrip val_loss:2.0727 val_bpb:1.2244
        final_int8_zlib_roundtrip_exact val_loss:2.07269931 val_bpb:1.2243657
        """

        bpb = self.runner._parse_final_bpb(logs)
        self.assertEqual(bpb, 1.2244)  # Should prefer rounded version

    def test_parse_final_bpb_exact_only(self):
        """Test parsing when only exact version exists."""
        logs = """
        final_int8_zlib_roundtrip_exact val_loss:2.07269931 val_bpb:1.2243657
        """

        bpb = self.runner._parse_final_bpb(logs)
        self.assertEqual(bpb, 1.2243657)

    def test_parse_final_bpb_no_match(self):
        """Test failure when no valid line found."""
        logs = """
        step:1000/20000 val_loss:2.0606 val_bpb:1.2172
        Some other output
        """

        bpb = self.runner._parse_final_bpb(logs)
        self.assertIsNone(bpb)

    def test_parse_final_bpb_int6(self):
        """Test parsing int6 roundtrip scores."""
        logs = "final_int6_zlib_roundtrip val_loss:2.1234 val_bpb:1.5678"
        bpb = self.runner._parse_final_bpb(logs)
        self.assertEqual(bpb, 1.5678)

    def test_parse_final_bpb_mixed(self):
        """Test parsing mixed precision scores."""
        logs = "final_mixed_zlib_roundtrip val_loss:2.1234 val_bpb:1.5678"
        bpb = self.runner._parse_final_bpb(logs)
        self.assertEqual(bpb, 1.5678)

    def test_get_artifact_size(self):
        """Test artifact size calculation."""
        # Create mock files
        quant_file = self.workspace / "final_model.int8.ptz"
        script_file = self.workspace / "train_gpt.py"

        quant_file.write_bytes(b"x" * 1000)
        script_file.write_bytes(b"x" * 500)

        size, breakdown = self.runner._get_artifact_size(self.workspace)
        self.assertEqual(size, 1500)
        self.assertIn("final_model.int8.ptz", breakdown)
        self.assertIn("train_gpt.py", breakdown)

    def test_get_artifact_size_missing_files(self):
        """Test artifact size when files don't exist."""
        size, breakdown = self.runner._get_artifact_size(self.workspace)
        self.assertEqual(size, 0)
        self.assertEqual(len(breakdown), 0)


if __name__ == "__main__":
    unittest.main()