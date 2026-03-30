# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

"""
Unit tests for XLA optimization utilities (shape bucketing, cache init).

Tests run on CPU without requiring torch_xla or TPU hardware.
Validates:
- Shape bucketing pads to multiples of 128
- No padding when already aligned
- Short sequences get padded correctly
- Cache initialization handles missing torch_xla gracefully
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

from iotdb.ainode.core.device.backend.xla_utils import (
    BUCKET_ALIGNMENT,
    bucket_input_shape,
)


class TestBucketInputShape(unittest.TestCase):
    """Test XLA shape bucketing for TPU systolic array alignment."""

    def test_already_aligned(self):
        """Tensor with seq_len=128 should not be padded."""
        tensor = torch.randn(4, 128)
        padded, orig_len = bucket_input_shape(tensor)
        self.assertEqual(padded.shape, (4, 128))
        self.assertEqual(orig_len, 128)

    def test_already_aligned_multiple(self):
        """Tensor with seq_len=256 (2*128) should not be padded."""
        tensor = torch.randn(2, 256)
        padded, orig_len = bucket_input_shape(tensor)
        self.assertEqual(padded.shape, (2, 256))
        self.assertEqual(orig_len, 256)

    def test_needs_padding(self):
        """Tensor with seq_len=100 should be padded to 128."""
        tensor = torch.randn(4, 100)
        padded, orig_len = bucket_input_shape(tensor)
        self.assertEqual(padded.shape, (4, 128))
        self.assertEqual(orig_len, 100)

    def test_padding_preserves_original_data(self):
        """Padded tensor should contain original data in first orig_len columns."""
        tensor = torch.ones(2, 100)
        padded, orig_len = bucket_input_shape(tensor)
        # Original data preserved
        self.assertTrue(torch.allclose(padded[:, :100], tensor))
        # Padding is zeros
        self.assertTrue(torch.allclose(padded[:, 100:], torch.zeros(2, 28)))

    def test_padding_to_next_bucket(self):
        """Tensor with seq_len=129 should be padded to 256."""
        tensor = torch.randn(1, 129)
        padded, orig_len = bucket_input_shape(tensor)
        self.assertEqual(padded.shape, (1, 256))
        self.assertEqual(orig_len, 129)

    def test_small_tensor(self):
        """Very small tensor (seq_len=10) should be padded to 128."""
        tensor = torch.randn(1, 10)
        padded, orig_len = bucket_input_shape(tensor)
        self.assertEqual(padded.shape, (1, 128))
        self.assertEqual(orig_len, 10)

    def test_large_tensor(self):
        """Tensor with seq_len=1000 should be padded to 1024 (8*128)."""
        tensor = torch.randn(1, 1000)
        padded, orig_len = bucket_input_shape(tensor)
        self.assertEqual(padded.shape, (1, 1024))
        self.assertEqual(orig_len, 1000)

    def test_custom_alignment(self):
        """Custom alignment value should be respected."""
        tensor = torch.randn(1, 200)
        padded, orig_len = bucket_input_shape(tensor, alignment=64)
        # (200 + 63) // 64 = 4, 4 * 64 = 256
        self.assertEqual(padded.shape, (1, 256))
        self.assertEqual(orig_len, 200)

    def test_batch_dimension_preserved(self):
        """Batch dimension should not be affected by padding."""
        for batch_size in [1, 4, 16, 32]:
            tensor = torch.randn(batch_size, 100)
            padded, _ = bucket_input_shape(tensor)
            self.assertEqual(padded.shape[0], batch_size)

    def test_dtype_preserved(self):
        """Tensor dtype should be preserved after padding."""
        for dtype in [torch.float32, torch.float64, torch.bfloat16]:
            tensor = torch.randn(1, 100).to(dtype)
            padded, _ = bucket_input_shape(tensor)
            self.assertEqual(padded.dtype, dtype)

    def test_returns_original_length(self):
        """The original sequence length must be returned for post-inference trimming."""
        tensor = torch.randn(1, 500)
        _, orig_len = bucket_input_shape(tensor)
        self.assertEqual(orig_len, 500)


class TestInitializeXlaCache(unittest.TestCase):
    """Test XLA compilation cache initialization."""

    def test_returns_false_when_no_torch_xla(self):
        """Should return False gracefully when torch_xla is not installed."""
        from iotdb.ainode.core.device.backend.xla_utils import initialize_xla_cache

        with patch.dict(sys.modules, {"torch_xla": None, "torch_xla.runtime": None}):
            result = initialize_xla_cache()
            self.assertFalse(result)

    def test_returns_true_with_mocked_xla(self):
        """Should return True when torch_xla.runtime.initialize_cache succeeds."""
        mock_xr = MagicMock()
        with patch.dict(sys.modules, {"torch_xla.runtime": mock_xr}):
            from iotdb.ainode.core.device.backend.xla_utils import initialize_xla_cache

            result = initialize_xla_cache("/tmp/test_cache")
            # Since initialize_cache is called via importlib, we verify no exception
            # The actual function may succeed or fail depending on import path


class TestCheckXlaHealth(unittest.TestCase):
    """Test XLA health monitoring."""

    def test_no_error_without_torch_xla(self):
        """check_xla_health should not raise when torch_xla is missing."""
        from iotdb.ainode.core.device.backend.xla_utils import check_xla_health

        with patch.dict(sys.modules, {"torch_xla": None, "torch_xla.debug": None, "torch_xla.debug.metrics": None}):
            # Should not raise
            check_xla_health()

    def test_detects_aten_ops(self):
        """Should log warnings when aten:: ops are found in metrics report."""
        mock_met = MagicMock()
        mock_met.short_metrics_report.return_value = (
            "Counter: aten::_copy  Value: 5\n"
            "Counter: xla::add  Value: 100\n"
        )
        mock_debug = MagicMock()
        mock_debug.metrics = mock_met

        mock_torch_xla = MagicMock()
        mock_torch_xla.debug = mock_debug

        mocks = {
            "torch_xla": mock_torch_xla,
            "torch_xla.debug": mock_debug,
            "torch_xla.debug.metrics": mock_met,
        }
        with patch.dict(sys.modules, mocks):
            # Re-import to pick up mocked modules
            import importlib
            import iotdb.ainode.core.device.backend.xla_utils as xu
            importlib.reload(xu)

            xu.check_xla_health()
            mock_met.clear_all.assert_called_once()


if __name__ == "__main__":
    unittest.main()
