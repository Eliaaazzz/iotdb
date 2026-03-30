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
Unit tests for TPUBackend (BackendAdapter protocol compliance).

All tests run on CPU without requiring physical TPU hardware.
torch_xla imports are mocked to validate the adapter logic in isolation.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from iotdb.ainode.core.device.backend.base import BackendType


def _build_mock_torch_xla(hw_type="TPU", device_count=4):
    """Build a mock torch_xla module tree that satisfies TPUBackend's imports."""
    mock_device = torch.device("cpu")  # stand-in for XLA device

    # Build xla_model mock
    xm = MagicMock()
    xm.xla_device_hw = MagicMock(return_value=hw_type)
    xm.get_memory_info = MagicMock(return_value={"kb_total": 32 * 1024 * 1024})

    # Build torch_xla.core with xla_model attribute
    torch_xla_core = types.ModuleType("torch_xla.core")
    torch_xla_core.xla_model = xm

    # Build torch_xla top-level with proper attribute chain
    # (import A.B.C as x resolves via getattr chain, not sys.modules lookup)
    torch_xla_mod = MagicMock()
    torch_xla_mod.device = MagicMock(return_value=mock_device)
    torch_xla_mod.device_count = MagicMock(return_value=device_count)
    torch_xla_mod.core = torch_xla_core

    return {
        "torch_xla": torch_xla_mod,
        "torch_xla.core": torch_xla_core,
        "torch_xla.core.xla_model": xm,
    }


class TestTPUBackendProtocol(unittest.TestCase):
    """Verify TPUBackend implements all four BackendAdapter methods."""

    def test_type_is_tpu(self):
        mocks = _build_mock_torch_xla()
        with patch.dict(sys.modules, mocks):
            from iotdb.ainode.core.device.backend.tpu_backend import TPUBackend
            backend = TPUBackend()
            self.assertEqual(backend.type, BackendType.TPU)

    def test_is_available_true_when_tpu_hw(self):
        mocks = _build_mock_torch_xla(hw_type="TPU")
        with patch.dict(sys.modules, mocks):
            from iotdb.ainode.core.device.backend.tpu_backend import TPUBackend
            backend = TPUBackend()
            self.assertTrue(backend.is_available())

    def test_is_available_false_when_not_tpu_hw(self):
        """When xla_device_hw reports 'CPU', TPUBackend should not claim available."""
        mocks = _build_mock_torch_xla(hw_type="CPU")
        with patch.dict(sys.modules, mocks):
            from iotdb.ainode.core.device.backend.tpu_backend import TPUBackend
            backend = TPUBackend()
            self.assertFalse(backend.is_available())

    def test_is_available_cached(self):
        """Second call to is_available() should use cached result."""
        mocks = _build_mock_torch_xla(hw_type="TPU")
        with patch.dict(sys.modules, mocks):
            from iotdb.ainode.core.device.backend.tpu_backend import TPUBackend
            backend = TPUBackend()
            self.assertTrue(backend.is_available())
            # Second call uses cached result (no re-import needed)
            self.assertTrue(backend.is_available())
            self.assertIsNotNone(backend._available)

    def test_is_available_false_when_import_fails(self):
        """When torch_xla is not installed, is_available returns False."""
        with patch.dict(sys.modules, {"torch_xla": None}):
            from iotdb.ainode.core.device.backend.tpu_backend import TPUBackend
            backend = TPUBackend()
            self.assertFalse(backend.is_available())

    def test_device_count(self):
        mocks = _build_mock_torch_xla(device_count=8)
        with patch.dict(sys.modules, mocks):
            from iotdb.ainode.core.device.backend.tpu_backend import TPUBackend
            backend = TPUBackend()
            self.assertEqual(backend.device_count(), 8)

    def test_make_device_returns_xla_device(self):
        mocks = _build_mock_torch_xla()
        with patch.dict(sys.modules, mocks):
            from iotdb.ainode.core.device.backend.tpu_backend import TPUBackend
            backend = TPUBackend()
            device = backend.make_device(0)
            mocks["torch_xla"].device.assert_called_with(0)
            self.assertIsNotNone(device)

    def test_make_device_none_index(self):
        mocks = _build_mock_torch_xla()
        with patch.dict(sys.modules, mocks):
            from iotdb.ainode.core.device.backend.tpu_backend import TPUBackend
            backend = TPUBackend()
            device = backend.make_device(None)
            mocks["torch_xla"].device.assert_called_with(None)
            self.assertIsNotNone(device)

    def test_set_device_stores_index(self):
        mocks = _build_mock_torch_xla()
        with patch.dict(sys.modules, mocks):
            from iotdb.ainode.core.device.backend.tpu_backend import TPUBackend
            backend = TPUBackend()
            backend.set_device(3)
            self.assertEqual(backend._selected_index, 3)


class TestBackendTypeEnum(unittest.TestCase):
    """Verify BackendType enum ordering: TPU before CUDA before CPU."""

    def test_tpu_is_first(self):
        members = list(BackendType)
        self.assertEqual(members[0], BackendType.TPU)
        self.assertEqual(members[1], BackendType.CUDA)
        self.assertEqual(members[2], BackendType.CPU)

    def test_tpu_value(self):
        self.assertEqual(BackendType.TPU.value, "xla")


if __name__ == "__main__":
    unittest.main()
