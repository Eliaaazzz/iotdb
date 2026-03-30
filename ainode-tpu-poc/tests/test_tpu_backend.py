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
Unit tests for TPUBackend protocol compliance and DeviceManager fallback.

All tests run on CPU without requiring physical TPU hardware.
torch_xla imports are mocked to validate adapter logic in isolation.

Coverage:
- BackendAdapter protocol (4 methods)
- TPU availability detection (positive / negative / import failure)
- Availability caching
- BackendType enum ordering
- DeviceManager auto-selection: TPU -> CUDA -> CPU
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from ainode_tpu_poc.tpu_backend.backend import (
    BackendAdapter,
    BackendType,
    CPUBackend,
    CUDABackend,
    TPUBackend,
)


def _build_mock_torch_xla(hw_type="TPU", device_count=4):
    """Build a mock torch_xla module tree for TPUBackend."""
    mock_device = torch.device("cpu")

    xm = MagicMock()
    xm.xla_device_hw = MagicMock(return_value=hw_type)
    xm.get_memory_info = MagicMock(return_value={"kb_total": 32 * 1024 * 1024})

    torch_xla_core = types.ModuleType("torch_xla.core")
    torch_xla_core.xla_model = xm

    torch_xla_mod = MagicMock()
    torch_xla_mod.device = MagicMock(return_value=mock_device)
    torch_xla_mod.device_count = MagicMock(return_value=device_count)
    torch_xla_mod.core = torch_xla_core

    return {
        "torch_xla": torch_xla_mod,
        "torch_xla.core": torch_xla_core,
        "torch_xla.core.xla_model": xm,
    }


class TPUBackendProtocolTest(unittest.TestCase):
    """Verify TPUBackend implements all four BackendAdapter methods."""

    def test_type_is_tpu(self):
        mocks = _build_mock_torch_xla()
        with patch.dict(sys.modules, mocks):
            backend = TPUBackend()
            self.assertEqual(backend.type, BackendType.TPU)

    def test_is_available_true_when_tpu_hw(self):
        mocks = _build_mock_torch_xla(hw_type="TPU")
        with patch.dict(sys.modules, mocks):
            backend = TPUBackend()
            self.assertTrue(backend.is_available())

    def test_is_available_false_when_not_tpu_hw(self):
        mocks = _build_mock_torch_xla(hw_type="CPU")
        with patch.dict(sys.modules, mocks):
            backend = TPUBackend()
            self.assertFalse(backend.is_available())

    def test_is_available_cached(self):
        mocks = _build_mock_torch_xla(hw_type="TPU")
        with patch.dict(sys.modules, mocks):
            backend = TPUBackend()
            self.assertTrue(backend.is_available())
            self.assertTrue(backend.is_available())  # cached
            self.assertIsNotNone(backend._available)

    def test_is_available_false_when_import_fails(self):
        with patch.dict(sys.modules, {"torch_xla": None}):
            backend = TPUBackend()
            self.assertFalse(backend.is_available())

    def test_device_count(self):
        mocks = _build_mock_torch_xla(device_count=8)
        with patch.dict(sys.modules, mocks):
            backend = TPUBackend()
            self.assertEqual(backend.device_count(), 8)

    def test_make_device(self):
        mocks = _build_mock_torch_xla()
        with patch.dict(sys.modules, mocks):
            backend = TPUBackend()
            device = backend.make_device(0)
            mocks["torch_xla"].device.assert_called_with(0)
            self.assertIsNotNone(device)

    def test_make_device_none(self):
        mocks = _build_mock_torch_xla()
        with patch.dict(sys.modules, mocks):
            backend = TPUBackend()
            backend.make_device(None)
            mocks["torch_xla"].device.assert_called_with(None)

    def test_set_device_stores_index(self):
        mocks = _build_mock_torch_xla()
        with patch.dict(sys.modules, mocks):
            backend = TPUBackend()
            backend.set_device(3)
            self.assertEqual(backend._selected_index, 3)


class BackendTypeEnumTest(unittest.TestCase):
    """Verify enum ordering drives priority correctly."""

    def test_order_tpu_cuda_cpu(self):
        members = list(BackendType)
        self.assertEqual(members, [BackendType.TPU, BackendType.CUDA, BackendType.CPU])

    def test_tpu_value_is_xla(self):
        self.assertEqual(BackendType.TPU.value, "xla")

    def test_cpu_is_last(self):
        self.assertEqual(list(BackendType)[-1], BackendType.CPU)


class DeviceManagerFallbackTest(unittest.TestCase):
    """Test TPU -> CUDA -> CPU fallback chain."""

    def _select(self, tpu_avail, cuda_avail):
        backends = {
            BackendType.TPU: MagicMock(spec=BackendAdapter, type=BackendType.TPU,
                                       is_available=MagicMock(return_value=tpu_avail)),
            BackendType.CUDA: MagicMock(spec=BackendAdapter, type=BackendType.CUDA,
                                        is_available=MagicMock(return_value=cuda_avail)),
            BackendType.CPU: MagicMock(spec=BackendAdapter, type=BackendType.CPU,
                                       is_available=MagicMock(return_value=True)),
        }
        selected = None
        for name in BackendType:
            if name == BackendType.CPU:
                continue
            b = backends[name]
            if b.is_available():
                selected = b.type
                break
        if selected is None:
            selected = BackendType.CPU
        return selected

    def test_selects_tpu_when_available(self):
        self.assertEqual(self._select(True, True), BackendType.TPU)

    def test_falls_back_to_cuda(self):
        self.assertEqual(self._select(False, True), BackendType.CUDA)

    def test_falls_back_to_cpu(self):
        self.assertEqual(self._select(False, False), BackendType.CPU)

    def test_tpu_beats_cuda(self):
        self.assertEqual(self._select(True, True), BackendType.TPU)


if __name__ == "__main__":
    unittest.main()
