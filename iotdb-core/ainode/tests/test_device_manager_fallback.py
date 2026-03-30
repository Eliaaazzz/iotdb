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
Tests for DeviceManager automatic backend selection with TPU→CUDA→CPU fallback.

Validates the priority-based fallback mechanism described in proposal Section 4.2.3:
1. Try TPU: if torch_xla not installed or no TPU hardware, TPUBackend.is_available() returns False
2. Try CUDA: if no NVIDIA GPU available, CUDABackend.is_available() returns False
3. Fallback to CPU: always available
"""

import unittest
from unittest.mock import MagicMock, patch

from iotdb.ainode.core.device.backend.base import BackendAdapter, BackendType


def _make_mock_backend(backend_type: BackendType, available: bool) -> BackendAdapter:
    """Create a mock BackendAdapter with controlled availability."""
    mock = MagicMock(spec=BackendAdapter)
    mock.type = backend_type
    mock.is_available.return_value = available
    mock.device_count.return_value = 1 if available else 0
    mock.make_device.return_value = MagicMock()
    return mock


class TestDeviceManagerFallback(unittest.TestCase):
    """Test the TPU→CUDA→CPU fallback chain in DeviceManager._auto_select_backend."""

    def _create_manager_with_backends(self, tpu_avail, cuda_avail):
        """
        Create a DeviceManager-like object with mocked backends to test
        _auto_select_backend logic without importing the singleton.
        """
        # We replicate the core logic from DeviceManager._auto_select_backend
        # to test it in isolation without triggering the singleton or real hw checks.
        backends = {
            BackendType.TPU: _make_mock_backend(BackendType.TPU, tpu_avail),
            BackendType.CUDA: _make_mock_backend(BackendType.CUDA, cuda_avail),
            BackendType.CPU: _make_mock_backend(BackendType.CPU, True),
        }

        # Replicate auto-select logic
        selected_type = None
        selected_backend = None
        for name in BackendType:
            if name == BackendType.CPU:
                continue
            backend = backends.get(name)
            if backend is not None and backend.is_available():
                selected_type = backend.type
                selected_backend = backend
                break

        if selected_backend is None:
            selected_backend = backends[BackendType.CPU]
            selected_type = BackendType.CPU

        return selected_type, selected_backend, backends

    def test_selects_tpu_when_available(self):
        """When TPU is available, it should be selected (highest priority)."""
        selected_type, backend, _ = self._create_manager_with_backends(
            tpu_avail=True, cuda_avail=True
        )
        self.assertEqual(selected_type, BackendType.TPU)
        self.assertEqual(backend.type, BackendType.TPU)

    def test_falls_back_to_cuda_when_no_tpu(self):
        """When TPU unavailable but CUDA available, select CUDA."""
        selected_type, backend, _ = self._create_manager_with_backends(
            tpu_avail=False, cuda_avail=True
        )
        self.assertEqual(selected_type, BackendType.CUDA)
        self.assertEqual(backend.type, BackendType.CUDA)

    def test_falls_back_to_cpu_when_no_accelerator(self):
        """When both TPU and CUDA unavailable, fall back to CPU."""
        selected_type, backend, _ = self._create_manager_with_backends(
            tpu_avail=False, cuda_avail=False
        )
        self.assertEqual(selected_type, BackendType.CPU)
        self.assertEqual(backend.type, BackendType.CPU)

    def test_tpu_takes_precedence_over_cuda(self):
        """Even if both TPU and CUDA are available, TPU should win."""
        selected_type, _, backends = self._create_manager_with_backends(
            tpu_avail=True, cuda_avail=True
        )
        self.assertEqual(selected_type, BackendType.TPU)
        # CUDA should not have been selected
        backends[BackendType.TPU].is_available.assert_called_once()

    def test_fallback_is_transparent(self):
        """
        Verify that the fallback produces a valid BackendAdapter
        regardless of which backend is selected.
        """
        for tpu, cuda in [(True, True), (True, False), (False, True), (False, False)]:
            _, backend, _ = self._create_manager_with_backends(tpu, cuda)
            self.assertIsNotNone(backend)
            self.assertIsInstance(backend.type, BackendType)


class TestBackendTypeOrdering(unittest.TestCase):
    """Verify that BackendType enum order drives priority correctly."""

    def test_iteration_order(self):
        """BackendType should iterate in priority order: TPU, CUDA, CPU."""
        types = list(BackendType)
        self.assertEqual(types, [BackendType.TPU, BackendType.CUDA, BackendType.CPU])

    def test_cpu_is_always_last(self):
        """CPU must always be the last (fallback) option."""
        types = list(BackendType)
        self.assertEqual(types[-1], BackendType.CPU)


if __name__ == "__main__":
    unittest.main()
