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
BackendAdapter protocol, BackendType enum, and all three backend
implementations (TPU, CUDA, CPU) plus the DeviceManager auto-selector.

This is a standalone extract of AINode's device management layer,
validating the TPU integration described in proposal Section 4.2.
"""

import logging
from enum import Enum
from typing import Optional, Protocol

import torch

logger = logging.getLogger(__name__)


# ---------- Protocol & Enum ----------

class BackendType(Enum):
    """
    Supported computation backends.
    AINode selects the first available backend in enum order.
    """
    TPU = "xla"    # priority 0 (highest)
    CUDA = "cuda"  # priority 1
    CPU = "cpu"    # priority 2 (fallback)


class BackendAdapter(Protocol):
    type: BackendType

    def is_available(self) -> bool: ...
    def device_count(self) -> int: ...
    def make_device(self, index: Optional[int]) -> torch.device: ...
    def set_device(self, index: int) -> None: ...


# ---------- TPUBackend ----------

class TPUBackend(BackendAdapter):
    """BackendAdapter implementation for Google TPU via torch_xla."""

    type = BackendType.TPU

    def __init__(self):
        self._available = None  # lazy check
        self._selected_index = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm

            device = torch_xla.device()
            hw = xm.xla_device_hw(device)
            self._available = (hw == "TPU")
            if self._available:
                logger.info(
                    "TPU detected: %d device(s), memory info: %s",
                    torch_xla.device_count(),
                    xm.get_memory_info(device),
                )
        except (ImportError, RuntimeError) as e:
            logger.debug("TPU not available: %s", e)
            self._available = False
        return self._available

    def device_count(self) -> int:
        import torch_xla
        return torch_xla.device_count()

    def make_device(self, index: Optional[int] = None) -> torch.device:
        import torch_xla
        return torch_xla.device(index)

    def set_device(self, index: int) -> None:
        # PJRT process isolation is configured before startup;
        # keep the ordinal for bookkeeping.
        self._selected_index = index


# ---------- CUDABackend ----------

class CUDABackend(BackendAdapter):
    """BackendAdapter implementation for NVIDIA CUDA GPUs."""

    type = BackendType.CUDA

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def device_count(self) -> int:
        return torch.cuda.device_count()

    def make_device(self, index: Optional[int] = None) -> torch.device:
        if index is None:
            raise ValueError("CUDA backend requires a valid device index")
        return torch.device(f"cuda:{index}")

    def set_device(self, index: int) -> None:
        torch.cuda.set_device(index)


# ---------- CPUBackend ----------

class CPUBackend(BackendAdapter):
    """BackendAdapter implementation for CPU (always available)."""

    type = BackendType.CPU

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def make_device(self, index: Optional[int] = None) -> torch.device:
        return torch.device("cpu")

    def set_device(self, index: int) -> None:
        return None


# ---------- DeviceManager ----------

class DeviceManager:
    """
    Unified device entry point with automatic TPU -> CUDA -> CPU fallback.

    Mirrors AINode's ``DeviceManager`` singleton. The PoC validates that
    the enum-iteration auto-selection works correctly with the new TPU
    entry at priority 0.
    """

    def __init__(self):
        self.backends: dict[BackendType, BackendAdapter] = {
            BackendType.TPU: TPUBackend(),
            BackendType.CUDA: CUDABackend(),
            BackendType.CPU: CPUBackend(),
        }
        self.type: BackendType
        self.backend: BackendAdapter = self._auto_select_backend()

    def _auto_select_backend(self) -> BackendAdapter:
        for name in BackendType:
            if name == BackendType.CPU:
                continue
            backend = self.backends.get(name)
            if backend is not None and backend.is_available():
                self.type = backend.type
                logger.info("Selected backend: %s", backend.type.value)
                return backend
        logger.info("No accelerator available, falling back to CPU.")
        backend = self.backends[BackendType.CPU]
        self.type = backend.type
        return backend

    def move_tensor(self, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        return tensor.to(device)
