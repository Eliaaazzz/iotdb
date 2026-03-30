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

import logging
from typing import Optional

import torch

from iotdb.ainode.core.device.backend.base import BackendAdapter, BackendType

logger = logging.getLogger(__name__)


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
            self._available = hw == "TPU"
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
