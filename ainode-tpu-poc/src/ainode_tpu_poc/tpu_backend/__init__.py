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

"""TPU BackendAdapter implementation and XLA optimization utilities.

Validates the TPU device detection, tensor movement, automatic fallback,
and XLA compilation optimizations described in proposal Sections 4.2 and 4.2.4.
"""

from ainode_tpu_poc.tpu_backend.backend import BackendType
from ainode_tpu_poc.tpu_backend.backend import BackendAdapter
from ainode_tpu_poc.tpu_backend.backend import TPUBackend
from ainode_tpu_poc.tpu_backend.backend import CUDABackend
from ainode_tpu_poc.tpu_backend.backend import CPUBackend
from ainode_tpu_poc.tpu_backend.backend import DeviceManager
from ainode_tpu_poc.tpu_backend.xla_utils import initialize_xla_cache
from ainode_tpu_poc.tpu_backend.xla_utils import bucket_input_shape
from ainode_tpu_poc.tpu_backend.xla_utils import tpu_inference
from ainode_tpu_poc.tpu_backend.xla_utils import check_xla_health

__all__ = [
    'BackendType',
    'BackendAdapter',
    'TPUBackend',
    'CUDABackend',
    'CPUBackend',
    'DeviceManager',
    'initialize_xla_cache',
    'bucket_input_shape',
    'tpu_inference',
    'check_xla_health',
]
