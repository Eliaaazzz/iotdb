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
XLA Compilation Cache and Shape Bucketing utilities.

Validates the optimizations described in proposal Section 4.2.4:
- Persistent compilation cache via ``xr.initialize_cache()``.
- Fixed-shape padding to multiples of 128 (TPU systolic array width).
- BFloat16 autocast for TPU-native mixed precision.
- XLA health monitoring for CPU fallback detection.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

DEFAULT_XLA_CACHE_DIR = "/var/cache/ainode/xla_compile_cache"
BUCKET_ALIGNMENT = 128


def initialize_xla_cache(
    cache_dir: str = DEFAULT_XLA_CACHE_DIR,
    readonly: bool = False,
) -> bool:
    """Initialize persistent XLA compilation cache at AINode startup."""
    try:
        import torch_xla.runtime as xr
        xr.initialize_cache(cache_dir, readonly=readonly)
        logger.info("XLA compilation cache initialized at: %s", cache_dir)
        return True
    except ImportError:
        logger.debug("torch_xla not available, skipping XLA cache init")
        return False
    except Exception as e:
        logger.warning("Failed to initialize XLA cache: %s", e)
        return False


def bucket_input_shape(
    tensor: torch.Tensor,
    alignment: int = BUCKET_ALIGNMENT,
) -> tuple[torch.Tensor, int]:
    """Pad a batched context tensor to the nearest multiple of ``alignment``."""
    batch_size, seq_len = tensor.shape
    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
    if padded_len > seq_len:
        padding = torch.zeros(
            batch_size, padded_len - seq_len,
            dtype=tensor.dtype, device=tensor.device,
        )
        tensor = torch.cat([tensor, padding], dim=-1)
    return tensor, seq_len


def tpu_inference(
    model: torch.nn.Module,
    past_values: list[torch.Tensor],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Run inference on TPU with BFloat16 autocast and explicit XLA sync."""
    import torch_xla

    if device is None:
        device = torch_xla.device()

    x = [ts.to(device=device, dtype=torch.float32) for ts in past_values]

    with torch.autocast("xla", dtype=torch.bfloat16):
        with torch.no_grad():
            outputs = model(past_values=x, return_dict=True)

    torch_xla.sync(wait=True)
    return outputs.mean_predictions.float().cpu()


def check_xla_health() -> None:
    """Log warnings if aten:: CPU-fallback ops are detected."""
    try:
        import torch_xla.debug.metrics as met
        report = met.short_metrics_report()
        for line in report.split("\n"):
            if "aten::" in line:
                logger.warning("XLA CPU fallback: %s", line)
        met.clear_all()
    except ImportError:
        pass
