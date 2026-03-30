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

These optimizations reduce XLA graph recompilation overhead on TPU:
- Persistent compilation cache: reuse compiled graphs across AINode restarts.
- Shape bucketing: pad input tensors to standard bucket sizes (multiples of
  128) so that variable-length inputs map to a small set of compiled graphs.
- XLA health monitoring: detect CPU fallback operations (aten:: ops).
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Default cache directory for XLA compilation artifacts
DEFAULT_XLA_CACHE_DIR = "/var/cache/ainode/xla_compile_cache"

# Bucket size alignment for TPU systolic array width
BUCKET_ALIGNMENT = 128


def initialize_xla_cache(
    cache_dir: str = DEFAULT_XLA_CACHE_DIR,
    readonly: bool = False,
) -> bool:
    """
    Initialize persistent XLA compilation cache at AINode startup.

    Must be called before any XLA computation is executed so that
    repeated requests with identical graph shapes can reuse compiled
    artifacts.

    Args:
        cache_dir: Directory path for storing compiled XLA graphs.
        readonly: If True, only read from cache without writing new entries.

    Returns:
        True if cache was initialized successfully, False otherwise.
    """
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
    """
    Pad a batched context tensor to standard bucket sizes to minimize
    XLA graph recompilation caused by variable input shapes.

    TPU performance is optimal when tensor dimensions are multiples of
    128 (matching the systolic array width).

    Args:
        tensor: Input tensor of shape (batch_size, seq_len).
        alignment: Pad seq_len to nearest multiple of this value.

    Returns:
        Tuple of (padded_tensor, original_seq_len) so the caller can
        trim predictions back to the original length.
    """
    batch_size, seq_len = tensor.shape
    # Round up to nearest multiple of alignment
    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
    if padded_len > seq_len:
        padding = torch.zeros(
            batch_size,
            padded_len - seq_len,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.cat([tensor, padding], dim=-1)
    return tensor, seq_len


def tpu_inference(
    model: torch.nn.Module,
    past_values: list[torch.Tensor],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Run inference on TPU with XLA optimizations.

    Includes BFloat16 autocast for TPU-native mixed precision and
    explicit XLA sync to materialize lazy evaluation results.

    Args:
        model: The TimesFM model instance.
        past_values: List of 1D tensors, one per time series.
        device: XLA device; defaults to torch_xla.device().

    Returns:
        Mean predictions tensor moved back to CPU for TsBlock serialization.
    """
    import torch_xla

    if device is None:
        device = torch_xla.device()

    # Move per-series histories to TPU
    x = [ts.to(device=device, dtype=torch.float32) for ts in past_values]

    # Apply BFloat16 for TPU-native mixed precision
    # No gradient scaler needed (unlike GPU float16)
    with torch.autocast("xla", dtype=torch.bfloat16):
        with torch.no_grad():
            outputs = model(past_values=x, return_dict=True)

    # Explicit sync to launch and wait for pending XLA work
    torch_xla.sync(wait=True)

    # Move results back to CPU for TsBlock serialization
    return outputs.mean_predictions.float().cpu()


def check_xla_health() -> None:
    """
    Monitor for CPU fallback operations (aten:: ops).

    Any aten:: counters indicate operations that XLA couldn't lower to
    HLO, causing device transfers. Logs warnings if detected.
    """
    try:
        import torch_xla.debug.metrics as met

        report = met.short_metrics_report()
        # Log warning if aten:: ops detected
        for line in report.split("\n"):
            if "aten::" in line:
                logger.warning("XLA CPU fallback: %s", line)
        met.clear_all()
    except ImportError:
        pass
