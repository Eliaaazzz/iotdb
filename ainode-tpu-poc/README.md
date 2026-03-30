# IoTDB-AINode TPU Backend & TimesFM 2.5 PoC

This directory contains a standalone proof of concept for two additions to
IoTDB-AINode:

- **TPU BackendAdapter** implementing the `BackendAdapter` protocol via
  `torch_xla`, with automatic TPU -> CUDA -> CPU fallback.
- **TimesFMPipeline** integrating Google TimesFM 2.5 into the
  `ForecastPipeline` hierarchy using the upstream
  `transformers.TimesFm2_5ModelForPrediction` checkpoint.

The work is described in the GSoC 2026 proposal *"TPU Compatibility &
Integration of State-of-the-Art Time Series Foundation Models for
IoTDB-AINode"*. It is meant for design validation and review, not
production use.

## Quick Start

```bash
cd ainode-tpu-poc

uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

pytest tests/ -v
```

## Usage

### TPU Backend

```python
from ainode_tpu_poc.tpu_backend import DeviceManager, BackendType

dm = DeviceManager()
print(dm.type)  # BackendType.TPU / CUDA / CPU
```

### TimesFM Pipeline

```python
import torch
from ainode_tpu_poc.timesfm import TimesFMPipeline

model = load_timesfm_model()  # upstream Transformers checkpoint
pipeline = TimesFMPipeline(model, backend_type="cpu")

inputs = [{"targets": torch.randn(1, 512)}]
preprocessed = pipeline.preprocess(inputs, context_length=1024)
raw_output  = pipeline.forecast(preprocessed)
final        = pipeline.postprocess(raw_output)
# final[0].shape == (1, 96)
```

Runnable example: [`examples/forecast_example.py`](examples/forecast_example.py)

## Design Notes

### TPU Backend (Proposal Section 4.2)

- `TPUBackend.is_available()` lazily imports `torch_xla` and checks
  `xm.xla_device_hw()` for `"TPU"`.
- `BackendType` enum ordering (`TPU`, `CUDA`, `CPU`) drives priority;
  `DeviceManager` iterates and selects the first available.
- `set_device()` stores the ordinal for bookkeeping; actual TPU chip
  assignment is process-level via `PJRT_DEVICE`.

### XLA Optimizations (Proposal Section 4.2.4)

- **Compilation caching**: `initialize_xla_cache()` wraps
  `xr.initialize_cache()`.
- **Shape bucketing**: `bucket_input_shape()` pads to multiples of 128.
- **BFloat16 autocast**: `tpu_inference()` uses
  `torch.autocast("xla", dtype=torch.bfloat16)`.
- **Health monitoring**: `check_xla_health()` detects `aten::` fallback ops.

### TimesFM Pipeline (Proposal Section 4.5 / Appendix B)

- `preprocess()`: converts IoTDB `[1, target_count, seq_len]` tensors
  into `past_values` lists, handles NaN, truncates to context window.
- `forecast()`: calls upstream `model(past_values=..., return_dict=True)`,
  syncs XLA on TPU, returns CPU float32 tensors.
- `postprocess()`: extracts `mean_predictions` as the point forecast.

### AINode Mapping

| PoC Module | AINode Integration Target |
|---|---|
| `tpu_backend.backend.BackendType` | `device/backend/base.py` |
| `tpu_backend.backend.TPUBackend` | `device/backend/tpu_backend.py` |
| `tpu_backend.backend.DeviceManager` | `manager/device_manager.py` |
| `tpu_backend.xla_utils` | `device/backend/xla_utils.py` |
| `timesfm.pipeline.TimesFMPipeline` | `model/timesfm/pipeline_timesfm.py` |

## Project Layout

```text
ainode-tpu-poc/
|-- .gitignore
|-- README.md
|-- pyproject.toml
|-- docs/
|   `-- ainode-integration.md
|-- examples/
|   `-- forecast_example.py
|-- src/
|   `-- ainode_tpu_poc/
|       |-- __init__.py
|       |-- tpu_backend/
|       |   |-- __init__.py
|       |   |-- backend.py
|       |   `-- xla_utils.py
|       `-- timesfm/
|           |-- __init__.py
|           `-- pipeline.py
`-- tests/
    |-- __init__.py
    |-- test_tpu_backend.py
    `-- test_timesfm_pipeline.py
```

## Relationship to AINode Integration

This directory stays standalone. AINode-side wiring (modifying
`base.py`, `device_manager.py`, `model_info.py`, etc.) lives in the
main source tree on the `ainode-tpu-poc` branch. See
[`docs/ainode-integration.md`](docs/ainode-integration.md) for the
split rationale.

## Limitations

- Mock model only (no real TimesFM checkpoint download).
- In-memory state; no Thrift RPC or SQL integration.
- XLA optimizations validated structurally, not on physical TPU.

## License

Apache-2.0
