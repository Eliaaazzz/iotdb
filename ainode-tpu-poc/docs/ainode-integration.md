# AINode Integration Split

This directory is the standalone PoC half of the integration.

## Standalone PoC

- Path: `ainode-tpu-poc/`
- Purpose: validate TPU backend protocol, XLA optimizations, and TimesFM
  pipeline contract independently of the full AINode runtime
- No dependency on `iotdb.ainode.core` or Thrift

## AINode Integration

- Path: `iotdb-core/ainode/`
- Branch: `ainode-tpu-poc`
- Focus: wire validated components into the AINode source tree:
  - `device/backend/base.py` -- extend `BackendType` with `TPU = "xla"`
  - `device/backend/tpu_backend.py` -- production `TPUBackend`
  - `manager/device_manager.py` -- register TPU in backend dict
  - `model/model_info.py` -- add `"timesfm"` to `BUILTIN_HF_TRANSFORMERS_MODEL_MAP`
  - `model/timesfm/pipeline_timesfm.py` -- `TimesFMPipeline(ForecastPipeline)`

## Review Order

1. Review this PoC first for backend protocol compliance, XLA optimization
   correctness, pipeline method contracts, and test coverage.
2. Review the AINode integration second for SDK wiring only.
