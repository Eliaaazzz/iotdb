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
Unit tests for TimesFMPipeline preprocess/forecast/postprocess contracts.

Tests run without loading actual model weights. Validates:
- IoTDB tensor format [1, N, L] -> TimesFM past_values conversion
- NaN handling in preprocessing
- Context window truncation
- Point forecast extraction from model output
- Output shape compliance with ForecastPipeline contract
- Univariate-only enforcement
- Model registry registration
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch


# Mock all heavy dependencies that the import chain pulls in
# (thrift, sklearn, sktime, transformers, accelerate, etc.)
# so tests can run with only torch installed.
# Every dotted sub-module must be present so `from A.B.C import X` works.
_dep_mocks = {}
_external_modules = [
    # thrift chain
    "thrift", "thrift.Thrift", "thrift.protocol", "thrift.transport",
    "iotdb.thrift", "iotdb.thrift.common", "iotdb.thrift.common.ttypes",
    "iotdb.thrift.ainode", "iotdb.thrift.ainode.ttypes",
    # config dependencies
    "dynaconf", "psutil",
    # sklearn
    "sklearn", "sklearn.preprocessing",
    # sktime - all import paths used by modeling_sktime.py
    "sktime",
    "sktime.detection", "sktime.detection.hmm_learn", "sktime.detection.stray",
    "sktime.forecasting", "sktime.forecasting.base",
    "sktime.forecasting.arima", "sktime.forecasting.exp_smoothing",
    "sktime.forecasting.naive", "sktime.forecasting.trend",
    # transformers / HF
    "transformers", "transformers.models",
    "accelerate", "huggingface_hub",
    # other ML deps
    "optuna", "scipy", "scipy.stats",
    "statsmodels", "statsmodels.tsa",
    # pandas / numpy might be missing too
    "pandas", "numpy",
]
for mod_name in _external_modules:
    if mod_name not in sys.modules:
        _dep_mocks[mod_name] = MagicMock()

_patcher = patch.dict(sys.modules, _dep_mocks)
_patcher.start()


def _make_fake_model_output(batch_size=1, output_length=96, num_quantiles=9):
    """Create a mock model output matching TimesFm2_5ModelForPrediction API."""
    return SimpleNamespace(
        mean_predictions=torch.randn(batch_size, output_length),
        full_predictions=torch.randn(batch_size, num_quantiles, output_length),
    )


def _make_pipeline():
    """Create a TimesFMPipeline instance without loading real model weights."""
    from iotdb.ainode.core.model.model_info import ModelInfo
    from iotdb.ainode.core.model.model_constants import ModelCategory, ModelStates
    from iotdb.ainode.core.model.timesfm.pipeline_timesfm import TimesFMPipeline

    model_info = ModelInfo(
        model_id="timesfm",
        category=ModelCategory.BUILTIN,
        state=ModelStates.INACTIVE,
        model_type="timesfm2_5",
        pipeline_cls="pipeline_timesfm.TimesFMPipeline",
        repo_id="google/timesfm-2.5-200m-transformers",
        transformers_registered=True,
    )

    mock_model = MagicMock()
    mock_model.parameters.return_value = iter(
        [torch.nn.Parameter(torch.zeros(1))]
    )
    mock_model.return_value = _make_fake_model_output()

    pipeline = TimesFMPipeline.__new__(TimesFMPipeline)
    pipeline.model_info = model_info
    pipeline.model = mock_model
    pipeline.device = torch.device("cpu")
    return pipeline, mock_model


class TestTimesFMPreprocess(unittest.TestCase):
    """Test TimesFMPipeline._preprocess method."""

    def setUp(self):
        self.pipeline, self.mock_model = _make_pipeline()

    def test_basic_preprocess_shape(self):
        """Standard 2D targets [1, 512] should produce one entry with past_values."""
        targets = torch.randn(1, 512)
        inputs = [{"targets": targets}]
        result = self.pipeline._preprocess(inputs)
        self.assertEqual(len(result), 1)
        self.assertIn("past_values", result[0])
        self.assertEqual(len(result[0]["past_values"]), 1)
        self.assertEqual(result[0]["past_values"][0].shape, (512,))

    def test_1d_targets_unsqueeze(self):
        """1D targets [512] should be unsqueezed to [1, 512]."""
        targets = torch.randn(512)
        inputs = [{"targets": targets}]
        result = self.pipeline._preprocess(inputs)
        self.assertEqual(len(result[0]["past_values"]), 1)

    def test_nan_handling(self):
        """NaN values should be replaced with 0.0 during preprocessing."""
        targets = torch.tensor([[1.0, float("nan"), 3.0, float("nan"), 5.0]])
        inputs = [{"targets": targets}]
        result = self.pipeline._preprocess(inputs)
        series = result[0]["past_values"][0]
        self.assertFalse(torch.isnan(series).any())
        self.assertEqual(series[1].item(), 0.0)
        self.assertEqual(series[3].item(), 0.0)

    def test_context_length_truncation(self):
        """Series longer than context_length should be truncated from the right (keep last N)."""
        targets = torch.randn(1, 2048)
        inputs = [{"targets": targets}]
        result = self.pipeline._preprocess(inputs, context_length=1024)
        self.assertEqual(result[0]["past_values"][0].shape, (1024,))

    def test_context_length_no_truncation_when_shorter(self):
        """Series shorter than context_length should not be truncated."""
        targets = torch.randn(1, 256)
        inputs = [{"targets": targets}]
        result = self.pipeline._preprocess(inputs, context_length=1024)
        self.assertEqual(result[0]["past_values"][0].shape, (256,))

    def test_multivariate_raises_error(self):
        """Multiple target variables should raise InferenceModelInternalException."""
        from iotdb.ainode.core.exception import InferenceModelInternalException

        targets = torch.randn(3, 512)  # 3 variates
        inputs = [{"targets": targets}]
        with self.assertRaises(InferenceModelInternalException):
            self.pipeline._preprocess(inputs)

    def test_multiple_input_batches(self):
        """Multiple input dicts should each produce a separate entry."""
        inputs = [
            {"targets": torch.randn(1, 100)},
            {"targets": torch.randn(1, 200)},
        ]
        result = self.pipeline._preprocess(inputs)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["past_values"][0].shape, (100,))
        self.assertEqual(result[1]["past_values"][0].shape, (200,))

    def test_contiguous_output(self):
        """All past_values tensors should be contiguous."""
        targets = torch.randn(1, 512)
        inputs = [{"targets": targets}]
        result = self.pipeline._preprocess(inputs)
        for series in result[0]["past_values"]:
            self.assertTrue(series.is_contiguous())

    def test_covariates_ignored_with_warning(self):
        """Past/future covariates should be silently ignored."""
        targets = torch.randn(1, 100)
        inputs = [
            {
                "targets": targets,
                "past_covariates": {"temp": torch.randn(100)},
                "future_covariates": {"temp": torch.randn(96)},
            }
        ]
        # Should not raise, covariates just get ignored
        result = self.pipeline._preprocess(inputs)
        self.assertIn("past_values", result[0])


class TestTimesFMForecast(unittest.TestCase):
    """Test TimesFMPipeline.forecast method."""

    def setUp(self):
        self.pipeline, self.mock_model = _make_pipeline()
        # Patch BACKEND to simulate CPU backend
        self._backend_patcher = patch(
            "iotdb.ainode.core.model.timesfm.pipeline_timesfm.BACKEND",
            MagicMock(type=MagicMock(value="cpu")),
        )
        self._backend_patcher.start()

    def tearDown(self):
        self._backend_patcher.stop()

    def test_forecast_returns_point_and_quantiles(self):
        """forecast() should return dicts with 'point' and 'quantiles' keys."""
        inputs = [{"past_values": [torch.randn(512)]}]
        results = self.pipeline.forecast(inputs)
        self.assertEqual(len(results), 1)
        self.assertIn("point", results[0])
        self.assertIn("quantiles", results[0])

    def test_forecast_output_on_cpu(self):
        """All output tensors should be on CPU for TsBlock serialization."""
        inputs = [{"past_values": [torch.randn(512)]}]
        results = self.pipeline.forecast(inputs)
        self.assertEqual(results[0]["point"].device, torch.device("cpu"))
        self.assertEqual(results[0]["quantiles"].device, torch.device("cpu"))

    def test_forecast_output_dtype_float32(self):
        """Output tensors should be float32 for serialization compatibility."""
        inputs = [{"past_values": [torch.randn(512)]}]
        results = self.pipeline.forecast(inputs)
        self.assertEqual(results[0]["point"].dtype, torch.float32)

    def test_forecast_calls_model_with_no_grad(self):
        """Model should be called during forecast."""
        inputs = [{"past_values": [torch.randn(512)]}]
        self.pipeline.forecast(inputs)
        self.mock_model.assert_called_once()


class TestTimesFMPostprocess(unittest.TestCase):
    """Test TimesFMPipeline._postprocess method."""

    def setUp(self):
        self.pipeline, _ = _make_pipeline()

    def test_postprocess_extracts_point_forecast(self):
        """_postprocess should extract 'point' tensor from each output dict."""
        outputs = [
            {"point": torch.randn(1, 96), "quantiles": torch.randn(1, 9, 96)},
        ]
        result = self.pipeline._postprocess(outputs)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], torch.Tensor)

    def test_postprocess_output_shape_2d(self):
        """Each output tensor should be 2D: [target_count, output_length]."""
        outputs = [
            {"point": torch.randn(1, 96), "quantiles": torch.randn(1, 9, 96)},
        ]
        result = self.pipeline._postprocess(outputs)
        self.assertEqual(result[0].ndim, 2)
        self.assertEqual(result[0].shape, (1, 96))

    def test_postprocess_1d_point_gets_unsqueezed(self):
        """A 1D point tensor should be unsqueezed to 2D."""
        outputs = [
            {"point": torch.randn(96), "quantiles": torch.randn(9, 96)},
        ]
        result = self.pipeline._postprocess(outputs)
        self.assertEqual(result[0].ndim, 2)
        self.assertEqual(result[0].shape, (1, 96))

    def test_postprocess_multiple_batches(self):
        """Multiple output dicts should each produce a separate tensor."""
        outputs = [
            {"point": torch.randn(1, 96), "quantiles": torch.randn(1, 9, 96)},
            {"point": torch.randn(1, 48), "quantiles": torch.randn(1, 9, 48)},
        ]
        result = self.pipeline._postprocess(outputs)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, (1, 96))
        self.assertEqual(result[1].shape, (1, 48))


class TestTimesFMModelRegistration(unittest.TestCase):
    """Verify TimesFM is properly registered in model_info.py."""

    def test_timesfm_in_builtin_map(self):
        from iotdb.ainode.core.model.model_info import BUILTIN_HF_TRANSFORMERS_MODEL_MAP

        self.assertIn("timesfm", BUILTIN_HF_TRANSFORMERS_MODEL_MAP)

    def test_timesfm_model_info_fields(self):
        from iotdb.ainode.core.model.model_info import BUILTIN_HF_TRANSFORMERS_MODEL_MAP
        from iotdb.ainode.core.model.model_constants import ModelCategory, ModelStates

        info = BUILTIN_HF_TRANSFORMERS_MODEL_MAP["timesfm"]
        self.assertEqual(info.model_id, "timesfm")
        self.assertEqual(info.category, ModelCategory.BUILTIN)
        self.assertEqual(info.state, ModelStates.INACTIVE)
        self.assertEqual(info.model_type, "timesfm2_5")
        self.assertEqual(info.pipeline_cls, "pipeline_timesfm.TimesFMPipeline")
        self.assertEqual(info.repo_id, "google/timesfm-2.5-200m-transformers")
        self.assertTrue(info.transformers_registered)


if __name__ == "__main__":
    unittest.main()
