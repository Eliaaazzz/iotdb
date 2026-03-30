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
Unit tests for TimesFMPipeline preprocess / forecast / postprocess.

Coverage:
- IoTDB tensor format [1, L] -> TimesFM past_values conversion
- NaN handling in preprocessing
- Context window truncation
- Univariate-only enforcement
- Point forecast extraction and shape compliance
- Multiple batch inputs
- XLA shape bucketing
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from ainode_tpu_poc.timesfm.pipeline import TimesFMPipeline
from ainode_tpu_poc.tpu_backend.xla_utils import bucket_input_shape


def _mock_model(output_length=96, num_quantiles=9):
    model = MagicMock()
    model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
    model.return_value = SimpleNamespace(
        mean_predictions=torch.randn(1, output_length),
        full_predictions=torch.randn(1, num_quantiles, output_length),
    )
    return model


class PreprocessTest(unittest.TestCase):

    def setUp(self):
        self.pipeline = TimesFMPipeline(_mock_model())

    def test_basic_2d_shape(self):
        result = self.pipeline.preprocess([{"targets": torch.randn(1, 512)}])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["past_values"][0].shape, (512,))

    def test_1d_unsqueeze(self):
        result = self.pipeline.preprocess([{"targets": torch.randn(512)}])
        self.assertEqual(len(result[0]["past_values"]), 1)

    def test_nan_replaced_with_zero(self):
        t = torch.tensor([[1.0, float("nan"), 3.0, float("nan"), 5.0]])
        result = self.pipeline.preprocess([{"targets": t}])
        series = result[0]["past_values"][0]
        self.assertFalse(torch.isnan(series).any())
        self.assertEqual(series[1].item(), 0.0)
        self.assertEqual(series[3].item(), 0.0)

    def test_context_length_truncation(self):
        result = self.pipeline.preprocess(
            [{"targets": torch.randn(1, 2048)}], context_length=1024
        )
        self.assertEqual(result[0]["past_values"][0].shape, (1024,))

    def test_no_truncation_when_shorter(self):
        result = self.pipeline.preprocess(
            [{"targets": torch.randn(1, 256)}], context_length=1024
        )
        self.assertEqual(result[0]["past_values"][0].shape, (256,))

    def test_multivariate_raises(self):
        with self.assertRaises(ValueError):
            self.pipeline.preprocess([{"targets": torch.randn(3, 512)}])

    def test_multiple_batches(self):
        result = self.pipeline.preprocess([
            {"targets": torch.randn(1, 100)},
            {"targets": torch.randn(1, 200)},
        ])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["past_values"][0].shape, (100,))
        self.assertEqual(result[1]["past_values"][0].shape, (200,))

    def test_contiguous(self):
        result = self.pipeline.preprocess([{"targets": torch.randn(1, 512)}])
        self.assertTrue(result[0]["past_values"][0].is_contiguous())


class ForecastTest(unittest.TestCase):

    def setUp(self):
        self.model = _mock_model()
        self.pipeline = TimesFMPipeline(self.model)

    def test_returns_point_and_quantiles(self):
        results = self.pipeline.forecast([{"past_values": [torch.randn(512)]}])
        self.assertIn("point", results[0])
        self.assertIn("quantiles", results[0])

    def test_output_on_cpu(self):
        results = self.pipeline.forecast([{"past_values": [torch.randn(512)]}])
        self.assertEqual(results[0]["point"].device, torch.device("cpu"))

    def test_output_dtype_float32(self):
        results = self.pipeline.forecast([{"past_values": [torch.randn(512)]}])
        self.assertEqual(results[0]["point"].dtype, torch.float32)

    def test_model_called(self):
        self.pipeline.forecast([{"past_values": [torch.randn(512)]}])
        self.model.assert_called_once()


class PostprocessTest(unittest.TestCase):

    def setUp(self):
        self.pipeline = TimesFMPipeline(_mock_model())

    def test_extracts_point(self):
        outputs = [{"point": torch.randn(1, 96), "quantiles": torch.randn(1, 9, 96)}]
        result = self.pipeline.postprocess(outputs)
        self.assertIsInstance(result[0], torch.Tensor)

    def test_output_2d(self):
        outputs = [{"point": torch.randn(1, 96), "quantiles": torch.randn(1, 9, 96)}]
        result = self.pipeline.postprocess(outputs)
        self.assertEqual(result[0].ndim, 2)
        self.assertEqual(result[0].shape, (1, 96))

    def test_1d_point_unsqueezed(self):
        outputs = [{"point": torch.randn(96), "quantiles": torch.randn(9, 96)}]
        result = self.pipeline.postprocess(outputs)
        self.assertEqual(result[0].shape, (1, 96))

    def test_multiple_batches(self):
        outputs = [
            {"point": torch.randn(1, 96), "quantiles": torch.randn(1, 9, 96)},
            {"point": torch.randn(1, 48), "quantiles": torch.randn(1, 9, 48)},
        ]
        result = self.pipeline.postprocess(outputs)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, (1, 96))
        self.assertEqual(result[1].shape, (1, 48))


class ShapeBucketingTest(unittest.TestCase):

    def test_aligned_no_pad(self):
        t, orig = bucket_input_shape(torch.randn(4, 128))
        self.assertEqual(t.shape, (4, 128))
        self.assertEqual(orig, 128)

    def test_unaligned_pads(self):
        t, orig = bucket_input_shape(torch.randn(4, 100))
        self.assertEqual(t.shape, (4, 128))
        self.assertEqual(orig, 100)

    def test_preserves_data(self):
        src = torch.ones(2, 100)
        t, _ = bucket_input_shape(src)
        self.assertTrue(torch.allclose(t[:, :100], src))
        self.assertTrue(torch.allclose(t[:, 100:], torch.zeros(2, 28)))

    def test_small_to_128(self):
        t, orig = bucket_input_shape(torch.randn(1, 10))
        self.assertEqual(t.shape, (1, 128))

    def test_dtype_preserved(self):
        for dt in [torch.float32, torch.float64, torch.bfloat16]:
            t, _ = bucket_input_shape(torch.randn(1, 100).to(dt))
            self.assertEqual(t.dtype, dt)


if __name__ == "__main__":
    unittest.main()
