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

"""Runnable example: end-to-end TimesFM forecast through the pipeline.

Demonstrates the preprocess -> forecast -> postprocess flow using a
mock model that produces deterministic output, validating the data
format adaptation between IoTDB tensors and TimesFM's Transformers API.

Usage::

    python examples/forecast_example.py
"""

import logging
from types import SimpleNamespace

import torch
from torch import nn

from ainode_tpu_poc.tpu_backend import DeviceManager, bucket_input_shape
from ainode_tpu_poc.timesfm import TimesFMPipeline


class MockTimesFMModel(nn.Module):
    """Minimal mock matching TimesFm2_5ModelForPrediction output contract."""

    def __init__(self, output_length: int = 96, num_quantiles: int = 9):
        super().__init__()
        self.output_length = output_length
        self.num_quantiles = num_quantiles
        # Register a dummy parameter so next(model.parameters()) works.
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, past_values, return_dict=True):
        batch_size = len(past_values)
        mean = torch.randn(batch_size, self.output_length)
        full = torch.randn(batch_size, self.num_quantiles, self.output_length)
        return SimpleNamespace(mean_predictions=mean, full_predictions=full)


def run():
    logging.basicConfig(level=logging.INFO)

    # 1. Device selection (auto-fallback)
    dm = DeviceManager()
    logging.info("Active backend: %s", dm.type.value)

    # 2. Simulate IoTDB input: shape [1, 512] (univariate, 512 time steps)
    iotdb_input = [{"targets": torch.randn(1, 512)}]

    # 3. Pipeline: preprocess -> forecast -> postprocess
    model = MockTimesFMModel(output_length=96)
    pipeline = TimesFMPipeline(model, backend_type=dm.type.value)

    preprocessed = pipeline.preprocess(iotdb_input, context_length=1024)
    logging.info(
        "Preprocessed: %d batch(es), series length %d",
        len(preprocessed),
        preprocessed[0]["past_values"][0].shape[0],
    )

    raw_output = pipeline.forecast(preprocessed)
    logging.info("Forecast point shape: %s", raw_output[0]["point"].shape)
    logging.info("Forecast quantiles shape: %s", raw_output[0]["quantiles"].shape)

    final = pipeline.postprocess(raw_output)
    logging.info("Final output shape: %s (target_count, output_length)", final[0].shape)

    # 4. Optional: demonstrate shape bucketing
    sample = torch.randn(1, 500)
    padded, orig_len = bucket_input_shape(sample)
    logging.info(
        "Shape bucketing: %d -> %d (original %d)",
        500, padded.shape[1], orig_len,
    )


if __name__ == "__main__":
    run()
