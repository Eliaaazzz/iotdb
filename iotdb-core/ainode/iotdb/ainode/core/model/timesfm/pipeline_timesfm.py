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

import torch

from iotdb.ainode.core.device.backend.base import BackendType
from iotdb.ainode.core.exception import InferenceModelInternalException
from iotdb.ainode.core.inference.pipeline.basic_pipeline import (
    BACKEND,
    ForecastPipeline,
)
from iotdb.ainode.core.log import Logger
from iotdb.ainode.core.model.model_info import ModelInfo

logger = Logger()


class TimesFMPipeline(ForecastPipeline):
    """ForecastPipeline for Google TimesFM 2.5."""

    def __init__(self, model_info: ModelInfo, **model_kwargs):
        super().__init__(model_info, **model_kwargs)

    def _preprocess(self, inputs, **infer_kwargs):
        """
        Convert IoTDB's targets tensor (shape [1, N, L]) into the
        list[dict] format expected by TimesFm2_5ModelForPrediction,
        truncating each series to the active context window and
        normalizing missing-value handling explicitly.

        Args:
            inputs: list[dict] with 'targets' key holding a torch.Tensor.
            **infer_kwargs: May include 'context_length' (default 1024).

        Returns:
            list[dict]: Each dict contains 'past_values' as a list of
                contiguous 1D tensors, one per univariate series in the batch.
        """
        context_length = infer_kwargs.get("context_length", 1024)
        model_id = self.model_info.model_id

        if inputs[0].get("past_covariates", None) or inputs[0].get(
            "future_covariates", None
        ):
            logger.warning(
                f"[Inference] Past_covariates and future_covariates will be "
                f"ignored, as they are not supported for model {model_id}."
            )

        validated = []
        for inp in inputs:
            targets = inp["targets"]  # torch.Tensor

            # Ensure 2D: (target_count, input_length)
            if targets.ndim == 1:
                targets = targets.unsqueeze(0)

            if targets.shape[0] != 1:
                raise InferenceModelInternalException(
                    f"Model {model_id} only supports univariate forecast, "
                    f"but receives {targets.shape[0]} target variables."
                )

            # Core scope keeps preprocessing explicit in the wrapper.
            # Replace NaN with 0.0 and truncate to context window
            targets = torch.nan_to_num(targets, nan=0.0)
            targets = targets[:, -context_length:]

            # Convert each row to a contiguous 1D tensor for past_values
            validated.append(
                {
                    "past_values": [
                        targets[i].contiguous() for i in range(targets.shape[0])
                    ],
                }
            )
        return validated

    def forecast(self, inputs, **infer_kwargs):
        """
        Run model inference via the upstream TimesFm2_5ModelForPrediction
        forward path: model(past_values=..., return_dict=True).

        On TPU, inserts torch_xla.sync(wait=True) after inference to
        materialize results before host-side access.

        Args:
            inputs: list[dict] from preprocess, each with 'past_values'.
            **infer_kwargs: May include 'output_length' (default 96).

        Returns:
            list[dict]: Each dict has 'point' (mean_predictions) and
                'quantiles' (full_predictions) as CPU float tensors.
        """
        device = next(self.model.parameters()).device
        results = []

        for inp in inputs:
            past_values = [
                series.to(device=device, dtype=torch.float32)
                for series in inp["past_values"]
            ]

            with torch.no_grad():
                outputs = self.model(
                    past_values=past_values,
                    return_dict=True,
                )

            if BACKEND.type == BackendType.TPU:
                try:
                    import torch_xla

                    torch_xla.sync(wait=True)
                except ImportError:
                    pass

            results.append(
                {
                    "point": outputs.mean_predictions.float().cpu(),
                    "quantiles": outputs.full_predictions.float().cpu(),
                }
            )
        return results

    def _postprocess(self, outputs, **infer_kwargs) -> list[torch.Tensor]:
        """
        Current SQL path returns point forecasts.
        Extracts mean_predictions as the core point-forecast output.

        Args:
            outputs: list[dict] from forecast(), each with 'point' key.
            **infer_kwargs: Additional keyword arguments.

        Returns:
            list[torch.Tensor]: Each tensor has shape [1, output_length]
                matching the ForecastPipeline contract.
        """
        processed = []
        for out in outputs:
            point = out["point"]
            # Ensure 2D shape [target_count=1, output_length]
            if point.ndim == 1:
                point = point.unsqueeze(0)
            elif point.ndim > 2:
                # Take mean across extra dimensions if needed
                while point.ndim > 2:
                    point = point.mean(dim=0)
            processed.append(point)
        return processed
