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
Standalone TimesFMPipeline validating the ForecastPipeline contract
from proposal Section 4.5 / Appendix B.

This PoC version is decoupled from the AINode runtime so it can be
tested with only ``torch`` installed. The three pipeline methods
(``preprocess``, ``forecast``, ``postprocess``) follow the same
contract as AINode's ``ForecastPipeline`` hierarchy.
"""

import logging

import torch

logger = logging.getLogger(__name__)


class TimesFMPipeline:
    """ForecastPipeline for Google TimesFM 2.5.

    Mirrors the AINode integration target at
    ``model/timesfm/pipeline_timesfm.py`` but without importing
    the full AINode runtime.

    Attributes:
        model: A ``TimesFm2_5ModelForPrediction`` instance (or mock).
        backend_type: ``"xla"`` | ``"cuda"`` | ``"cpu"``.
    """

    def __init__(self, model, backend_type: str = "cpu"):
        self.model = model
        self.backend_type = backend_type

    # ========================= Preprocess =========================

    def preprocess(
        self,
        inputs: list[dict],
        *,
        context_length: int = 1024,
    ) -> list[dict]:
        """Convert IoTDB targets tensors into TimesFM past_values.

        Args:
            inputs: list[dict] where each dict has a ``"targets"`` key
                holding a torch.Tensor of shape ``(input_length,)`` or
                ``(target_count, input_length)``.
            context_length: Maximum number of time steps to retain
                (truncates from the left).

        Returns:
            list[dict] where each dict has ``"past_values"`` as a list
            of contiguous 1-D tensors.

        Raises:
            ValueError: If target_count != 1 (univariate only).
        """
        validated = []
        for inp in inputs:
            targets = inp["targets"]

            # Ensure 2D: (target_count, input_length)
            if targets.ndim == 1:
                targets = targets.unsqueeze(0)

            if targets.shape[0] != 1:
                raise ValueError(
                    f"TimesFM only supports univariate forecast, "
                    f"but received {targets.shape[0]} target variables."
                )

            targets = torch.nan_to_num(targets, nan=0.0)
            targets = targets[:, -context_length:]

            validated.append({
                "past_values": [
                    targets[i].contiguous() for i in range(targets.shape[0])
                ],
            })
        return validated

    # ========================== Forecast ==========================

    def forecast(self, inputs: list[dict]) -> list[dict]:
        """Run model inference via upstream forward path.

        On TPU, inserts ``torch_xla.sync(wait=True)`` after inference
        to materialize lazy evaluation results.

        Args:
            inputs: list[dict] from ``preprocess``, each with
                ``"past_values"``.

        Returns:
            list[dict] with ``"point"`` (mean predictions) and
            ``"quantiles"`` (full predictions) as CPU float32 tensors.
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

            if self.backend_type == "xla":
                try:
                    import torch_xla
                    torch_xla.sync(wait=True)
                except ImportError:
                    pass

            results.append({
                "point": outputs.mean_predictions.float().cpu(),
                "quantiles": outputs.full_predictions.float().cpu(),
            })
        return results

    # ========================= Postprocess ========================

    def postprocess(self, outputs: list[dict]) -> list[torch.Tensor]:
        """Extract point forecasts for the current SQL path.

        Returns:
            list[torch.Tensor] each of shape ``(1, output_length)``,
            matching AINode's ``ForecastPipeline`` postprocess contract.
        """
        processed = []
        for out in outputs:
            point = out["point"]
            if point.ndim == 1:
                point = point.unsqueeze(0)
            elif point.ndim > 2:
                while point.ndim > 2:
                    point = point.mean(dim=0)
            processed.append(point)
        return processed
