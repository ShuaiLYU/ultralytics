# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly v2 Segmentation predictor with external-mask injection."""

from __future__ import annotations

import torch

from ultralytics.models.yolo.segment import SegmentationPredictor

from ..anomaly_v2._util import resolve_v2_model


class AnomalyV2SegPredictor(SegmentationPredictor):
    """Segment predictor with set_external_mask_once glue. Inherits mask postprocessing
    from SegmentationPredictor; only adds the per-forward mask injection."""

    external_mask: torch.Tensor | None = None

    def preprocess(self, im):
        """Set mask state on the model based on configured external mask."""
        m = resolve_v2_model(self.model)
        if m is not None and hasattr(m, "disable_mask_once"):
            if self.external_mask is not None:
                if not hasattr(m, "set_external_mask_once"):
                    raise RuntimeError(
                        "Model does not support external_mask; rebuild from this branch."
                    )
                m.set_external_mask_once(
                    self.external_mask.to(next(m.parameters()).device)
                )
            else:
                m.disable_mask_once()
        return super().preprocess(im)
