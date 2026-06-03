# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly v2 Segmentation trainer.

Thin extension of ``SegmentationTrainer``. The differences:
  * ``get_model`` returns a ``YOLOAnomalyV2SegModel`` instead of a plain ``SegmentationModel``.
  * ``get_validator`` returns an ``AnomalyV2SegValidator`` that runs val twice
    (mask-on and mask-off).

Everything else (dataset, dataloader, augmentation, loss aggregation, plot,
auto_batch, etc.) is inherited from ``SegmentationTrainer`` unchanged.
"""

from __future__ import annotations

from copy import copy

from ultralytics.models import yolo
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.nn.tasks import YOLOAnomalyV2SegModel
from ultralytics.utils import DEFAULT_CFG, RANK


class AnomalyV2SegTrainer(SegmentationTrainer):
    """Trainer for the anomaly_v2_seg task. Mirrors SegmentationTrainer but yields
    a YOLOAnomalyV2SegModel."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = YOLOAnomalyV2SegModel(
            cfg, nc=self.data["nc"], verbose=verbose and RANK == -1
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "seg_loss"
        return yolo.anomaly_v2_seg.AnomalyV2SegValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
