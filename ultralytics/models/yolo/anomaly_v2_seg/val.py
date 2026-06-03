# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly v2 Segmentation validator — runs val twice (mask-on / mask-off)."""

from __future__ import annotations

from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils import LOGGER

from ..anomaly_v2._util import resolve_v2_model


class AnomalyV2SegValidator(SegmentationValidator):
    """Segmentation validator that evaluates the v2 seg model in both mask-on and mask-off modes."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "anomaly_v2_seg"
        self._mask_mode = "on"
        self._model_ref = None

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def init_metrics(self, model) -> None:
        super().init_metrics(model)
        self._model_ref = model

    def preprocess(self, batch):
        """Set v2 seg mask state on the model based on current ``_mask_mode``."""
        batch = super().preprocess(batch)
        model = resolve_v2_model(self._model_ref)
        if model is None or not hasattr(model, "set_mask_prior"):
            return batch
        if self._mask_mode == "on":
            prior = model._build_mask_prior(batch)
            model.set_mask_prior(prior)
        else:
            model.disable_mask_once()
        return batch

    # ------------------------------------------------------------------
    # Two-pass __call__
    # ------------------------------------------------------------------
    def __call__(self, trainer=None, model=None):
        """Run validation twice (mask-on and mask-off), merge metric dicts."""
        # Pass 1: mask-on
        self._mask_mode = "on"
        stats_on = super().__call__(trainer=trainer, model=model)

        # Pass 2: mask-off
        self._mask_mode = "off"
        try:
            stats_off = super().__call__(trainer=trainer, model=model)
        except Exception as e:
            LOGGER.warning(f"AnomalyV2SegValidator: mask-off pass failed: {e}; falling back to mask-on only.")
            stats_off = {}

        self._mask_mode = "on"

        if not isinstance(stats_on, dict):
            return stats_on
        merged = dict(stats_on)
        if isinstance(stats_off, dict):
            for k, v in stats_off.items():
                merged[f"mask_off/{k}"] = v
        return merged
