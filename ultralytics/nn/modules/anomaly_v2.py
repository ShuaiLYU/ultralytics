# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLOA v2 modules: bbox-mask renderer, heatmap fusion, and memory-bank scorer.

Soft-hint fusion: a 1-channel mask is turned into a bounded per-pixel bias added
(broadcast over channels) to PAN features before the Detect head. PAN feature
addition keeps the Detect head unmodified, lets reg and cls both see the bias
(empirical question — see spec §2), and is bounded vs the previous multiplicative
amplifier that forced detections.

See docs_yoloa_v2/specs/2026-06-02-softhint-fusion-design.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "BboxMaskRenderer",
    "HeatmapBiasFusion",
    "BackboneMemoryBank",
)


class BboxMaskRenderer(nn.Module):
    """Render normalized YOLO-format bboxes into a 1xHxW mask.

    Two modes:
      - "rect":  hard rectangle (inside bbox = 1, outside = 0)
      - "gauss": per-bbox 2D Gaussian centered at bbox center,
                 sigma_x = w * sigma_factor, sigma_y = h * sigma_factor;
                 multiple bboxes in the same image are combined with max.

    ``sigma_factor`` may be a scalar (fixed width) or a ``[lo, hi]`` range. With a range, each
    bbox draws its own factor ~ U(lo, hi) while training (prior-width augmentation toward the
    diffuse inference heatmap) and uses the midpoint deterministically at eval.

    Output spatial size is fixed at construction (default 80 to match P3).
    """

    def __init__(self, mask_size: int = 80, mode: str = "rect", sigma_factor: float | list = 0.25):
        super().__init__()
        assert mode in ("rect", "gauss"), f"mode must be 'rect' or 'gauss', got {mode!r}"
        self.mask_size = int(mask_size)
        self.mode = mode
        if isinstance(sigma_factor, (list, tuple)):
            self.sigma_lo, self.sigma_hi = float(sigma_factor[0]), float(sigma_factor[1])
        else:
            self.sigma_lo = self.sigma_hi = float(sigma_factor)
        self.sigma_factor = self.sigma_lo  # back-compat scalar handle (== lo)
        ys, xs = torch.meshgrid(
            torch.arange(self.mask_size, dtype=torch.float32),
            torch.arange(self.mask_size, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("grid_x", xs + 0.5, persistent=False)
        self.register_buffer("grid_y", ys + 0.5, persistent=False)

    def __setstate__(self, state):
        """Backfill sigma_lo/sigma_hi for checkpoints pickled before the [lo, hi] range knob."""
        super().__setstate__(state)
        if not hasattr(self, "sigma_lo"):
            sf = float(getattr(self, "sigma_factor", 0.25))
            self.sigma_lo = self.sigma_hi = sf

    def forward(self, bboxes: torch.Tensor, batch_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
        H = self.mask_size
        device = self.grid_x.device
        dtype = self.grid_x.dtype
        mask = torch.zeros(batch_size, 1, H, H, device=device, dtype=dtype)
        if bboxes.numel() == 0:
            return mask

        bboxes = bboxes.to(device=device, dtype=dtype)
        batch_idx = batch_idx.to(device=device, dtype=torch.long)

        cx = bboxes[:, 0] * H
        cy = bboxes[:, 1] * H
        w = bboxes[:, 2] * H
        h = bboxes[:, 3] * H

        if self.mode == "rect":
            x1 = (cx - w / 2)[:, None, None]
            x2 = (cx + w / 2)[:, None, None]
            y1 = (cy - h / 2)[:, None, None]
            y2 = (cy + h / 2)[:, None, None]
            inside = (
                (self.grid_x[None] >= x1)
                & (self.grid_x[None] < x2)
                & (self.grid_y[None] >= y1)
                & (self.grid_y[None] < y2)
            ).to(dtype)
        else:  # gauss
            # Per-bbox sigma factor: random in [lo, hi] while training, midpoint at eval.
            if self.training and self.sigma_hi > self.sigma_lo:
                sf = torch.empty_like(w).uniform_(self.sigma_lo, self.sigma_hi)
            else:
                sf = 0.5 * (self.sigma_lo + self.sigma_hi)
            sigma_x = (w * sf).clamp(min=0.5)
            sigma_y = (h * sf).clamp(min=0.5)
            dx = self.grid_x[None] - cx[:, None, None]
            dy = self.grid_y[None] - cy[:, None, None]
            inside = torch.exp(-(dx**2 / (2 * sigma_x[:, None, None] ** 2) + dy**2 / (2 * sigma_y[:, None, None] ** 2)))

        for b in range(batch_size):
            sel = batch_idx == b
            if sel.any():
                mask[b, 0] = inside[sel].max(dim=0).values
        return mask

    def render_per_instance(
        self, bboxes: torch.Tensor, batch_idx: torch.Tensor, batch_size: int, sigma_factor: float | None = None
    ) -> list[torch.Tensor]:
        """Render each bbox as its own gauss mask, grouped per image (NO per-image max merge).

        Training-only target for query-to-instance matching. Unlike ``forward``, the per-bbox
        masks are kept separate so each can be Hungarian-matched to a query attention map.

        Args:
            bboxes: (N, 4) normalized YOLO ``[cx, cy, w, h]``.
            batch_idx: (N,) image index per bbox.
            batch_size: number of images B.
            sigma_factor: fixed gauss width factor; defaults to ``self.sigma_lo`` (independent of
                the fusion-prior sigma so the query GT stays sharp).

        Returns:
            list of length B, each ``(N_b, H, H)`` in [0, 1] (empty ``(0, H, H)`` for images with
            no bboxes).
        """
        H = self.mask_size
        device = self.grid_x.device
        dtype = self.grid_x.dtype
        empty = torch.zeros(0, H, H, device=device, dtype=dtype)
        if bboxes.numel() == 0:
            return [empty for _ in range(batch_size)]
        bboxes = bboxes.to(device=device, dtype=dtype)
        batch_idx = batch_idx.to(device=device, dtype=torch.long)
        cx = bboxes[:, 0] * H
        cy = bboxes[:, 1] * H
        w = bboxes[:, 2] * H
        h = bboxes[:, 3] * H
        sf = float(self.sigma_lo if sigma_factor is None else sigma_factor)
        sigma_x = (w * sf).clamp(min=0.5)
        sigma_y = (h * sf).clamp(min=0.5)
        dx = self.grid_x[None] - cx[:, None, None]
        dy = self.grid_y[None] - cy[:, None, None]
        inst = torch.exp(
            -(dx**2 / (2 * sigma_x[:, None, None] ** 2) + dy**2 / (2 * sigma_y[:, None, None] ** 2))
        )  # (N, H, H)
        return [inst[batch_idx == b] for b in range(batch_size)]

    def extra_repr(self) -> str:
        return f"mask_size={self.mask_size}, mode={self.mode!r}, sigma_factor=[{self.sigma_lo}, {self.sigma_hi}]"


class HeatmapBiasFusion(nn.Module):
    """Soft-hint fusion: mask (+ optional PAN feature) -> bounded per-pixel bias.

    Output shape ``(B, 1, H, W)`` — the caller broadcasts (adds) it to a PAN feature
    of shape ``(B, C, H, W)``. The caller is responsible for resizing the mask to each
    scale before calling forward.

    Per-scale magnitude is controlled by ``beta[i]``, initialized to zero so training
    starts as pure passthrough (vanilla YOLO). Without a hard cap, beta can in
    principle grow large; that is intentional — the detection loss decides how much
    to lean on the heatmap. Output per pixel is in ``[-beta_i, +beta_i]`` via tanh.

    Modes (all default OFF -> byte-identical to the original mask-only shared-conv module):

    - ``feat=True`` (direction B, feature-conditioned): the module also consumes the PAN
      feature ``p`` it is about to bias. A per-scale 1x1 projection ``C_i -> k_feat`` is
      concatenated with the mask, so the conv can (a) SUPPRESS bias where the prior fires
      but features disagree (precision / mAP50) and (b) ADD bias where features look
      anomalous but a sharp prior missed (recall / mAP10). Unlike the GT-extent oracle,
      the feature is available at deploy, so the signal transfers. Requires ``ch`` (the
      per-scale PAN channel counts) at construction.
    - ``per_scale=True`` (direction A): unshare the conv across P3/P4/P5 so the fine scale
      can sharpen (mAP50) while the coarse scale stays broad (mAP10). Cheap (~3x a tiny conv).
    - ``depth=N`` (N>=1): deepen the conv stack for stronger expressiveness. The block is
      ``Conv(in->c_mid) -> GELU -> [Conv(c_mid->c_mid) -> GELU] * N -> Conv(c_mid->1)``, i.e. N
      extra hidden 3x3 convs (at c_mid width) inserted before the output projection. ``depth=0``
      (default) is byte-identical to the original 2-conv block.

    Whether the feature-input path back-propagates into the backbone is controlled by the
    CALLER (it passes ``feat=p`` or ``feat=p.detach()``); this module is agnostic.
    """

    def __init__(self, num_scales: int = 3, c_mid: int = 8, inst_norm: bool = False, residual: bool = False,
                 ch=None, feat: bool = False, k_feat: int = 8, per_scale: bool = False, depth: int = 0):
        super().__init__()
        self.inst_norm = nn.InstanceNorm2d(1, affine=False, track_running_stats=False) if inst_norm else None
        self.residual = residual
        self.feat = bool(feat) and ch is not None
        self.per_scale = bool(per_scale)
        in_ch = 1
        if self.feat:
            # Per-scale 1x1 projection C_i -> k_feat (P3/P4/P5 differ in channel count).
            self.feat_proj = nn.ModuleList([nn.Conv2d(int(c), k_feat, 1) for c in ch])
            in_ch = 1 + k_feat

        n_hidden = max(0, int(depth))

        def _block():
            layers = [nn.Conv2d(in_ch, c_mid, 3, padding=1), nn.GELU()]
            for _ in range(n_hidden):  # extra hidden 3x3 convs at c_mid width (depth>=1)
                layers += [nn.Conv2d(c_mid, c_mid, 3, padding=1), nn.GELU()]
            layers.append(nn.Conv2d(c_mid, 1, 3, padding=1))
            return nn.Sequential(*layers)

        # Unshared per-scale convs (direction A) or a single shared stack (original behavior).
        self.conv = nn.ModuleList([_block() for _ in range(num_scales)]) if self.per_scale else _block()
        self.beta = nn.Parameter(torch.zeros(num_scales))

    def forward(self, mask: torch.Tensor, scale_idx: int, feat: torch.Tensor | None = None) -> torch.Tensor:
        """Return bias (B, 1, H, W) for the given PAN scale.

        Args:
            mask: (B, 1, H, W) already resized to the target PAN scale.
            scale_idx: index into ``self.beta`` (and the per-scale conv / feat_proj).
            feat: (B, C_i, H, W) PAN feature for feature-conditioned mode; pass ``p`` to let
                the feature path reach the backbone, or ``p.detach()`` to block it. Ignored
                unless the module was built with ``feat=True``.

        Returns:
            Bias tensor (B, 1, H, W) in ``[-beta_i, +beta_i]``.
        """
        inst_norm = getattr(self, "inst_norm", None)
        x = inst_norm(mask) if inst_norm is not None else mask
        if getattr(self, "feat", False) and feat is not None:
            x = torch.cat([x, self.feat_proj[scale_idx](feat)], dim=1)
        conv = self.conv[scale_idx] if getattr(self, "per_scale", False) else self.conv
        y = conv(x)
        if getattr(self, "residual", False):
            y = y + mask  # residual on the raw mask channel
        return self.beta[scale_idx] * torch.tanh(y)


def _sincos_pos2d(h: int, w: int, dim: int, device, dtype) -> torch.Tensor:
    """Fixed (non-learnable) 2D sinusoidal positional encoding, shape ``(1, dim, h, w)``.

    DETR-style: half the channels encode the row (y) coordinate, half the column (x), each via
    geometric-frequency sin/cos. Parameter-free and resolution-agnostic (computed from the formula).
    """
    d4 = dim // 4
    omega = 1.0 / (10000.0 ** (torch.arange(d4, device=device, dtype=torch.float32) / d4))  # (d4,)
    y = torch.arange(h, device=device, dtype=torch.float32)[:, None] * omega[None, :]  # (h, d4)
    x = torch.arange(w, device=device, dtype=torch.float32)[:, None] * omega[None, :]  # (w, d4)
    pe_y = torch.cat([y.sin(), y.cos()], dim=1)  # (h, dim//2)
    pe_x = torch.cat([x.sin(), x.cos()], dim=1)  # (w, dim//2)
    pe = torch.cat(
        [pe_y[:, None, :].expand(h, w, dim // 2), pe_x[None, :, :].expand(h, w, dim // 2)], dim=2
    )  # (h, w, dim)
    return pe.permute(2, 0, 1).unsqueeze(0).to(dtype)  # (1, dim, h, w)


class BackboneMemoryBank(nn.Module):
    """Memory-bank anomaly heatmap from backbone features (v1 ADMBHead inference logic).

    Stores L2-normalised normal-image features in a single position-stacked format
    ``bank_stacked [P, C, max_n]`` scored by one batched matmul (ONNX exports as a
    single MatMul node — no per-position loop):

      - global bank (``spatial=False``): P = 1 — every query position is scored
        against the same pooled bank, at the query's native resolution.
      - per-position bank (``spatial=True``): P = H*W — each query position is
        scored only against its own (h, w) sub-bank, at the bank's resolution.

    The flat/global bank is just the P=1 special case: storage, cache format,
    scoring kernel and all call-site checks are shared between the two modes.

    Two modes controlled by ``update``:
      - ``True`` (build): ``forward()`` returns zeros — bank not yet frozen.
      - ``False`` (inference): ``forward()`` returns anomaly scores in [0, 1].

    Calibration measures compactness (mean local cosine density) on a flat coreset
    and tunes β/threshold from a holdout so the normal-score tail sits at
    ``calibration_target_score``; per-position mode calibrates once on the pooled
    bank and broadcasts the scalars into ``bank_thresh``.
    """

    def __init__(
        self,
        temperature: float = 3.0,
        K: int = 5,
        max_bank_size: int | None = None,
        calibration_target_score: float = 0.2,
        calibration_target_quantile: float = 0.95,
        hmap_stretch_strength: float = 0.0,
        holdout_max: int = 5000,
        spatial: bool = False,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.K = int(K)
        self.max_bank_size = max_bank_size
        self.calibration_target_score = float(calibration_target_score)
        self.calibration_target_quantile = float(calibration_target_quantile)
        self.hmap_stretch_strength = float(hmap_stretch_strength)
        self.holdout_max = int(holdout_max)
        self.spatial = bool(spatial)
        self._calibrated = False
        self.feature_dim: int | None = None
        self.update = True
        self._bb_layer_indices: list[int] = []
        self._compactness: float | None = None  # normal-manifold tightness from coreset
        self._threshold: float | None = None  # sigmoid threshold in cosine space
        self.score_chunk_elems = 1 << 27  # max elements per similarity slice in _score_stacked
        # Unified bank storage — the global/flat bank is the P=1 case.
        self._bank_H: int = 0  # 0 = global bank (score at query resolution)
        self._bank_W: int = 0
        self._build_chunks: list[torch.Tensor] = []  # [B, H, W, C] normalised, build-time only
        self.register_buffer("bank_stacked", torch.empty(0, 0, 0), persistent=True)  # [P, C, max_n]
        self.register_buffer("bank_sizes", torch.empty(0, dtype=torch.long), persistent=True)  # [P]
        self.register_buffer("bank_thresh", torch.empty(0), persistent=True)  # [P], NaN = recompute lazily

    @property
    def built(self) -> bool:
        return not self.update

    @property
    def bank_built(self) -> bool:
        """``True`` when a frozen bank with at least one feature vector is loaded."""
        return self.bank_sizes.numel() > 0 and bool((self.bank_sizes > 0).any())

    @property
    def num_features(self) -> int:
        """Total feature vectors across all positions."""
        return int(self.bank_sizes.sum().item()) if self.bank_sizes.numel() else 0

    def __setstate__(self, state):
        """Backfill attributes added after the checkpoint was saved and migrate legacy bank formats."""
        super().__setstate__(state)
        # Drop deprecated projection state from old checkpoints.
        if "proj_dim" in self.__dict__:
            delattr(self, "proj_dim")
        if "_proj_weight" in self._buffers:
            del self._buffers["_proj_weight"]
        if not hasattr(self, "_compactness"):
            self._compactness = None
        if not hasattr(self, "_threshold"):
            self._threshold = None
        if not hasattr(self, "calibration_target_quantile"):
            self.calibration_target_quantile = 0.95
        if not hasattr(self, "hmap_stretch_strength"):
            self.hmap_stretch_strength = 0.0
        if not hasattr(self, "holdout_max"):
            self.holdout_max = 5000
        if not hasattr(self, "spatial"):
            self.spatial = False
        if not hasattr(self, "_bank_H"):
            self._bank_H = 0
            self._bank_W = 0
        if not hasattr(self, "_build_chunks"):
            self._build_chunks = []
        if "bank_stacked" not in self._buffers:
            self.register_buffer("bank_stacked", torch.empty(0, 0, 0), persistent=True)
        if "bank_sizes" not in self._buffers:
            self.register_buffer("bank_sizes", torch.empty(0, dtype=torch.long), persistent=True)
        if "bank_thresh" not in self._buffers:
            self.register_buffer("bank_thresh", torch.empty(0), persistent=True)
        # Migrate the legacy per-position format (transient dev format, pre-unification).
        legacy_stacked = self._buffers.pop("_spatial_bank_stacked", None)
        legacy_sizes = self._buffers.pop("_spatial_bank_sizes", None)
        legacy_thresh = self._buffers.pop("_spatial_thresh_stacked", None)
        self._buffers.pop("_spatial_comp_stacked", None)
        legacy_h = self.__dict__.pop("_spatial_H", 0)
        legacy_w = self.__dict__.pop("_spatial_W", 0)
        self.__dict__.pop("_spatial_bank_chunks", None)
        self.__dict__.pop("_bank_chunks", None)
        if legacy_stacked is not None and legacy_stacked.numel() and not self.bank_stacked.numel():
            self.bank_stacked = legacy_stacked
            self.bank_sizes = legacy_sizes.long()
            P = legacy_stacked.shape[0]
            has_thresh = legacy_thresh is not None and legacy_thresh.numel() == P
            self.bank_thresh = legacy_thresh if has_thresh else torch.full((P,), float("nan"))
            self._bank_H, self._bank_W = int(legacy_h), int(legacy_w)
        # Migrate the legacy flat format ([M, C] buffer).
        legacy_flat = self._buffers.pop("memory_bank", None)
        if legacy_flat is not None and legacy_flat.numel() and not self.bank_stacked.numel():
            valid = legacy_flat[legacy_flat.norm(dim=1) > 0]  # strip zero-padding placeholders
            if valid.shape[0] > 0:
                self.feature_dim = valid.shape[1]
                self._store_global(valid)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_bank(self, data: torch.Tensor | dict) -> None:
        """Load a pre-built memory bank (unified stacked format, or a legacy cache format).

        Accepts:
          - a dict from :meth:`export_state` — ``bank_stacked [P, C, max_n]``, ``bank_sizes [P]``,
            ``bank_thresh [P]``, ``H`` / ``W`` (0 = global), ``feature_dim``, ``temperature``,
            calibration scalars;
          - legacy flat: an L2-normalised ``[M, C]`` tensor, or a cache dict wrapping it under
            ``"memory_bank"`` (repacked as the P=1 stacked case);
          - legacy per-position cache dicts (``comp`` / ``thresh`` array keys).
        """
        device = self.bank_stacked.device
        if isinstance(data, dict) and isinstance(data.get("memory_bank"), dict):
            # Legacy per-position cache: the outer dict wraps the real payload.
            data = dict(data["memory_bank"])
        if isinstance(data, dict) and data.get("bank_stacked") is not None:
            bank = data["bank_stacked"]
            if bank.numel() == 0:
                return
            self.bank_stacked = bank.to(device)
            self.bank_sizes = data["bank_sizes"].to(device).long()
            P = bank.shape[0]
            thresh = data.get("bank_thresh", data.get("thresh"))
            has_thresh = thresh is not None and thresh.numel() == P
            self.bank_thresh = thresh.to(device).reshape(P) if has_thresh else torch.full(
                (P,), float("nan"), device=device
            )
            self._bank_H, self._bank_W = int(data.get("H", 0)), int(data.get("W", 0))
            self.feature_dim = int(data["feature_dim"])
            if data.get("temperature") is not None:
                self.temperature = float(data["temperature"])
            self._compactness = data.get("_compactness")
            self._threshold = data.get("_threshold")
            self._calibrated = bool(data.get("_calibrated", True))
            self.update = False
            return
        # Legacy flat: raw tensor or {"memory_bank": tensor, ...} cache dict.
        features = data["memory_bank"] if isinstance(data, dict) else data
        if features is None or features.numel() == 0:
            return
        if not torch.isfinite(features).all():
            raise ValueError(f"BackboneMemoryBank.load_bank: features contain NaN/Inf ({features.shape})")
        self.feature_dim = features.shape[1]
        if isinstance(data, dict):
            if data.get("temperature") is not None:
                self.temperature = float(data["temperature"])
            self._threshold = data.get("_threshold")
            self._compactness = data.get("_compactness")
        else:
            self._threshold = None  # trigger lazy compactness/threshold recompute on next score
            self._compactness = None
        self._calibrated = True
        self._store_global(F.normalize(features.to(device), p=2, dim=1))
        self.update = False

    def export_state(self) -> dict:
        """Serializable bank state — the single cache format for both bank modes."""
        return {
            "format": "yoloa.bank.stacked.v1",
            "bank_stacked": self.bank_stacked.detach().cpu(),
            "bank_sizes": self.bank_sizes.detach().cpu(),
            "bank_thresh": self.bank_thresh.detach().cpu(),
            "H": self._bank_H,
            "W": self._bank_W,
            "feature_dim": self.feature_dim,
            "temperature": float(self.temperature),
            "_compactness": self._compactness,
            "_threshold": self._threshold,
            "_calibrated": bool(self._calibrated),
        }

    def _store_global(self, mem: torch.Tensor) -> None:
        """Store a flat [M, C] bank as the P=1 case of the stacked format."""
        device = self.bank_stacked.device
        self.bank_stacked = mem.t().unsqueeze(0).contiguous().to(device)  # [1, C, M]
        self.bank_sizes = torch.tensor([mem.shape[0]], dtype=torch.long, device=device)
        t = self._threshold
        self.bank_thresh = torch.full((1,), float(t) if t is not None else float("nan"), device=device)
        self._bank_H = self._bank_W = 0

    def freeze_memory_bank(self) -> None:
        """Coreset-compress, calibrate via compactness + holdout, then freeze the bank."""
        import logging

        if self.spatial:
            self._freeze_spatial()
        else:
            self._freeze_global()
        self._build_chunks = []
        self.update = False
        logger = logging.getLogger(__name__)
        if logger.isEnabledFor(logging.DEBUG):
            n_pos = int((self.bank_sizes > 0).sum().item()) if self.bank_sizes.numel() else 0
            logger.debug(
                "BackboneMemoryBank: frozen %d position(s)  grid=%dx%d  max_n=%d  total_vecs=%d",
                n_pos,
                self._bank_H,
                self._bank_W,
                int(self.bank_sizes.max().item()) if self.bank_sizes.numel() else 0,
                self.num_features,
            )

    def _freeze_global(self) -> None:
        """Global-mode freeze: pool all positions, coreset once, store as P=1."""
        mem = torch.cat([c.reshape(-1, c.shape[-1]) for c in self._build_chunks], dim=0) if self._build_chunks else (
            self.bank_stacked.new_zeros(0, 0)
        )
        # Coreset subsample, collect holdout (features not selected into coreset)
        holdout = None
        if self.max_bank_size is not None and mem.shape[0] > self.max_bank_size:
            coreset, coreset_idx = self._coreset_subsample(mem, self.max_bank_size, return_indices=True)
            holdout_mask = torch.ones(mem.shape[0], dtype=torch.bool, device=mem.device)
            holdout_mask[coreset_idx] = False
            holdout = mem[holdout_mask]
            mem = coreset

        if mem.shape[0] == 0:
            return
        self._calibrate_compactness(mem)
        if holdout is not None and holdout.shape[0] > 0:
            if holdout.shape[0] > self.holdout_max:
                holdout = holdout[: self.holdout_max]
            self._calibrate_threshold_from_holdout(holdout, mem)
        self._store_global(mem)

    def _freeze_spatial(self) -> None:
        """Per-position freeze: per-position coreset -> stacked tensors.

        Each position gets its own sub-bank capped at ``max_bank_size`` features.
        Calibration uses a flat reference bank built from pooled per-position samples
        so thresholds match the global bank's broader normal-distribution view.
        """
        from ultralytics.utils import TQDM

        H, W, C = self._bank_H, self._bank_W, self.feature_dim
        P = H * W
        if P == 0 or not self._build_chunks:
            return
        all_feats = torch.cat(self._build_chunks, dim=0)  # [N, H, W, C]

        per_pos_cap = self.max_bank_size  # direct per-position cap
        per_pos_holdout_cap = max(1, (self.holdout_max // P) + 1) if self.holdout_max else 20
        device = all_feats.device

        # Phase 1: per-position coresets + collect holdouts for flat calibration
        spatial_banks: list[torch.Tensor | None] = [None] * P
        flat_holdout_parts: list[torch.Tensor] = []

        pbar = TQDM(total=P, desc="Spatial banks", leave=False)
        for pos in range(P):
            pbar.update(1)
            raw = all_feats[:, pos // W, pos % W, :]  # [N_pos, C]
            n_raw = raw.shape[0]
            if n_raw == 0:
                continue

            if per_pos_cap is not None and n_raw > per_pos_cap:
                mem, coreset_idx = self._coreset_subsample(
                    raw, per_pos_cap, return_indices=True)
                holdout_mask = torch.ones(n_raw, dtype=torch.bool, device=device)
                holdout_mask[coreset_idx] = False
                holdout = raw[holdout_mask]
            else:
                mem = raw
                if n_raw >= 3:
                    n_h = min(max(1, n_raw // 3), per_pos_holdout_cap)
                    perm = torch.randperm(n_raw, device=device)
                    mem = raw[perm[n_h:]]
                    holdout = raw[perm[:n_h]]
                else:
                    holdout = None

            spatial_banks[pos] = mem

            if holdout is not None and holdout.shape[0] > 0:
                n_h = min(holdout.shape[0], per_pos_holdout_cap)
                if n_h < holdout.shape[0]:
                    idx = torch.randperm(holdout.shape[0], device=device)[:n_h]
                    holdout = holdout[idx]
                flat_holdout_parts.append(holdout)
        pbar.close()

        if all(b is None for b in spatial_banks):
            self.bank_stacked = torch.zeros(P, 0, 0, device=device)
            self.bank_sizes = torch.zeros(P, dtype=torch.long, device=device)
            self.bank_thresh = torch.zeros(P, device=device)
            return

        max_n = max((b.shape[0] for b in spatial_banks if b is not None), default=0)
        stacked = torch.zeros(P, C, max_n, device=device)
        sizes = torch.zeros(P, dtype=torch.long, device=device)

        # Phase 2: flat reference bank for calibration — pools per-position banks
        # into a single flat bank, calibrates once, then broadcasts to all positions.
        flat_pool = torch.cat([b for b in spatial_banks if b is not None], dim=0)
        ref_size = min(flat_pool.shape[0], (self.max_bank_size or 10000))
        ref_bank, _ = self._coreset_subsample(flat_pool, ref_size, return_indices=True)

        self._calibrate_compactness(ref_bank)
        if flat_holdout_parts:
            flat_holdout = torch.cat(flat_holdout_parts, dim=0)
            if flat_holdout.shape[0] > self.holdout_max:
                idx = torch.randperm(flat_holdout.shape[0], device=device)[:self.holdout_max]
                flat_holdout = flat_holdout[idx]
            self._calibrate_threshold_from_holdout(flat_holdout, ref_bank)

        for pos in range(P):
            bank = spatial_banks[pos]
            if bank is None or bank.shape[0] == 0:
                continue
            n = bank.shape[0]
            stacked[pos, :, :n] = bank.t()  # [C, n]
            sizes[pos] = n

        self.bank_stacked = stacked
        self.bank_sizes = sizes
        self.bank_thresh = torch.full((P,), float(getattr(self, "_threshold", 0.4) or 0.4), device=device)
        self._calibrated = True

    def reset_memory_bank(self) -> None:
        """Clear the bank and return to build mode."""
        device = self.bank_stacked.device
        self.feature_dim = None
        self._calibrated = False
        self._compactness = None
        self._threshold = None
        self.update = True
        self._build_chunks = []  # defer cat until freeze
        self._bank_H = self._bank_W = 0
        self.bank_stacked = torch.empty(0, 0, 0, device=device)
        self.bank_sizes = torch.empty(0, dtype=torch.long, device=device)
        self.bank_thresh = torch.empty(0, device=device)

    def accumulate_features(self, feat_dict: dict[int, torch.Tensor]) -> None:
        """Extract and accumulate backbone features into the memory bank (build phase).

        Fused backbone features are L2-normalised per spatial position and appended
        to a chunk list; the bank is materialised once in ``freeze_memory_bank``
        to avoid O(N²) reallocation from repeated ``torch.cat``. The same chunks
        serve both modes: global freeze pools every position, per-position freeze
        (``spatial=True``) splits them by (h, w).
        """
        if not feat_dict:
            return
        fused = self._build_fused_feature(feat_dict)  # (B, C, H, W)
        C, H, W = fused.shape[1], fused.shape[2], fused.shape[3]
        if self.feature_dim is None:
            self.feature_dim = C

        if self.spatial:
            if self._bank_H == 0:
                self._bank_H, self._bank_W = H, W
            elif H != self._bank_H or W != self._bank_W:
                raise ValueError(
                    f"Spatial dims changed during build: ({self._bank_H},{self._bank_W}) -> ({H},{W})"
                )
        normed = F.normalize(fused.permute(0, 2, 3, 1).reshape(-1, C), p=2, dim=1)  # [B*H*W, C]
        self._build_chunks.append(normed.reshape(-1, H, W, C))  # [B, H, W, C]

    def forward(self, feat_dict: dict[int, torch.Tensor]) -> torch.Tensor:
        """Produce (B, 1, H, W) anomaly heatmap from backbone features.

        During build mode or when the bank is empty, returns zeros. Both bank modes
        run the same stacked scoring kernel; a per-position bank additionally
        interpolates the query to the bank's grid and scores at that resolution.
        """
        b, device = self._resolve_batch_size(feat_dict)
        h = feat_dict[list(feat_dict.keys())[0]].shape[2] if feat_dict else 80
        w = feat_dict[list(feat_dict.keys())[0]].shape[3] if feat_dict else 80
        if self.update:
            return torch.zeros(b, 1, h, w, device=device)
        fused = self._build_fused_feature(feat_dict)  # (B, C, H, W)
        B, C, H, W = fused.shape
        if C != self.feature_dim or not self.bank_built:
            return torch.zeros(b, 1, H, W, device=device)
        self._ensure_thresh()
        if self._bank_H > 0:  # per-position bank: score each query cell against its own sub-bank
            if H != self._bank_H or W != self._bank_W:
                fused = F.interpolate(fused, size=(self._bank_H, self._bank_W), mode="bilinear", align_corners=False)
            P = self._bank_H * self._bank_W
            q = F.normalize(fused.permute(0, 2, 3, 1).reshape(B, P, C), p=2, dim=-1)
            scores = self._score_stacked(q.transpose(0, 1))  # [P, B]
            hmap = scores.transpose(0, 1).reshape(B, 1, self._bank_H, self._bank_W)
        else:  # global bank: every query cell against the pooled bank, native resolution
            q = F.normalize(fused.permute(0, 2, 3, 1).reshape(1, B * H * W, C), p=2, dim=-1)
            hmap = self._score_stacked(q).reshape(B, 1, H, W)
        s = self.hmap_stretch_strength
        if s:
            hmap = (hmap + s * hmap * hmap).clamp(0, 1)
        return hmap

    def _score_stacked(self, q: torch.Tensor) -> torch.Tensor:
        """Noisy-OR anomaly scores for position-grouped queries — the single scoring kernel.

        One batched matmul serves both bank modes (ONNX: a single MatMul node, no
        per-position loop). Zero-padded bank slots are masked to cosine −1 so they
        can never win the top-K.

        Args:
            q: [P, N, C] L2-normalised queries (global: P=1, N=B·H·W; per-position: N=B).

        Returns:
            [P, N] anomaly scores in [0, 1]; positions with an empty sub-bank score 0.
        """
        bank = self.bank_stacked  # [P, C, max_n]
        P, _, max_n = bank.shape
        n_q = q.shape[1]
        pad = torch.arange(max_n, device=bank.device).view(1, 1, max_n) >= self.bank_sizes.view(P, 1, 1)
        thresh = self.bank_thresh.view(P, 1, 1)
        beta = self.temperature
        k = min(self.K, max_n)
        chunk = max(1, int(getattr(self, "score_chunk_elems", 1 << 27)) // max(P * max_n, 1))
        out = []
        for i in range(0, n_q, chunk):
            cos = torch.bmm(q[:, i : i + chunk], bank)  # [P, n, max_n]
            cos = cos.masked_fill(pad, -1.0)
            if self.training:
                cos = cos.masked_fill(cos > 0.999, float("-inf"))  # self-match exclusion
            psi = torch.sigmoid(beta * (cos.topk(k, dim=-1).values - thresh))
            out.append(torch.exp(torch.log((1.0 - psi).clamp(min=1e-8)).mean(dim=-1)))
        score = torch.cat(out, dim=1) if len(out) > 1 else out[0]
        return score.masked_fill(self.bank_sizes.view(P, 1) == 0, 0.0).clamp(0, 1)

    def _ensure_thresh(self) -> None:
        """Materialise ``bank_thresh`` — lazy compactness/threshold recompute after legacy loads."""
        P = self.bank_stacked.shape[0]
        if self.bank_thresh.numel() == P and bool(torch.isfinite(self.bank_thresh).all()):
            return
        if self._compactness is None:
            self._compactness = self._measure_compactness(self._flat_view())
            self._threshold = None  # force recompute below
        if self._threshold is None:
            import math

            t = self.calibration_target_score
            logit = math.log(max((1.0 - t) / max(t, 1e-6), 1e-6))
            self._threshold = self._compactness - logit / max(self.temperature, 0.1)
        self.bank_thresh = torch.full((P,), float(self._threshold), device=self.bank_stacked.device)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_batch_size(feat_dict: dict[int, torch.Tensor]) -> tuple[int, torch.device]:
        for v in feat_dict.values():
            return v.shape[0], v.device
        return 1, torch.device("cpu")

    def _effective_bank(self) -> torch.Tensor:
        """Return real memory-bank entries as a flat [M, C] view (padding stripped)."""
        if self.update and self._build_chunks:
            return torch.cat([c.reshape(-1, c.shape[-1]) for c in self._build_chunks], dim=0)
        return self._flat_view()

    def _flat_view(self) -> torch.Tensor:
        """All frozen bank vectors as a flat [M, C] tensor (P=1: the original bank, zero-copy order)."""
        bank, sizes = self.bank_stacked, self.bank_sizes
        if bank.numel() == 0:
            return bank.new_zeros(0, 0)
        cols = [bank[p, :, : int(sizes[p])].t() for p in range(bank.shape[0]) if int(sizes[p]) > 0]
        return torch.cat(cols, dim=0) if cols else bank.new_zeros(0, 0)

    def _build_fused_feature(self, feat_dict: dict[int, torch.Tensor]) -> torch.Tensor:
        """Gather backbone features at configured layer indices and concat."""
        indices = self._bb_layer_indices
        if not indices:
            return feat_dict[list(feat_dict.keys())[0]]
        feats = [feat_dict[i] for i in indices if i in feat_dict]
        if not feats:
            return feat_dict[list(feat_dict.keys())[0]]
        target_size = feats[0].shape[-2:]
        aligned = [
            F.interpolate(f, size=target_size, mode="nearest") if f.shape[-2:] != target_size else f for f in feats
        ]
        return torch.cat(aligned, dim=1) if len(aligned) > 1 else aligned[0]

    @staticmethod
    def _sample_idx(n: int, k: int, device) -> torch.Tensor:
        """First ``k`` of a seeded CPU randperm, moved to ``device``.

        Calibration (compactness / β / threshold) must be reproducible: the bank cache stores the
        calibrated state, and an unseeded device-side randperm made β/threshold jitter between
        otherwise-identical builds (and between CUDA and MPS). CPU RNG with a fixed seed gives the
        same sample everywhere.
        """
        g = torch.Generator().manual_seed(0)
        return torch.randperm(n, generator=g)[:k].to(device)

    def estimate_temperature(self) -> float:
        """Estimate β from the current bank WITHOUT modifying state (lightweight, for MoCo live estimate)."""
        import math

        mem = self._effective_bank()
        if mem.shape[0] < 2:
            return self.temperature
        with torch.no_grad():
            k = min(self.K, mem.shape[0])
            n_sample = min(512, mem.shape[0])
            idx = self._sample_idx(mem.shape[0], n_sample, mem.device)
            sample = mem[idx]
            sim = sample @ mem.t()
            topk_sim = sim.topk(k=k, dim=1).values
            mean_topk = topk_sim.mean(dim=1)
            s_max = mean_topk.max().clamp(0.0, 1.0 - 1e-4).item()
            beta = -math.log(1.0 - self.calibration_target_score) / max(1.0 - s_max, 1e-6)
            beta = max(0.1, min(20.0, beta))
        return beta

    def _measure_compactness(self, mem: torch.Tensor) -> float:
        """Compute compactness from the bank without touching ``self.temperature``.

        Used for lazy restore after state_dict load and by ``_calibrate_compactness``.
        """
        with torch.no_grad():
            k = min(self.K, mem.shape[0])
            n_sample = min(512, mem.shape[0])
            idx = self._sample_idx(mem.shape[0], n_sample, mem.device)
            sample = mem[idx]
            sim = sample @ mem.t()  # [n_sample, M]
            sim[torch.arange(n_sample, device=mem.device), idx] = -1.0
            topk_sim = sim.topk(k=k, dim=1).values
            local_density = topk_sim.mean(dim=1)
            return local_density.mean().clamp(0.0, 1.0 - 1e-4).item()

    def _calibrate_compactness(self, mem: torch.Tensor) -> None:
        """Measure compactness and calibrate sigmoid threshold in cosine space.

        Compactness = mean local cosine density on the coreset.  The sigmoid
        operates directly on cosine similarity so that the dynamic range is
        never compressed by the bank spread:

            psi = sigmoid(β × (cos − threshold_cos))

        threshold_cos = compactness − logit(1−target)/β.

        A normal query has cos ≈ compactness → psi ≈ 1−target → normal
        anomaly score ≈ target.  β is the user-controlled ``temperature``.
        """
        import math

        compactness = self._measure_compactness(mem)
        self._compactness = compactness
        beta = self.temperature
        t = self.calibration_target_score
        logit = math.log(max((1.0 - t) / max(t, 1e-6), 1e-6))
        self._threshold = compactness - logit / max(beta, 0.1)
        self._calibrated = True
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(
            "BackboneMemoryBank: compactness=%.4f  β=%.3f  threshold_cos=%.4f  target=%.2f",
            compactness,
            beta,
            self._threshold,
            self.calibration_target_score,
        )

    def _calibrate_threshold_from_holdout(self, holdout: torch.Tensor, mem: torch.Tensor) -> None:
        """Calibrate β & threshold so p95 of hold-out normal scores ≈ target.

        The user β is a sensitivity floor — it is never lowered.  For each
        candidate β, binary-searches ``threshold_cos`` to put p95 at
        ``calibration_target_score``, then picks the β that gives the tightest
        normal-score distribution (smallest p95−p5).
        """
        import math
        import logging

        logger = logging.getLogger(__name__)

        n_holdout = holdout.shape[0]
        target = self.calibration_target_score
        target_q = self.calibration_target_quantile  # e.g. 0.95 → 95% of normal scores ≤ target
        c = self._compactness
        beta0 = self.temperature

        # β candidates: user β as floor, log-spaced upward
        beta_candidates = [beta0]
        n_extra = 8
        for i in range(1, n_extra + 1):
            b = beta0 * (10 ** (i / n_extra))
            if b > 100:
                break
            beta_candidates.append(round(b, 4))

        # Pre-compute cosine matrix and top-K once (sigmoid is monotonic,
        # so the top-K indices of cos are identical to top-K of psi).
        q_feats = F.normalize(holdout.view(-1, self.feature_dim), p=2, dim=1)
        cos_mat = q_feats @ mem.t()  # [N_holdout, M]
        k = min(self.K, mem.shape[0])
        topk_cos = cos_mat.topk(k=k, dim=1).values  # [N_holdout, k] — only these matter

        # z-score for the target quantile (√2 · erf⁻¹(2q−1))
        z_q = math.sqrt(2) * torch.erfinv(torch.tensor(2.0 * target_q - 1.0)).item()

        def _scores_for(beta, thresh):
            """Compute anomaly scores from pre-computed top-K cos values."""
            psi = torch.sigmoid(beta * (topk_cos - thresh))
            log_prob = torch.log((1.0 - psi).clamp(min=1e-8)).mean(dim=1)
            return torch.exp(log_prob).clamp(0, 1)

        def _tail_stat(scores):
            """Gaussian tail: μ + z·σ."""
            return scores.mean().item() + z_q * scores.std().item()

        def _spread(scores):
            """Score spread: standard deviation (smaller = tighter normal distribution)."""
            return scores.std().item()

        def _find_thresh(beta):
            """Binary-search threshold so tail-stat ≈ target."""
            half_range = 3.0 / max(beta, 0.1)
            lo, hi = c - half_range, c + half_range
            s_lo = _tail_stat(_scores_for(beta, lo))
            s_hi = _tail_stat(_scores_for(beta, hi))
            for _ in range(5):
                if s_lo > target and lo > c - 5.0:
                    lo -= half_range
                    s_lo = _tail_stat(_scores_for(beta, lo))
                if s_hi < target and hi < c + 5.0:
                    hi += half_range
                    s_hi = _tail_stat(_scores_for(beta, hi))
            if not (s_lo <= target <= s_hi):
                return None, float("inf"), float("inf")
            for _ in range(20):
                mid = (lo + hi) / 2
                sm = _tail_stat(_scores_for(beta, mid))
                if sm > target:
                    hi = mid
                else:
                    lo = mid
            thresh = (lo + hi) / 2
            scores = _scores_for(beta, thresh)
            achieved = _tail_stat(scores)
            spread = _spread(scores)
            return thresh, achieved, spread

        best_beta, best_thresh, best_achieved, best_spread = beta0, c, float("inf"), float("inf")
        for beta in beta_candidates:
            thresh, achieved, spread = _find_thresh(beta)
            if thresh is not None and spread < best_spread:
                best_spread, best_achieved, best_beta, best_thresh = spread, achieved, beta, thresh

        old_beta = self.temperature
        self.temperature = best_beta
        self._threshold = best_thresh
        logger.debug(
            "BackboneMemoryBank: gauss calibration  n=%d  q=%.2f  z=%.4f  "
            "β %.3f→%.3f  thresh(formula)=%.4f→(holdout)=%.4f  "
            "target=%.2f→achieved=%.4f  spread=%.4f",
            n_holdout,
            target_q,
            z_q,
            old_beta,
            best_beta,
            c - math.log(max((1.0 - target) / max(target, 1e-6), 1e-6)) / max(old_beta, 0.1),
            best_thresh,
            target,
            best_achieved,
            best_spread,
        )

    @staticmethod
    def _coreset_subsample(mem: torch.Tensor, max_size: int, return_indices: bool = False):
        """Greedy k-center coreset on L2-normalised features using cosine distance.

        Complexity: O(max_size × M). Features must be L2-normalised.

        Batched greedy: selects ``batch_size`` farthest points per iteration and
        computes distances against all of them in one GEMM call, amortizing the
        large bank read over multiple centres and trading a small amount of
        greediness for a large wall-clock speedup on memory-bandwidth-limited
        devices (MPS / CPU).

        Args:
            mem: [M, C] L2-normalised feature bank.
            max_size: Target coreset size.
            return_indices: If True, also return the indices of selected rows.

        Returns:
            Coreset tensor, or (coreset, indices) if ``return_indices``.
        """
        from ultralytics.utils import TQDM

        M = mem.shape[0]
        if M <= max_size:
            idx = torch.arange(M, device=mem.device)
            return (mem, idx) if return_indices else mem
        device = mem.device
        BATCH = 64  # centres per GEMM call — amortizes bank reads
        dist = torch.full((M,), float("inf"), device=device, dtype=torch.float32)
        selected: list[int] = []
        mean = mem.mean(dim=0)
        mean = mean / mean.norm().clamp(min=1e-8)
        seed = int((mem @ mean).argmax().item())
        selected.append(seed)
        # seed distance so the first topk isn't random (all-inf)
        centre = mem[seed].unsqueeze(0)
        seed_cos = (mem @ centre.t()).squeeze(1)
        dist = (1.0 - seed_cos).clamp(min=0.0)
        n_needed = max_size - len(selected)
        pbar = TQDM(total=n_needed, desc="Coreset subsample", leave=False)
        while n_needed > 0:
            k = min(BATCH, max_size - len(selected))
            _, top_idx = dist.topk(k)
            selected.extend(top_idx.tolist())
            centres = mem[top_idx]  # [k, C]
            cos_sim = mem @ centres.t()  # [M, k]
            new_dist = (1.0 - cos_sim).clamp(min=0.0).min(dim=1).values  # [M]
            dist = torch.minimum(dist, new_dist)
            n_needed = max_size - len(selected)
            pbar.update(k)
        pbar.close()
        sel = torch.tensor(selected, device=device)
        return (mem[sel], sel) if return_indices else mem[sel]


def heatmap_local_contrast(h: torch.Tensor, k: int = 9, eps: float = 1e-3) -> torch.Tensor:
    """Local z-score of a [B,1,H,W] map in [0,1]: (h - local_mean) / (local_std + eps), ~[-1,1]."""
    mean = F.avg_pool2d(h, k, stride=1, padding=k // 2)
    sq = F.avg_pool2d(h * h, k, stride=1, padding=k // 2)
    std = (sq - mean * mean).clamp(min=0).sqrt()
    z = (h - mean) / (std + eps)
    return (z / 3.0).clamp(-1, 1)
