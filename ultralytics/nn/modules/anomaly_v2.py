# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLOA v2 modules: bbox-mask renderer, heatmap bias fusion, optional SegBranch.

Soft-hint fusion: a 1-channel mask is turned into a bounded per-pixel bias added
(broadcast over channels) to PAN features before the Detect head. PAN feature
addition keeps the Detect head unmodified, lets reg and cls both see the bias
(empirical question — see spec §2), and is bounded vs the previous multiplicative
amplifier that forced detections.

See docs_yoloa_v2/specs/2026-06-02-softhint-fusion-design.md.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER

from .conv import Conv

__all__ = ("BboxMaskRenderer", "HeatmapBiasFusion", "SegBranch", "binary_seg_loss", "BackboneMemoryBank")


class BboxMaskRenderer(nn.Module):
    """Render normalized YOLO-format bboxes into a 1xHxW mask.

    Two modes:
      - "rect":  hard rectangle (inside bbox = 1, outside = 0)
      - "gauss": per-bbox 2D Gaussian centered at bbox center,
                 sigma_x = w * sigma_factor, sigma_y = h * sigma_factor;
                 multiple bboxes in the same image are combined with max.

    Output spatial size is fixed at construction (default 80 to match P3).
    """

    def __init__(self, mask_size: int = 80, mode: str = "rect", sigma_factor: float = 0.25):
        super().__init__()
        assert mode in ("rect", "gauss"), f"mode must be 'rect' or 'gauss', got {mode!r}"
        self.mask_size = int(mask_size)
        self.mode = mode
        self.sigma_factor = float(sigma_factor)
        ys, xs = torch.meshgrid(
            torch.arange(self.mask_size, dtype=torch.float32),
            torch.arange(self.mask_size, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("grid_x", xs + 0.5, persistent=False)
        self.register_buffer("grid_y", ys + 0.5, persistent=False)

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
            sigma_x = (w * self.sigma_factor).clamp(min=0.5)
            sigma_y = (h * self.sigma_factor).clamp(min=0.5)
            dx = self.grid_x[None] - cx[:, None, None]
            dy = self.grid_y[None] - cy[:, None, None]
            inside = torch.exp(
                -(dx**2 / (2 * sigma_x[:, None, None] ** 2) + dy**2 / (2 * sigma_y[:, None, None] ** 2))
            )

        for b in range(batch_size):
            sel = batch_idx == b
            if sel.any():
                mask[b, 0] = inside[sel].max(dim=0).values
        return mask

    def extra_repr(self) -> str:
        return f"mask_size={self.mask_size}, mode={self.mode!r}, sigma_factor={self.sigma_factor}"


class HeatmapBiasFusion(nn.Module):
    """Soft-hint fusion: 1-ch mask -> bounded per-pixel bias broadcast onto PAN features.

    Output shape ``(B, 1, H, W)`` — the caller broadcasts (adds) it to a PAN feature
    of shape ``(B, C, H, W)``. The conv stack is SHARED across PAN scales; the caller
    is responsible for resizing the mask to each scale before calling forward.

    Per-scale magnitude is controlled by ``beta[i]``, initialized to zero so training
    starts as pure passthrough (vanilla YOLO). Without a hard cap, beta can in
    principle grow large; that is intentional — the detection loss decides how much
    to lean on the heatmap.

    Output per pixel is in ``[-beta_i, +beta_i]`` via tanh.
    """

    def __init__(self, num_scales: int = 3, c_mid: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, c_mid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c_mid, 1, 3, padding=1),
        )
        self.beta = nn.Parameter(torch.zeros(num_scales))

    def forward(self, mask: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """Return bias (B, 1, H, W) for the given PAN scale.

        Args:
            mask: (B, 1, H, W) already resized to the target PAN scale.
            scale_idx: index into ``self.beta``.

        Returns:
            Bias tensor (B, 1, H, W) in ``[-beta_i, +beta_i]``.
        """
        return self.beta[scale_idx] * torch.tanh(self.conv(mask))


class SegBranch(nn.Module):
    """Lightweight semantic-segmentation head that predicts a 1-channel anomaly heatmap.

    Consumes the P3 and P4 PAN features and emits per-pixel logits at P3 resolution
    (e.g. 80x80 for 640 input). A P4 auxiliary head provides deep supervision during
    training.
    """

    def __init__(self, ch: tuple, nc: int = 1, c_mid: int | None = None):
        super().__init__()
        self.nc = nc
        c_mid = ch[0] if c_mid is None else c_mid
        self.classifier = nn.Sequential(Conv(ch[0], c_mid, 3), nn.Conv2d(c_mid, nc, 1))
        self.aux_head = nn.Sequential(Conv(ch[1], c_mid, 3), nn.Conv2d(c_mid, nc, 1)) if len(ch) > 1 else None

    def forward(self, x: list[torch.Tensor]):
        logits = self.classifier(x[0])
        if self.training and self.aux_head is not None:
            return logits, self.aux_head(x[1])
        return logits


def binary_seg_loss(
    logits: torch.Tensor, target: torch.Tensor, aux_logits: torch.Tensor | None = None, aux_weight: float = 0.4
) -> torch.Tensor:
    """BCE + soft-Dice loss for a single-channel anomaly heatmap."""
    if target.shape[2:] != logits.shape[2:]:
        target = F.interpolate(target, size=logits.shape[2:], mode="nearest")
    target = target.to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = logits.sigmoid()
    inter = (prob * target).sum(dim=(1, 2, 3))
    card = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (1.0 - (2.0 * inter + 1.0) / (card + 1.0)).mean()
    loss = bce + dice
    if aux_logits is not None:
        aux_t = target
        if aux_t.shape[2:] != aux_logits.shape[2:]:
            aux_t = F.interpolate(target, size=aux_logits.shape[2:], mode="nearest")
        loss = loss + aux_weight * F.binary_cross_entropy_with_logits(aux_logits, aux_t)
    return loss


def _coreset_subsample(mem: torch.Tensor, max_size: int) -> torch.Tensor:
    """Greedy k-center coreset: select *max_size* rows that maximally cover the
    feature space (minimises the largest nearest-neighbour gap).

    Args:
        mem: [M, C] L2-normalised feature matrix.
        max_size: target number of entries to keep.

    Returns:
        [max_size, C] selected subset.
    """
    M = mem.shape[0]
    if M <= max_size:
        return mem
    device = mem.device
    dist = torch.full((M,), float("inf"), device=device, dtype=torch.float32)
    selected: list[int] = []
    mean = mem.mean(0)
    mean = mean / mean.norm().clamp(min=1e-8)
    selected.append(int((mem @ mean).argmax().item()))
    for _ in range(max_size - 1):
        centre = mem[selected[-1]].unsqueeze(0)
        new_dist = (1.0 - (mem @ centre.t()).squeeze(1)).clamp(min=0.0)
        dist = torch.minimum(dist, new_dist)
        selected.append(int(dist.argmax().item()))
    return mem[torch.tensor(selected, device=device)]


class BackboneMemoryBank(nn.Module):
    """Training-free anomaly prior: KNN cosine distance against stored normal features.

    Accumulate backbone features from normal images via ``accumulate(feat)``,
    then call ``freeze()`` to compress and lock the bank.  At inference,
    ``heatmap(feat)`` returns a ``(B, 1, H, W)`` anomaly score map in [0, 1]
    that can be fed directly into ``set_external_mask_once()``.

    The bank stores **L2-normalised** features.  Scores are computed as
    ``1 - mean_top_k_cosine_similarity`` converted to a probability via a
    Noisy-OR KNN geometric non-match score:
        psi_i = exp(-temperature * cosine_distance_i)
        score = exp(mean(log(1 - psi_i)))

    Args:
        feature_dim: Expected channel depth of the backbone feature (resolved
            automatically on first call to ``accumulate`` if left None).
        K: Number of nearest neighbours for KNN scoring.
        temperature: Noisy-OR temperature β.  Higher → sharper separation.
        max_bank_size: Compress to this many entries after ``freeze()`` via
            greedy k-center coreset.  None = no compression.
    """

    def __init__(
        self,
        feature_dim: int | None = None,
        K: int = 5,
        temperature: float = 3.0,
        max_bank_size: int | None = 50_000,
        accumulate_thresh: float = 0.0,
        auto_temperature: bool = True,
        calibration_target_score: float = 0.2,
        min_calibration_bank_size: int = 50,
    ) -> None:
        super().__init__()
        self.K = int(K)
        self.temperature = float(temperature)
        self.max_bank_size = max_bank_size
        self.feature_dim = feature_dim
        self.accumulate_thresh = float(accumulate_thresh)
        self.auto_temperature = bool(auto_temperature)
        self.calibration_target_score = float(calibration_target_score)
        self.min_calibration_bank_size = int(min_calibration_bank_size)
        self._frozen = False
        # bank: [M, C], empty until first accumulate()
        self.register_buffer("bank", torch.empty(0, 0), persistent=True)

    # ── build ──────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Discard all stored features and unfreeze."""
        self.bank = torch.empty(0, 0, device=self.bank.device)
        self.feature_dim = None
        self.accumulate_thresh = getattr(self, "accumulate_thresh", 0.0)
        self._frozen = False

    def accumulate(self, feat: torch.Tensor) -> None:
        """Append L2-normalised spatial features from one forward pass.

        Features too similar to existing bank entries are dropped (OBMA-style
        novelty filter) when ``accumulate_thresh > 0``.

        Args:
            feat: Backbone feature map ``(B, C, H, W)`` or patch matrix ``(N, C)``.
        """
        if self._frozen:
            return
        if feat.dim() == 4:
            B, C, H, W = feat.shape
            feat = feat.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        feat = F.normalize(feat.float(), p=2, dim=1)
        if self.feature_dim is None:
            self.feature_dim = feat.shape[1]
        # OBMA-style novelty filter: drop features already well-represented in the bank.
        if self.accumulate_thresh > 0.0 and self.bank.numel() > 0:
            sim = feat @ self.bank.to(feat.device).t()   # (N, M)
            max_sim = sim.max(dim=1).values               # (N,) cosine sim to nearest
            feat = feat[(1.0 - max_sim) > self.accumulate_thresh]
            if feat.numel() == 0:
                return
        if self.bank.numel() == 0:
            self.bank = feat.detach()
        else:
            self.bank = torch.cat([self.bank, feat.detach()], dim=0)

    def freeze(self) -> None:
        """Optionally compress via coreset, auto-calibrate temperature, then lock the bank."""
        if self._frozen:
            return
        if self.max_bank_size is not None and self.bank.shape[0] > self.max_bank_size:
            self.bank = _coreset_subsample(self.bank, self.max_bank_size)
        if self.auto_temperature:
            self._calibrate_temperature()
        self._frozen = True

    def _calibrate_temperature(self) -> None:
        """Auto-calibrate the Noisy-OR temperature β and store normal-score statistics.

        Solves for β such that a *typical* normal feature (90th-percentile of
        mean top-k cosine similarities within the bank itself) receives an
        anomaly score equal to ``calibration_target_score``.

        Also stores ``_bank_score_mean`` and ``_bank_score_std`` — the mean and
        std of the Noisy-OR scores of bank features scored against themselves.
        These are used by ``heatmap()`` for population-level normalization.
        """
        mem = self.bank
        if mem.shape[0] < self.min_calibration_bank_size:
            LOGGER.debug(
                "BackboneMemoryBank: skipping temperature calibration "
                "(bank_size=%d < min=%d)",
                mem.shape[0], self.min_calibration_bank_size,
            )
            return
        with torch.no_grad():
            n_mem = mem.shape[0]
            if n_mem > 512:
                sample_idx = torch.randperm(n_mem, device=mem.device)[:512]
            else:
                sample_idx = torch.arange(n_mem, device=mem.device)
            sample = mem[sample_idx]

            # Leave-one-out calibration: exclude exact self-match entries.
            # Including self-match drives scores unrealistically toward 0 and
            # causes inference-time normalization to saturate at 1.
            k = min(self.K, max(1, n_mem - 1))
            sim = sample @ mem.t()                       # [n, M]
            sim[torch.arange(sample.shape[0], device=sim.device), sample_idx] = -1.0
            topk_sim = sim.topk(k=k, dim=1).values      # [n, k]
            mean_topk = topk_sim.mean(dim=1)             # [n]
            s_typical = torch.quantile(mean_topk, 0.90).clamp(0.0, 1.0 - 1e-4)
            beta = -math.log(1.0 - self.calibration_target_score) / (1.0 - s_typical.item())
            beta = float(max(0.1, min(20.0, beta)))

            # Compute raw Noisy-OR scores for the sample using the calibrated β.
            # These represent the normal-population score distribution.
            psi = torch.exp(-beta * (1.0 - topk_sim))               # [n, k]
            raw = torch.exp(torch.log((1.0 - psi).clamp(1e-8)).mean(dim=-1))  # [n]
            self._bank_score_mean = float(raw.mean())
            self._bank_score_std  = float(raw.std().clamp(min=1e-6))

        old_temp = self.temperature
        self.temperature = beta
        LOGGER.info(
            "BackboneMemoryBank: auto-calibrated temperature %.3f \u2192 %.3f "
            "(s_90=%.4f, score_mean=%.4f, score_std=%.4f, bank_size=%d)",
            old_temp, beta, s_typical.item(),
            self._bank_score_mean, self._bank_score_std, mem.shape[0],
        )

    @property
    def is_ready(self) -> bool:
        """True when the bank has at least one feature and is frozen."""
        return self._frozen and self.bank.numel() > 0

    # ── inference ──────────────────────────────────────────────────────────

    def heatmap(self, feat: torch.Tensor) -> torch.Tensor:
        """Compute a dense anomaly heatmap from a backbone feature map.

        Scores are Noisy-OR KNN values, then normalised using the **bank
        population statistics** stored at calibration time.  This means:

        * Features that match the normal bank → score near 0.
        * Features that are genuinely out-of-distribution → score > 0.
        * A completely normal image → heatmap stays near 0 across all pixels
          (no false priors).

        Normalisation: ``clamp((raw - μ_bank) / (N_sigma × σ_bank), 0, 1)``
        where N_sigma=3 so that a feature +3σ above the bank mean scores 1.0.

        Args:
            feat: ``(B, C, H, W)`` backbone feature tensor.

        Returns:
            ``(B, 1, H, W)`` float heatmap in [0, 1].
        """
        if not self.is_ready:
            B = feat.shape[0]
            H, W = feat.shape[-2], feat.shape[-1]
            return torch.zeros(B, 1, H, W, device=feat.device, dtype=feat.dtype)

        B, C, H, W = feat.shape
        q = F.normalize(feat.permute(0, 2, 3, 1).reshape(-1, C).float(), p=2, dim=1)
        mem = self.bank.to(q.device)

        k = min(self.K, mem.shape[0])
        sim = q @ mem.t()                                          # (N, M)
        topk_cos = sim.topk(k, dim=-1).values                     # (N, k)
        psi = torch.exp(-self.temperature * (1.0 - topk_cos))    # (N, k)
        raw = torch.exp(torch.log((1.0 - psi).clamp(1e-8)).mean(dim=-1))  # (N,)

        # Population-level one-sided normalisation using bank statistics.
        # Normal features cluster at _bank_score_mean; anomalies lie above it.
        mu    = getattr(self, "_bank_score_mean", self.calibration_target_score)
        sigma = getattr(self, "_bank_score_std",  0.05)
        scores = ((raw - mu) / (3.0 * sigma)).clamp(0.0, 1.0)
        return scores.view(B, 1, H, W).to(feat.dtype)


if __name__ == "__main__":
    # Smoke test: render, fusion init=0 passthrough, fusion with learned beta bounded,
    # broadcast onto a C-channel PAN feature.
    B, H = 4, 80
    renderer = BboxMaskRenderer(mask_size=H, mode="rect")
    bboxes = torch.tensor(
        [
            [0.25, 0.25, 0.20, 0.20],
            [0.75, 0.75, 0.30, 0.10],
            [0.50, 0.50, 0.10, 0.10],
        ]
    )
    batch_idx = torch.tensor([0, 0, 2], dtype=torch.long)
    mask = renderer(bboxes, batch_idx, B)
    assert mask.shape == (B, 1, H, H)
    assert mask[1].sum().item() == 0.0  # image with no bboxes

    fusion = HeatmapBiasFusion(num_scales=3)
    # beta init=0 -> bias is exactly zero for every scale
    for s in range(3):
        bias = fusion(mask, s)
        assert bias.shape == (B, 1, H, H)
        assert bias.abs().max().item() == 0.0, f"scale {s} bias not zero at init"
    print("HeatmapBiasFusion init=0 passthrough OK.")

    # With beta set non-zero, output is bounded in [-beta, +beta]
    with torch.no_grad():
        fusion.beta.fill_(1.5)
    bias = fusion(mask, 0)
    assert bias.abs().max().item() <= 1.5 + 1e-6, "bias exceeded beta after tanh"
    print(f"HeatmapBiasFusion beta=1.5 bounded OK (max abs = {bias.abs().max().item():.4f}).")

    # Broadcast onto a C-channel PAN feature: (B, 1, H, W) + (B, C, H, W) -> (B, C, H, W).
    p = torch.randn(B, 256, H, H)
    p_fused = p + fusion(mask, 0)
    assert p_fused.shape == p.shape
    print(f"Broadcast OK: P (B,256,H,W) + bias (B,1,H,W) -> {tuple(p_fused.shape)}.")

    # Resize-per-scale smoke
    mask_p4 = F.interpolate(mask, size=(40, 40), mode="bilinear", align_corners=False)
    mask_p5 = F.interpolate(mask, size=(20, 20), mode="bilinear", align_corners=False)
    assert fusion(mask_p4, 1).shape == (B, 1, 40, 40)
    assert fusion(mask_p5, 2).shape == (B, 1, 20, 20)
    print("HeatmapBiasFusion multi-scale OK.")

    print("\nAll smoke tests passed.")
