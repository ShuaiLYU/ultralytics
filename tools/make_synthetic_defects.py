"""REB-style synthetic defect generator for YOLOA detection data.

Takes normal/good images and pastes a synthetic defect onto each, producing a
defect image plus a YOLO **detect** label (bbox, class 0 = anomaly). The output
is meant to be dropped into the training set as an ordinary labeled defect
source — no model / training-code changes.

Pipeline (per image), mirroring REB's DefectMaker:
  1. shape  : a localized binary mask (perlin blob | bezier blob | scar)
  2. fill   : appearance inside the mask (cutpaste from a donor normal | noise)
  3. blend  : composite fill into the image (poisson seamlessClone | alpha)
  4. label  : tight bbox around the mask -> normalized xywh

Usage (smoke test, writes a viz grid):
    python make_synthetic_defects.py --smoke

Usage (batch generate into a YOLO source dir):
    python make_synthetic_defects.py \
        --src-list good_images.txt \
        --out /path/to/synthetic_defect \
        --n 30000
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Perlin noise (DRAEM/REB formulation)
# --------------------------------------------------------------------------- #


def _lerp(a, b, w):
    return a + w * (b - a)


def _rand_perlin_2d(shape, res, rng):
    """Generate a 2D perlin noise field in [-1, 1]; res must divide shape."""
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
    angles = 2 * math.pi * rng.random((res[0] + 1, res[1] + 1))
    gradients = np.dstack((np.cos(angles), np.sin(angles)))

    def tile(s0, s1):
        g = gradients[s0[0] : s0[1], s1[0] : s1[1]]
        return np.repeat(np.repeat(g, d[0], 0), d[1], 1)

    def dot(grad, shift):
        gx = grid[: shape[0], : shape[1], 0] + shift[0]
        gy = grid[: shape[0], : shape[1], 1] + shift[1]
        return (np.dstack((gx, gy)) * grad[: shape[0], : shape[1]]).sum(-1)

    n00 = dot(tile([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile([0, -1], [1, None]), [0, -1])
    n11 = dot(tile([1, None], [1, None]), [-1, -1])
    t = 6 * grid**5 - 15 * grid**4 + 10 * grid**3
    n0 = _lerp(n00, n10, t[: shape[0], : shape[1], 0])
    n1 = _lerp(n01, n11, t[: shape[0], : shape[1], 0])
    return math.sqrt(2) * _lerp(n0, n1, t[: shape[0], : shape[1], 1])


# --------------------------------------------------------------------------- #
# Shape makers -> binary mask (uint8 0/255) sized (h, w)
# --------------------------------------------------------------------------- #


def perlin_mask(h, w, rng, thr_range=(0.4, 0.7)):
    """Organic blob from thresholded perlin noise, restricted to largest blob."""
    side = 256
    sx = 2 ** rng.integers(1, 4)  # scale in {2,4,8}
    sy = 2 ** rng.integers(1, 4)
    noise = _rand_perlin_2d((side, side), (int(sx), int(sy)), rng)
    noise = (noise - noise.min()) / (np.ptp(noise) + 1e-8)
    thr = rng.uniform(*thr_range)
    m = (noise > thr).astype(np.uint8)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return _largest_component(m * 255)


def bezier_mask(h, w, rng, n_pts=None, rough=0.35):
    """Smooth closed blob: random radial points around a center, filled + blurred."""
    n_pts = n_pts or int(rng.integers(6, 14))
    cx, cy = w / 2, h / 2
    base_r = min(h, w) * rng.uniform(0.28, 0.45)
    angles = np.sort(rng.uniform(0, 2 * math.pi, n_pts))
    radii = base_r * (1 + rng.uniform(-rough, rough, n_pts))
    pts = np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], 1)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    k = max(3, (min(h, w) // 12) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return (mask > 127).astype(np.uint8) * 255


def scar_mask(h, w, rng):
    """Thin rotated rectangle — scratch-like defect."""
    mask = np.zeros((h, w), np.uint8)
    sw = int(rng.integers(2, max(3, w // 8)))
    sh = int(rng.integers(h // 3, h))
    angle = rng.uniform(-45, 45)
    rect = ((w / 2, h / 2), (sw, sh), angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(mask, [box], 255)
    return mask


def _largest_component(mask):
    """Keep only the largest connected component; empty -> small center blob."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        h, w = mask.shape
        cv2.circle(mask, (w // 2, h // 2), max(2, min(h, w) // 4), 255, -1)
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((lab == biggest).astype(np.uint8)) * 255


SHAPE_FNS = {"perlin": perlin_mask, "bezier": bezier_mask, "scar": scar_mask}


# --------------------------------------------------------------------------- #
# Foreground extraction — keep defects on the object, not the background
# --------------------------------------------------------------------------- #


def saliency_classical(img):
    """Soft saliency (uint8 0..255) via border-background color distance."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = img.shape[:2]
    b = max(2, min(h, w) // 25)
    border = np.concatenate(
        [lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3), lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)]
    )
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    return (dist / (dist.max() + 1e-6) * 255).astype(np.uint8)


_U2NET = None
U2NET_ONNX = str(Path.home() / ".u2net" / "u2netp.onnx")  # 4.4MB lite salient-object net


def saliency_u2net(img):
    """Soft saliency (uint8 0..255) from U2Netp run directly via onnxruntime."""
    global _U2NET
    if _U2NET is None:
        import onnxruntime as ort

        _U2NET = ort.InferenceSession(U2NET_ONNX, providers=["CPUExecutionProvider"])
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    inp = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_AREA)
    inp = inp / (inp.max() + 1e-6)
    inp = (inp - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    inp = inp.transpose(2, 0, 1)[None].astype(np.float32)
    pred = _U2NET.run(None, {_U2NET.get_inputs()[0].name: inp})[0][0, 0]
    pred = (pred - pred.min()) / (np.ptp(pred) + 1e-6)
    return cv2.resize((pred * 255).astype(np.uint8), (img.shape[1], img.shape[0]))


SALIENCY_FNS = {"classical": saliency_classical, "u2net": saliency_u2net}


# --------------------------------------------------------------------------- #
# Per-source taxonomy: object (saliency-constrained) vs texture (place anywhere)
# --------------------------------------------------------------------------- #

# Sources whose "good" set is unreliable (unlabeled real defects) or whose
# defects span the whole object (objectness, not localized) — not used as canvases.
EXCLUDE_SOURCES = {"carscratch", "apple-defect"}

# Explicit object/texture overrides; everything else falls to the rule below.
TEXTURE_SOURCES = {"magnetictiledefects", "tianchifabirc"}
OBJECT_SOURCES = {"visa", "casedefec"}


def source_type(name):
    """Classify a source prefix -> 'object' | 'texture' | 'exclude'."""
    if name in EXCLUDE_SOURCES:
        return "exclude"
    if name in TEXTURE_SOURCES or name.startswith("dagm-") or "fabric" in name:
        return "texture"
    if name in OBJECT_SOURCES or name.startswith("realiad-") or name.startswith("goodsad-"):
        return "object"
    return "object"  # default: assume object (saliency, with per-image fallback)


def source_of(path):
    """Extract the source prefix from a merged filename (`<source>__...`)."""
    return Path(path).name.split("__")[0]


def foreground_mask(img, method="classical", min_frac=0.03, max_frac=0.92, return_soft=False):
    """Binarize a saliency map into an object mask, with morphological cleanup.

    Returns a uint8 0/255 mask, or None when no clear object is present
    (texture / near-uniform image) — caller then places defects anywhere.
    `return_soft=True` also returns the raw soft saliency map (for viz).
    """
    h, w = img.shape[:2]
    soft = SALIENCY_FNS[method](img)
    _, m = cv2.threshold(soft, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, (min(h, w) // 30) | 1),) * 2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    ff = m.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    m = m | cv2.bitwise_not(ff)  # fill interior holes
    m = _largest_component(m)
    frac = (m > 0).mean()
    mask = None if (frac < min_frac or frac > max_frac) else m
    return (mask, soft) if return_soft else mask


# --------------------------------------------------------------------------- #
# Fill makers -> BGR fill patch sized (h, w)
# --------------------------------------------------------------------------- #


def cutpaste_fill(h, w, rng, donor):
    """Random crop from a donor image, resized to (h, w)."""
    dh, dw = donor.shape[:2]
    cw = int(min(dw, max(8, w * rng.uniform(0.8, 1.5))))
    ch = int(min(dh, max(8, h * rng.uniform(0.8, 1.5))))
    x = int(rng.integers(0, max(1, dw - cw)))
    y = int(rng.integers(0, max(1, dh - ch)))
    crop = donor[y : y + ch, x : x + cw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def noise_fill(h, w, rng):
    """Structured noise around a random base color."""
    mean = rng.integers(0, 256, 3)
    fluct = rng.integers(20, 90)
    small = rng.integers(
        np.clip(mean - fluct, 0, 255), np.clip(mean + fluct + 1, 1, 256), (h // 4 + 1, w // 4 + 1, 3)
    ).astype(np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def color_jitter(patch, rng):
    """Brightness / contrast jitter so cut patches don't match the source exactly."""
    a = rng.uniform(0.7, 1.3)
    b = rng.uniform(-30, 30)
    return np.clip(patch.astype(np.float32) * a + b, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# DefectMaker
# --------------------------------------------------------------------------- #


class DefectMaker:
    """Composite one synthetic defect onto a normal image; return (image, bbox)."""

    def __init__(self, seed=0, area_range=(0.005, 0.08), aspect_range=(0.3, 3.3)):
        self.rng = np.random.default_rng(seed)
        self.area_range = area_range
        self.aspect_range = aspect_range

    def _region(self, H, W, fg=None):
        """Pick a defect region box (x, y, w, h) by area ratio + aspect.

        When `fg` is given, the box is centered on a random foreground pixel so
        the defect lands on the object, not the background.
        """
        area = self.rng.uniform(*self.area_range) * H * W
        aspect = math.exp(self.rng.uniform(math.log(self.aspect_range[0]), math.log(self.aspect_range[1])))
        rw = int(np.clip(math.sqrt(area * aspect), 8, W - 1))
        rh = int(np.clip(math.sqrt(area / aspect), 8, H - 1))
        if fg is not None:
            ys, xs = np.where(fg > 0)
            i = int(self.rng.integers(0, len(xs)))
            x = int(np.clip(xs[i] - rw // 2, 0, max(0, W - rw)))
            y = int(np.clip(ys[i] - rh // 2, 0, max(0, H - rh)))
        else:
            x = int(self.rng.integers(0, max(1, W - rw)))
            y = int(self.rng.integers(0, max(1, H - rh)))
        return x, y, rw, rh

    def make(self, image, donor=None, fg=None):
        H, W = image.shape[:2]
        x, y, rw, rh = self._region(H, W, fg)

        shape = self.rng.choice(list(SHAPE_FNS))
        local_mask = SHAPE_FNS[shape](rh, rw, self.rng)

        if donor is not None and self.rng.random() < 0.7:
            fill = color_jitter(cutpaste_fill(rh, rw, self.rng, donor), self.rng)
        else:
            fill = noise_fill(rh, rw, self.rng)

        full_mask = np.zeros((H, W), np.uint8)
        full_mask[y : y + rh, x : x + rw] = local_mask
        if fg is not None:  # clip defect to the object (allow a small spill)
            kf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, (min(H, W) // 50) | 1),) * 2)
            full_mask = cv2.bitwise_and(full_mask, cv2.dilate(fg, kf))
        if full_mask.max() == 0:
            return None

        full_fill = image.copy()
        full_fill[y : y + rh, x : x + rw] = fill

        beta = self.rng.uniform(0.6, 1.0)  # opacity of the defect
        use_poisson = shape != "scar" and self.rng.random() < 0.5
        out = self._blend(image, full_fill, full_mask, beta, use_poisson)

        ys, xs = np.where(full_mask > 0)
        bbox = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)  # x1,y1,x2,y2
        return out, bbox

    def _blend(self, image, fill, mask, beta, use_poisson):
        if use_poisson:
            x, y, w, h = cv2.boundingRect(mask)
            center = (x + w // 2, y + h // 2)
            try:
                return cv2.seamlessClone(fill, image, mask, center, cv2.NORMAL_CLONE)
            except cv2.error:
                pass
        # feathered alpha blend
        k = max(3, (min(image.shape[:2]) // 60) | 1)
        soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (k, k), 0) * beta
        soft = soft[..., None]
        return (image * (1 - soft) + fill * soft).astype(np.uint8)


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(src_list, src_dir):
    if src_list:
        return [Path(p) for p in Path(src_list).read_text().splitlines() if p.strip()]
    return sorted(p for p in Path(src_dir).rglob("*") if p.suffix.lower() in IMG_EXT)


def bbox_to_yolo(bbox, W, H):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


# --------------------------------------------------------------------------- #
# Smoke test — read local MVTec good images, draw a viz grid
# --------------------------------------------------------------------------- #

MVTEC = Path("/Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/MVTEC/mvtec_anomaly_detection")


def _overlay(img, mask, color=(0, 200, 255), a=0.45):
    if mask is None:
        return img.copy()
    out = img.copy().astype(np.float32)
    sel = mask > 0
    out[sel] = out[sel] * (1 - a) + np.array(color, np.float32) * a
    return out.astype(np.uint8)


def v5_demo(method="classical"):
    """v5 good-image sample, 4 cols: orig | soft saliency | foreground | defect+bbox."""
    src_dir = Path(__file__).parent / "v5_good_sample"
    imgs = list_images(None, str(src_dir))
    rng = np.random.default_rng(0)
    maker = DefectMaker(seed=2)
    rows = []
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (256, 256))
        fg, soft = foreground_mask(img, method=method, return_soft=True)
        donor = cv2.resize(cv2.imread(str(imgs[int(rng.integers(0, len(imgs)))])), (256, 256))
        res = maker.make(img, donor, fg)
        col_sal = cv2.applyColorMap(soft, cv2.COLORMAP_JET)
        col_fg = _overlay(img, fg)
        if res is None:
            col_def = img.copy()
        else:
            col_def, (x1, y1, x2, y2) = res
            cv2.rectangle(col_def, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(col_fg, "FG" if fg is not None else "any", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, p.name.split("__")[0][:14], (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        rows.append(np.hstack([img, col_sal, col_fg, col_def]))
    grid = np.vstack(rows)
    out_path = Path(__file__).parent / f"synthetic_defect_v5_{method}.jpg"
    cv2.imwrite(str(out_path), grid)
    print(f"wrote {out_path}  ({grid.shape[1]}x{grid.shape[0]}, {len(rows)} imgs)")


def gallery():
    """Contact sheet: one good image per source, labeled with proposed type."""
    gdir = Path(__file__).parent / "v5_gallery"
    files = sorted(gdir.glob("*@@*"))
    color = {"object": (0, 200, 0), "texture": (255, 150, 0), "exclude": (0, 0, 255)}
    cells = []
    counts = {"object": 0, "texture": 0, "exclude": 0}
    for f in files:
        src = f.name.split("@@")[0]
        t = source_type(src)
        counts[t] += 1
        im = cv2.resize(cv2.imread(str(f)), (224, 224))
        cv2.rectangle(im, (0, 0), (223, 223), color[t], 6)
        cv2.rectangle(im, (0, 0), (223, 28), color[t], -1)
        cv2.putText(im, f"{src[:20]} [{t[:3]}]", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cells.append(im)
    cols = 7
    while len(cells) % cols:
        cells.append(np.zeros((224, 224, 3), np.uint8))
    rows = [np.hstack(cells[i : i + cols]) for i in range(0, len(cells), cols)]
    grid = np.vstack(rows)
    out_path = Path(__file__).parent / "v5_source_gallery.jpg"
    cv2.imwrite(str(out_path), grid)
    print(f"wrote {out_path}  ({grid.shape[1]}x{grid.shape[0]})  counts={counts}")


def smoke():
    cats = ["bottle", "carpet", "hazelnut", "wood", "transistor", "leather", "tile", "screw"]
    rng = np.random.default_rng(0)
    maker = DefectMaker(seed=1)
    rows = []
    for cat in cats:
        good = sorted((MVTEC / cat / "train" / "good").glob("*.png"))
        if not good:
            continue
        cells = []
        for _ in range(4):
            img = cv2.imread(str(good[int(rng.integers(0, len(good)))]))
            img = cv2.resize(img, (256, 256))
            donor = cv2.imread(str(good[int(rng.integers(0, len(good)))]))
            donor = cv2.resize(donor, (256, 256))
            res = maker.make(img, donor)
            if res is None:
                cells.append(img)
                continue
            out, (x1, y1, x2, y2) = res
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(out, cat, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cells.append(out)
        rows.append(np.hstack(cells))
    grid = np.vstack(rows)
    out_path = Path(__file__).parent / "synthetic_defect_smoke.jpg"
    cv2.imwrite(str(out_path), grid)
    print(f"wrote {out_path}  ({grid.shape[1]}x{grid.shape[0]})")


# --------------------------------------------------------------------------- #
# Batch generation
# --------------------------------------------------------------------------- #


def generate(args):
    """Generate `args.n` synthetic defect images, source-type-aware.

    object sources -> saliency-constrained placement; texture sources -> anywhere;
    excluded sources -> never used as a canvas. Donors are drawn from the same
    source so the cut texture matches the material.
    """
    srcs = list_images(args.src_list, args.src_dir)
    if not srcs:
        raise SystemExit("no source images found")
    by_src = {}
    for p in srcs:
        t = source_type(source_of(p))
        if t != "exclude":
            by_src.setdefault(source_of(p), []).append(p)
    names = sorted(by_src)
    files = [p for s in names for p in by_src[s]]  # natural (per-good-count) weighting
    print(f"usable canvases: {len(files)} from {len(names)} sources "
          f"(object={sum(source_type(s)=='object' for s in names)}, "
          f"texture={sum(source_type(s)=='texture' for s in names)})")

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    maker = DefectMaker(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    n_done = tries = 0
    while n_done < args.n and tries < args.n * 5:
        tries += 1
        base = files[int(rng.integers(0, len(files)))]
        src = source_of(base)
        img = cv2.imread(str(base))
        if img is None:
            continue
        fg = foreground_mask(img, method=args.method) if source_type(src) == "object" else None
        peers = by_src[src]
        donor = cv2.imread(str(peers[int(rng.integers(0, len(peers)))]))
        res = maker.make(img, donor, fg)
        if res is None:
            continue
        defect, bbox = res
        H, W = defect.shape[:2]
        stem = f"syn__{src}__{args.tag}{n_done:07d}"
        cv2.imwrite(str(out / "images" / f"{stem}.jpg"), defect)
        (out / "labels" / f"{stem}.txt").write_text(bbox_to_yolo(bbox, W, H) + "\n")
        n_done += 1
        if n_done % 2000 == 0:
            print(f"{n_done}/{args.n}")
    print(f"done: {n_done} synthetic defect images -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run local MVTec viz smoke test")
    ap.add_argument("--v5demo", action="store_true", help="run viz on the pulled v5 good-image sample")
    ap.add_argument("--gallery", action="store_true", help="render labeled per-source contact sheet")
    ap.add_argument("--no-fg", action="store_true", help="disable foreground constraint (place anywhere)")
    ap.add_argument("--method", choices=["classical", "u2net"], default="classical", help="saliency backend")
    ap.add_argument("--src-list", help="text file of source (good) image paths")
    ap.add_argument("--src-dir", help="directory to recursively scan for source images")
    ap.add_argument("--out", help="output dir (creates images/ and labels/)")
    ap.add_argument("--n", type=int, default=30000, help="number of synthetic images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="filename prefix tag (use distinct tags for parallel shards)")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    elif args.gallery:
        gallery()
    elif args.v5demo:
        v5_demo(args.method)
    else:
        if not args.out or not (args.src_list or args.src_dir):
            raise SystemExit("need --out and one of --src-list / --src-dir")
        generate(args)


if __name__ == "__main__":
    main()
