"""Data loading for UPAR 2027 (shared by training and both submission wrappers).

CSV facts (verified):
  gt.csv       header '# image,<40 attribute names>'; rows '<relpath>,0/1*40'
  ids.csv      NO header, CRLF endings: '<relpath>,<query row index>'
  queries.csv  header '# <40 attribute names>'; rows '0/1*40'
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# CLIP ViT-B/16 normalization
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# 40 attributes in canonical challenge order
ATTRIBUTE_NAMES = [
    "Age-Young", "Age-Adult", "Age-Old", "Gender-Female",
    "Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald",
    "UpperBody-Length-Short",
    "UpperBody-Color-Black", "UpperBody-Color-Blue", "UpperBody-Color-Brown",
    "UpperBody-Color-Green", "UpperBody-Color-Grey", "UpperBody-Color-Orange",
    "UpperBody-Color-Pink", "UpperBody-Color-Purple", "UpperBody-Color-Red",
    "UpperBody-Color-White", "UpperBody-Color-Yellow", "UpperBody-Color-Other",
    "LowerBody-Length-Short",
    "LowerBody-Color-Black", "LowerBody-Color-Blue", "LowerBody-Color-Brown",
    "LowerBody-Color-Green", "LowerBody-Color-Grey", "LowerBody-Color-Orange",
    "LowerBody-Color-Pink", "LowerBody-Color-Purple", "LowerBody-Color-Red",
    "LowerBody-Color-White", "LowerBody-Color-Yellow", "LowerBody-Color-Other",
    "LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress",
    "Accessory-Backpack", "Accessory-Bag", "Accessory-Glasses-Normal",
    "Accessory-Glasses-Sun", "Accessory-Hat",
]


def _open_csv(path: str | Path):
    """Open an annotation CSV, failing fast with a clear message if unreadable."""
    try:
        return open(path, newline="", encoding="utf-8")
    except OSError as err:
        raise RuntimeError(f"cannot read annotation CSV {path}: {err}") from err


def load_gt(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Returns (image relative paths in row order, int8 [N,40] labels)."""
    names, rows = [], []
    with _open_csv(path) as fh:
        r = csv.reader(fh)
        header = next(r)
        assert header[0] == "# image", f"unexpected gt header: {header[0]!r}"
        names = header[1:]
        for row in r:
            if not row:
                continue
            rows.append(row)
    paths = [row[0] for row in rows]
    labels = np.asarray([[int(x) for x in row[1:]] for row in rows], dtype=np.int8)
    assert names == ATTRIBUTE_NAMES, "attribute order mismatch"
    return paths, labels


def load_ids(path: str | Path) -> dict[str, int]:
    """ids.csv: no header, CRLF. Map relpath -> query row index."""
    ids: dict[str, int] = {}
    with _open_csv(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p, q = line.rsplit(",", 1)
            ids[p] = int(q)
    return ids


def load_queries(path: str | Path) -> np.ndarray:
    """queries.csv: header '# <attrs>'; returns int8 [Q,40]."""
    qs = []
    with _open_csv(path) as fh:
        next(fh)  # header
        for line in fh:
            line = line.strip()
            if not line:
                continue
            qs.append([int(x) for x in line.split(",")])
    return np.asarray(qs, dtype=np.int8)


def letterbox(img: Image.Image, size: int = 224) -> Image.Image:
    """Aspect-preserving resize with grey padding - suits full-body pedestrian crops."""
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(round(w * s), 1), max(round(h * s), 1)
    img = img.convert("RGB").resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def preprocess_tensor(img: Image.Image, size: int = 224) -> torch.Tensor:
    """Letterbox -> CHW float -> CLIP normalize. Returns [3,size,size]."""
    img = letterbox(img, size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(CLIP_MEAN, dtype=np.float32)) / np.asarray(CLIP_STD, dtype=np.float32)
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t


class UPARDataset(Dataset):
    def __init__(self, root: str | Path, gt_csv: str | Path, train: bool = True):
        self.root = Path(root)
        self.paths, self.labels = load_gt(gt_csv)
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.root / self.paths[i])
        t = preprocess_tensor(img)
        y = torch.from_numpy(self.labels[i].astype(np.float32))
        return t, y


class TrainTransform:
    """Light augmentation for pedestrian crops (applied on the letterboxed tensor)."""

    def __init__(self, size: int = 224, p_erase: float = 0.25):
        self.size = size
        self.p_erase = p_erase

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # random horizontal flip in PIL space
        if np.random.rand() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        t = preprocess_tensor(img, self.size)
        # random erasing (grey square) - robustness to occlusions in surveillance
        if np.random.rand() < self.p_erase:
            _, h, w = t.shape
            ew, eh = np.random.randint(w // 8, w // 3), np.random.randint(h // 8, h // 3)
            x0, y0 = np.random.randint(0, w - ew), np.random.randint(0, h - eh)
            pad = torch.tensor(0.0, dtype=t.dtype)  # grey==normalized 0
            t[:, y0:y0 + eh, x0:x0 + ew] = pad
        return t


class UPARTrainDataset(Dataset):
    def __init__(self, root: str | Path, gt_csv: str | Path):
        self.root = Path(root)
        self.paths, self.labels = load_gt(gt_csv)
        self.tf = TrainTransform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.root / self.paths[i])
        t = self.tf(img)
        y = torch.from_numpy(self.labels[i].astype(np.float32))
        return t, y


def positive_rates(labels: np.ndarray) -> np.ndarray:
    return labels.astype(np.float64).mean(0)
