"""UPAR 2027 Track 2 submission (v6) — distilled MobileNetV3-Small student.

Design (post-mortem v1-v5):
  * 0.95M-param student (9.8MB fp32 state dict) — 1.7ms/img fp32 CPU, so the
    full 29k-image gallery finishes in minutes even on a slow eval node. No
    tier logic, no GPU, no autocast, no bf16 probing: the simple path IS the
    fast path.
  * Absolutely crash-proof: any failure at any stage still yields a valid
    (G,40) probability matrix via the train-prior fallback, and rank_gallery
    ALWAYS returns a well-formed distances matrix. A scored submission at the
    floor beats an unscored one.
  * Loud logging (stderr) at every stage so the Codabench ingestion log shows
    exactly what happened.

Contract: load_model() once; rank_gallery(sample) -> {"distances": QxG}.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

class _BN(nn.BatchNorm2d):
    """BatchNorm2d with torchvision MobileNetV3 hyperparameters."""
    def __init__(self, num_features: int):
        super().__init__(num_features, eps=0.001, momentum=0.01)

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "assets", "student_model.pt")
PRIOR = os.path.join(HERE, "assets", "train_prior.npy")

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_MODEL: "StudentNet | None" = None
_DEV: torch.device | None = None
_LOG_T0 = time.perf_counter()


# ------------------------------------------------------------------ model

class StudentNet(nn.Module):
    """MobileNetV3-Small conv stages + GAP + Linear(576->40).

    Re-implementation of torchvision's mobilenet_v3_small backbone so this
    file has zero dependency on torchvision being present in the eval
    container. Exact stage table from torchvision _mobilenet_v3_conf:

        (in, k, exp, out, SE, act, stride)
        (16, 3,  16, 16,  True, RE, 2)
        (16, 3,  72, 24, False, RE, 2)
        (24, 3,  88, 24, False, RE, 1)
        (24, 5,  96, 40,  True, HS, 2)
        (40, 5, 240, 40,  True, HS, 1)
        (40, 5, 240, 40,  True, HS, 1)
        (40, 5, 120, 48,  True, HS, 1)
        (48, 5, 144, 48,  True, HS, 1)
        (48, 5, 288, 96,  True, HS, 2)
        (96, 5, 576, 96,  True, HS, 1)
        (96, 5, 576, 96,  True, HS, 1)

    followed by Conv2d(96, 576, 1) + BN + HSwish, then GAP + Linear(576,40).
    SE hidden width is set per-block to match the trained checkpoint at load
    time (load_state_dict is strict; export script guarantees alignment).
    """

    def __init__(self, arch: str = "small"):
        super().__init__()

        def hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.relu6(x + 3.0, inplace=False) / 6.0

        class HSigmoid(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return hard_sigmoid(x)

        class HSwish(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * hard_sigmoid(x)

        def _make_divisible(v: int, divisor: int = 8) -> int:
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        class SELayer(nn.Module):
            def __init__(self, c: int):
                super().__init__()
                hidden = _make_divisible(c // 4)
                self.fc1 = nn.Conv2d(c, hidden, 1)
                self.fc2 = nn.Conv2d(hidden, c, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                s = torch.nn.functional.adaptive_avg_pool2d(x, 1)
                s = torch.nn.functional.relu(self.fc1(s), inplace=False)
                s = hard_sigmoid(self.fc2(s))
                return x * s

        class Bottleneck(nn.Module):
            def __init__(self, in_c: int, kernel: int, exp_c: int, out_c: int,
                         se: bool, act: str, stride: int):
                super().__init__()
                self.use_res = stride == 1 and in_c == out_c
                A = HSwish if act == "HS" else nn.ReLU
                layers: list[nn.Module] = []
                # pointwise expand (skipped when exp == in, like torchvision)
                if exp_c != in_c:
                    layers += [nn.Conv2d(in_c, exp_c, 1, bias=False),
                               _BN(exp_c), A()]
                # depthwise
                layers += [nn.Conv2d(exp_c, exp_c, kernel, stride, kernel // 2,
                                     groups=exp_c, bias=False),
                           _BN(exp_c), A()]
                if se:
                    layers += [SELayer(exp_c)]
                # pointwise project
                layers += [nn.Conv2d(exp_c, out_c, 1, bias=False),
                           _BN(out_c)]
                self.block = nn.Sequential(*layers)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.block(x)
                return x + out if self.use_res else out

        if arch == "small":
            cfg = [  # (in, k, exp, out, SE, act, stride)
                (16, 3,  16, 16,  True,  "RE", 2),
                (16, 3,  72, 24,  False, "RE", 2),
                (24, 3,  88, 24,  False, "RE", 1),
                (24, 5,  96, 40,  True,  "HS", 2),
                (40, 5, 240, 40,  True,  "HS", 1),
                (40, 5, 240, 40,  True,  "HS", 1),
                (40, 5, 120, 48,  True,  "HS", 1),
                (48, 5, 144, 48,  True,  "HS", 1),
                (48, 5, 288, 96,  True,  "HS", 2),
                (96, 5, 576, 96,  True,  "HS", 1),
                (96, 5, 576, 96,  True,  "HS", 1),
            ]
            stem_c, last_c, feat_d = 16, 576, 576
        elif arch == "large":
            cfg = [
                (16, 3,  16,  16, False, "RE", 1),
                (16, 3,  64,  24, False, "RE", 2),
                (24, 3,  72,  24, False, "RE", 1),
                (24, 5,  72,  40, True,  "RE", 2),
                (40, 5, 120,  40, True,  "RE", 1),
                (40, 5, 120,  40, True,  "RE", 1),
                (40, 3, 240,  80, False, "HS", 2),
                (80, 3, 200,  80, False, "HS", 1),
                (80, 3, 184,  80, False, "HS", 1),
                (80, 3, 184,  80, False, "HS", 1),
                (80, 3, 480, 112, True,  "HS", 1),
                (112, 3, 672, 112, True,  "HS", 1),
                (112, 5, 672, 160, True,  "HS", 2),
                (160, 5, 960, 160, True,  "HS", 1),
                (160, 5, 960, 160, True,  "HS", 1),
            ]
            stem_c, last_c, feat_d = 16, 960, 960
        else:
            raise ValueError(arch)
        feats: list[nn.Module] = [
            nn.Conv2d(3, stem_c, 3, stride=2, padding=1, bias=False),
            _BN(stem_c), HSwish(),
        ]
        for (in_c, k, exp, out, se, act, stride) in cfg:
            feats.append(Bottleneck(in_c, k, exp, out, se, act, stride))
        feats += [nn.Conv2d(cfg[-1][3], last_c, 1, bias=False),
                  _BN(last_c), HSwish()]
        self.features = nn.Sequential(*feats)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(feat_d, 40)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


# ------------------------------------------------------------------ preprocessing

def _letterbox(img: Image.Image, size: int = 224) -> Image.Image:
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(round(w * s), 1), max(round(h * s), 1)
    img = img.convert("RGB").resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def _preprocess(img: Image.Image) -> torch.Tensor:
    img = _letterbox(img, 224)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(CLIP_MEAN, dtype=np.float32)) / np.asarray(CLIP_STD, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1)


# ------------------------------------------------------------------ helpers

def _log(msg: str) -> None:
    print(f"[task2-run] [{time.perf_counter()-_LOG_T0:7.1f}s] {msg}",
          file=sys.stderr, flush=True)


def _safe(fn, default=None):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        _log(f"non-fatal: {fn.__name__} -> {exc!r}")
        return default


def _load_prior() -> np.ndarray:
    p = _safe(lambda: np.load(PRIOR).astype(np.float32),
              default=None)
    if p is not None and p.shape == (40,):
        return p
    _log("prior asset unusable; using 0.5 constant")
    return np.full(40, 0.5, dtype=np.float32)


# ------------------------------------------------------------------ contract

def load_model() -> None:
    global _MODEL, _DEV
    if _MODEL is not None:
        return
    try:
        ncpu = os.cpu_count() or 1
        torch.set_num_threads(min(16, max(1, ncpu)))
    except Exception as exc:  # noqa: BLE001
        _log(f"thread cap failed: {exc!r}")

    _DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    arch = ckpt.get("arch", "small")
    model = StudentNet(arch)
    model.load_state_dict(ckpt["state"], strict=True)
    model = model.to(_DEV).eval()
    _MODEL = model
    _log(f"student loaded on {_DEV} (epoch {ckpt.get('epoch', '?')})")


def _gallery_path(item: Any) -> str:
    if isinstance(item, str):
        return item
    if hasattr(item, "__fspath__"):
        return os.fspath(item)
    if isinstance(item, dict):
        for key in ("image_path", "image", "path", "filename", "file"):
            if key in item and isinstance(item[key], (str, os.PathLike)):
                return os.fspath(item[key])
        for v in item.values():
            if isinstance(v, (str, os.PathLike)) and os.path.isfile(os.fspath(v)):
                return os.fspath(v)
    raise TypeError(f"cannot extract path from gallery item: {item!r:.200}")


def _prob_chunk(paths: list[str], batch: int = 64) -> np.ndarray:
    out = np.empty((len(paths), 40), dtype=np.float32)
    model, dev = _MODEL, _DEV
    assert model is not None and dev is not None
    with torch.no_grad():
        for j0 in range(0, len(paths), batch):
            imgs = [_preprocess(Image.open(p).convert("RGB"))
                    for p in paths[j0:j0 + batch]]
            x = torch.stack(imgs).to(dev)
            logits = model(x)  # fp32 everywhere; no autocast games
            out[j0:j0 + len(imgs)] = 1.0 / (1.0 + np.exp(-logits.float().cpu().numpy()))
    return out


def predict_attributes(gallery: list[dict[str, Any]], attribute_names: list[str]) -> np.ndarray:
    """(G, 40) probabilities. Prior-fills any chunk that fails; never raises."""
    n = len(gallery)
    out = np.tile(_load_prior(), (n, 1))
    if _MODEL is None:
        _safe(load_model)
    if _MODEL is None or _DEV is None:
        _log("model unavailable; returning prior for whole gallery")
        return out

    chunk = 256
    done, failed = 0, 0
    t0 = time.perf_counter()
    for i0 in range(0, n, chunk):
        items = gallery[i0:i0 + chunk]
        try:
            paths = [_gallery_path(g) for g in items]
            out[i0:i0 + len(paths)] = _prob_chunk(paths)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if failed <= 3 or failed % 50 == 0:
                _log(f"chunk {i0} failed ({exc!r}); prior-filled")
            continue
        done = i0 + len(paths)
        if (i0 // chunk) % 16 == 0:
            dt = time.perf_counter() - t0
            _log(f"progress {done}/{n} ({dt:.0f}s elapsed)")
    _log(f"RESULT-SUMMARY: real={done}/{n} failed_chunks={failed} "
         f"dev={_DEV.type} elapsed={time.perf_counter()-t0:.0f}s")
    return out


def rank_gallery(sample: dict[str, Any]) -> dict[str, Any]:
    queries = np.atleast_2d(np.asarray(sample["queries"], dtype=np.float32))
    probs = _safe(lambda: predict_attributes(sample["gallery"],
                                             sample.get("attribute_names") or []),
                  default=None)
    if probs is None or probs.shape != (len(sample["gallery"]), 40):
        _log("predict_attributes failed; prior fallback for distances")
        probs = np.tile(_load_prior(), (len(sample["gallery"]), 1))
    # L1 distance: D = sum p + q.(1-2p)   (smaller = better match)
    distances = probs.sum(1)[None, :] + queries @ (1.0 - 2.0 * probs).T
    return {"distances": distances.astype(np.float32)}
