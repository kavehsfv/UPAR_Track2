"""Distill the fine-tuned CLIP ViT-B/16 PAR teacher into a tiny CPU-friendly student.

Motivation (task2 on-site post-mortem, 2026-10-16):
  The ViT-B/16 submission (345MB, ~110ms/img fp32 CPU) cannot complete real
  inference in the Track 2 ingestion environment; v4 silently prior-filled and
  v5 (loud) crashed. A ~2M-param student at ~5-10ms/img makes real inference
  feasible there and removes the size/speed attack surface entirely.

Teacher:  runs/clip_vitb16/ep4.pt (T2-optimal: val mADM 0.6716)
Student:  MobileNetV3-Small (ImageNet init) + Linear(576->40), 224px letterbox.
Loss:     BCE(student_logits, teacher_probs) + 0.5 * BCE(student_logits, hard GT)
Eval:     full-val track2 metrics each epoch (mADM model selection).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import data as D
import metrics as M
from vit import PARModel, VisionTransformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    _HERE = Path(__file__).resolve().parent
    p.add_argument("--teacher-ckpt", type=Path,
                   default=_HERE / "runs" / "clip_vitb16" / "ep4.pt")
    p.add_argument("--out", type=Path, default=_HERE / "runs" / "distill_mbv3s")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--gpu-throttle", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arch", choices=["small", "large"], default="small")
    return p.parse_args()


class StudentNet(nn.Module):
    """MobileNetV3 backbone (ImageNet) + GAP + Linear(feat_dim->40)."""

    def __init__(self, arch: str = "small"):
        super().__init__()
        if arch == "small":
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            feat = 576
        elif arch == "large":
            from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
            net = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            feat = 960
        else:
            raise ValueError(arch)
        self.features = net.features
        self.avgpool = net.avgpool          # AdaptiveAvgPool2d(1)
        self.head = nn.Linear(feat, 40)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


def build_student(arch: str = "small") -> nn.Module:
    return StudentNet(arch)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    # ---------------- teacher -------------------------------------------
    teacher = PARModel(VisionTransformer(), num_attributes=40)
    ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=True)
    teacher.visual.load_state_dict(ckpt["visual"], strict=True)
    teacher.head.load_state_dict(ckpt["head"], strict=True)
    teacher = teacher.to(dev).eval()
    print(f"teacher loaded from {args.teacher_ckpt}")

    # ---------------- student -------------------------------------------
    student = build_student(args.arch).to(dev)

    # ---------------- data ------------------------------------------------
    DATA = Path(__file__).resolve().parent.parent / "data"
    train_ds = D.UPARTrainDataset(DATA, DATA / "annotations/task1/train/gt.csv")
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        pin_memory=True, persistent_workers=args.workers > 0)
    val_paths, val_gt = D.load_gt(DATA / "annotations/task1/val/gt.csv")
    print(f"train {len(train_ds)}  val {len(val_paths)}")

    Q = D.load_queries(DATA / "annotations/task2/val/queries.csv")
    ids_map = D.load_ids(DATA / "annotations/task2/val/ids.csv")
    qids = np.asarray([ids_map[p] for p in val_paths], dtype=np.int64)
    Qf = Q[qids].astype(np.float32)   # (Nval, 40) query vec per val image
    gt8 = val_gt.astype(np.int8)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * len(loader)
    warmup = min(500, total_steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / total_steps))))

    history = []
    best_madm = -1.0
    step = 0
    for epoch in range(args.epochs):
        student.train()
        t_ep = time.perf_counter()
        running = 0.0
        n_batches = 0
        for imgs, labels in loader:
            imgs = imgs.to(dev, non_blocking=True)
            labels = labels.to(dev, non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                t_logits = teacher(imgs)
            t_probs = torch.sigmoid(t_logits.float())

            logits = student(imgs)
            loss = F.binary_cross_entropy_with_logits(logits, t_probs) \
                + 0.5 * F.binary_cross_entropy_with_logits(logits, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            running += float(loss.detach())
            n_batches += 1
        mean_loss = running / max(n_batches, 1)
        print(f"epoch {epoch} train loss {mean_loss:.4f} ({time.perf_counter()-t_ep:.0f}s)", flush=True)

        # ---------------- full-val evaluation ----------------------------
        student.eval()
        probs = infer_probs(student, val_paths, dev)
        Dmat = M.l1_distances(Qf, probs)
        t2 = M.track2_metrics(Dmat, qids, gt8)
        t1 = M.track1_metrics(gt8, probs)
        row = {"epoch": epoch, "loss": mean_loss,
               "t2_madm": t2["mADM"], "t2_map": t2["mAP"], "t2_r1": t2["Rank-1"],
               "t1_ca": t1["Challenge_Avg"], "t1_mA": t1["mA"]}
        history.append(row)
        print(json.dumps(row))
        ck = {"state": student.state_dict(), "epoch": epoch,
              "arch": f"mobilenet_v3_{args.arch}"}
        torch.save(ck, args.out / f"ep{epoch}.pt")
        if t2["mADM"] > best_madm:
            best_madm = t2["mADM"]
            torch.save(ck, args.out / "best.pt")
        (args.out / "history.json").write_text(json.dumps(history, indent=2))

    print("done; best mADM", best_madm)


def infer_probs(model: nn.Module, paths: list[str], dev: torch.device,
                batch: int = 256) -> np.ndarray:
    """Deterministic letterbox-224 inference, teacher-equivalent preprocess."""
    model.eval()
    out = np.empty((len(paths), 40), dtype=np.float32)
    from PIL import Image
    from pathlib import Path as _P
    DATA = _P(__file__).resolve().parent.parent / "data"
    with torch.no_grad():
        for i0 in range(0, len(paths), batch):
            imgs = []
            for p in paths[i0:i0 + batch]:
                with Image.open(DATA / p) as fh:
                    imgs.append(D.preprocess_tensor(D.letterbox(fh.convert("RGB"), 224)))
            x = torch.stack(imgs).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            out[i0:i0 + batch] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


if __name__ == "__main__":
    main()
