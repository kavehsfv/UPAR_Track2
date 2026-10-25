"""Fine-tune CLIP ViT-B/16 visual tower on UPAR 40-attribute multi-label PAR.

Usage (pt env):
  python train.py --epochs 6 --batch-size 256 --lr-backbone 2e-5 --lr-head 1e-3

Design notes:
  - Weighted BCE: per-attribute pos-weight = sqrt(neg/pos), tamed by exponent to
    avoid over-committing on 0.08%-positive attributes.
  - Differential LR: backbone 2e-5, head 1e-3 (new head needs to move far).
  - Cosine schedule + warmup, AdamW, AMP bf16.
  - Validates every epoch on the val split with BOTH tracks' official metrics;
    checkpoints best-by Challenge_Avg (T1) and keeps probs .npz for T2 eval.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (  # noqa: E402
    UPARTrainDataset,
    load_gt,
    load_ids,
    load_queries,
    preprocess_tensor,
)
from metrics import l1_distances, track1_metrics, track2_metrics  # noqa: E402
from vit import PARModel, VisionTransformer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent          # repo root
DATA = ROOT / "data"


def build_model(device: torch.device, init_from: str | None = None) -> PARModel:
    """CLIP-init by default; --init-from resumes visual+head from a best.pt."""
    vit = VisionTransformer()
    model = PARModel(vit).to(device)
    if init_from:
        ck = torch.load(init_from, map_location="cpu", weights_only=True)
        model.visual.load_state_dict(ck["visual"], strict=True)
        model.head.load_state_dict(ck["head"], strict=True)
        print(f"resumed weights from {init_from} (epoch {ck.get('epoch', '?')})", flush=True)
        return model
    init_path = Path(__file__).resolve().parent / "weights" / "clip_vitb16_visual.pt"
    sd = torch.load(init_path, map_location="cpu", weights_only=True)
    model.visual.load_state_dict(sd, strict=True)
    return model


@torch.no_grad()
def infer_probs(model: PARModel, paths, device, batch_size: int = 256) -> np.ndarray:
    """Deterministic val-time inference (no transform)."""
    from PIL import Image
    model.eval()
    out = np.empty((len(paths), 40), dtype=np.float32)
    for i0 in range(0, len(paths), batch_size):
        imgs = [preprocess_tensor(Image.open(DATA / p)) for p in paths[i0:i0 + batch_size]]
        x = torch.stack(imgs).to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        out[i0:i0 + len(imgs)] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--pos-weight-exp", type=float, default=0.5)
    ap.add_argument("--out", type=str, default="runs/clip_vitb16")
    ap.add_argument("--init-from", type=str, default=None,
                    help="resume visual+head from a prior best.pt")
    ap.add_argument("--gpu-throttle", type=float, default=0.0,
                    help="sleep this fraction of iteration time to reduce GPU load (0..1)")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = UPARTrainDataset(DATA, DATA / "annotations/task1/train/gt.csv")
    val_paths, val_gt = load_gt(DATA / "annotations/task1/val/gt.csv")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0,
    )
    print(f"train images: {len(train_ds)}  val images: {len(val_paths)}", flush=True)

    # pos-weights from train labels
    y_all = np.concatenate([train_ds.labels[i:i + 8192] for i in range(0, len(train_ds), 8192)])
    pos = y_all.mean(0)
    pw = ((1.0 - pos) / np.maximum(pos, 1e-6)) ** args.pos_weight_exp
    pw = torch.tensor(pw, dtype=torch.float32, device=device)
    print("pos-weight range:", float(pw.min()), float(pw.max()), flush=True)

    model = build_model(device, args.init_from)
    param_groups = [
        {"params": model.visual.parameters(), "lr": args.lr_backbone},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ]
    opt = torch.optim.AdamW(param_groups, weight_decay=args.wd)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs

    def lr_lambda(step: int) -> float:
        if step < args.warmup:
            return step / max(args.warmup, 1)
        p = (step - args.warmup) / max(total_steps - args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best = {"Challenge_Avg": -1.0}
    history = []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        it = -1
        for it, (x, y) in enumerate(train_loader):
            t_it = time.time()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if args.gpu_throttle > 0:
                time.sleep(args.gpu_throttle * (time.time() - t_it))
            running += float(loss.detach())
            if it % 100 == 0:
                print(f"ep {epoch} it {it}/{steps_per_epoch} loss {running/(it+1):.4f} "
                      f"({(time.time()-t0):.0f}s)", flush=True)
        print(f"epoch {epoch} done in {(time.time()-t0)/60:.1f} min, mean loss {running/max(it+1,1):.4f}", flush=True)
        probs = infer_probs(model, val_paths, device, batch_size=args.batch_size)
        t1 = track1_metrics(val_gt, probs)
        print(f"epoch {epoch} T1: " + "  ".join(f"{k}={v:.4f}" for k, v in t1.items()), flush=True)

        # Track 2 quick view (L1 on val gallery with val queries)
        ids = load_ids(DATA / "annotations/task2/val/ids.csv")
        Q = load_queries(DATA / "annotations/task2/val/queries.csv")
        qids = np.asarray([ids[p] for p in val_paths], dtype=np.int64)
        D = l1_distances(Q.astype(np.float32), probs)
        t2 = track2_metrics(D, qids, val_gt.astype(np.int8))
        print(f"epoch {epoch} T2: " + "  ".join(f"{k}={v:.4f}" for k, v in t2.items()), flush=True)

        history.append({"epoch": epoch, "t1": t1, "t2": {k: v for k, v in t2.items() if k != "n_eval"}})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        np.savez_compressed(out_dir / f"val_probs_ep{epoch}.npz", probs=probs)
        torch.save({"visual": model.visual.state_dict(),
                    "head": model.head.state_dict(),
                    "epoch": epoch}, out_dir / f"ep{epoch}.pt")

        if t1["Challenge_Avg"] > best["Challenge_Avg"]:
            best = t1
            torch.save({"visual": model.visual.state_dict(),
                        "head": model.head.state_dict(),
                        "epoch": epoch}, out_dir / "best.pt")
            print(f"epoch {epoch}: new best Challenge_Avg={t1['Challenge_Avg']:.4f}", flush=True)
    print("BEST:", json.dumps(best, indent=2), flush=True)


if __name__ == "__main__":
    main()
