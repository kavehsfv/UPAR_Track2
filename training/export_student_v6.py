"""Export distilled student -> run_v6 submission assets.

Two things happen here:
  1. KEY RENAME: the student was trained as torchvision mobilenet_v3_small
     (nested Sequential names like features.0.0.weight). run_v6.py's
     hand-rolled StudentNet uses flat names (features.0.weight). The rename
     map is derived by ORDERED-SHAPE MATCHING over the conv/bn/linear params,
     which is exact for two topologically identical networks, then verified
     by a strict load + numeric forward comparison against the original.
  2. Weight saving: run_v6 expects {"state": <flat state dict>, "epoch": N}.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent          # release root
sys.path.insert(0, str(ROOT / "training"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "training" / "runs" / "distill_mbv3l" / "ep4.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "weights" / "student_model.pt")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    tv_sd = ckpt["state"]                                # torchvision naming
    epoch = ckpt.get("epoch", -1)
    arch = ckpt.get("arch", "mobilenet_v3_small").replace("mobilenet_v3_", "")

    # ---- build run_v6 StudentNet --------------------------------------
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_v6", str(ROOT / "run.py"))
    assert spec is not None and spec.loader is not None
    v6 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v6)
    mine = v6.StudentNet(arch)
    my_sd = mine.state_dict()

    # ---- ordered-shape matching rename --------------------------------
    # Both state dicts enumerate parameters in the same topological order
    # (features -> head); same shapes must pair 1:1.
    my_items = [(k, v) for k, v in my_sd.items() if not k.endswith("num_batches_tracked")]
    tv_items = [(k, v) for k, v in tv_sd.items() if not k.endswith("num_batches_tracked")]
    assert len(my_items) == len(tv_items), (len(my_items), len(tv_items))
    new_sd = {}
    for (mk, mv), (tk, tv_) in zip(my_items, tv_items):
        assert tuple(mv.shape) == tuple(tv_.shape), (mk, tk, mv.shape, tv_.shape)
        new_sd[mk] = tv_
    # drop stale num_batches_tracked (defaults are fine for eval)

    # ---- verify: strict load + numeric forward parity ------------------
    missing, unexpected = mine.load_state_dict(new_sd, strict=True)[:2]
    assert not missing and not unexpected, (missing, unexpected)

    x = torch.randn(4, 3, 224, 224)
    mine.eval()
    with torch.no_grad():
        out_mine = mine(x)

    # reference: torchvision student with original weights
    sys.path.insert(0, str(ROOT / "training"))
    import importlib.util as _iu
    spec2 = _iu.spec_from_file_location("distill_t2", str(ROOT / "training" / "distill_t2.py"))
    assert spec2 is not None and spec2.loader is not None
    d2 = _iu.module_from_spec(spec2)
    spec2.loader.exec_module(d2)
    ref = d2.build_student(arch)
    ref.load_state_dict(tv_sd)
    ref.eval()
    with torch.no_grad():
        out_ref = ref(x)
    max_diff = (out_mine - out_ref).abs().max().item()
    print(f"parity max abs diff: {max_diff:.3e}")
    assert max_diff < 1e-5, "numeric parity failed"

    # ---- save -----------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": new_sd, "epoch": epoch, "arch": arch}, args.out)
    print(f"saved {args.out} ({args.out.stat().st_size/1e6:.1f} MB), epoch {epoch}")


if __name__ == "__main__":
    main()
