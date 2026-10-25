"""Download & convert CLIP ViT-B/16 (openai) visual tower into our plain-torch format.

Run once in an env with network + open_clip_torch. Output: weights/clip_vitb16_visual.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vit import VisionTransformer, load_open_clip_visual  # noqa: E402


def main() -> None:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai", cache_dir=str(Path(__file__).resolve().parent / "weights" / "hf_cache")
    )
    sd = model.state_dict()
    visual = load_open_clip_visual(sd)
    vit = VisionTransformer()
    missing, unexpected = vit.load_state_dict(visual, strict=False)
    # positional_embedding may already match; conv1/class_embedding/blocks should load.
    real_missing = [k for k in missing if not k.startswith("proj")]
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert not real_missing, f"missing keys: {real_missing}"
    # proj: open_clip visual.proj is [width, output_dim]; ours is Parameter(width, out)
    vit.proj.data = sd["visual.proj"].clone() if "visual.proj" in sd else vit.proj.data
    out = Path(__file__).resolve().parent / "weights"
    out.mkdir(exist_ok=True)
    torch.save(vit.state_dict(), out / "clip_vitb16_visual.pt")
    print(f"saved {out / 'clip_vitb16_visual.pt'}")


if __name__ == "__main__":
    main()
