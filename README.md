# UPAR 2027 Track 2 — Attribute-based Person Retrieval: Code Release

This release reproduces our Codabench Track 2 submission
(username `kavehsfv-996082-996082_VID-Docker-GPU`, leaderboard test-set results below).

**Method:** a MobileNetV3-Large student (3.0M parameters) distilled from a fine-tuned
CLIP ViT-B/16 teacher predicts 40 pedestrian-attribute probabilities per gallery image;
queries are ranked by plain L1 distance in probability space.

## Test-set results (Codabench leaderboard, 36,385 gallery images)

| Metric | Score |
|---|---|
| mADM (ranking metric) | 0.3929 |
| mAP | 0.1328 |
| Rank-1 | 0.1385 |
| Rank-5 | 0.2755 |
| Rank-10 | 0.3338 |
| mINP | 0.0996 |

## Layout

```
run.py                      <- exact submission entry point (Codabench ingestion contract)
build_submission.sh         <- rebuilds submission_zip/task2_v6_distill.zip from weights/
training/                   <- everything needed to retrain from scratch
  download_datasets.py      <- downloads Market1501 / PA100k / PETA (Google Drive / Dropbox)
  prepare_clip_init.py      <- converts open_clip CLIP ViT-B/16 (openai) visual tower
  train.py                  <- fine-tunes the TEACHER (CLIP ViT-B/16) on the UPAR train split
  distill_t2.py             <- distills the teacher into the MobileNetV3 student
  export_student_v6.py      <- converts the distilled checkpoint to the run.py flat format
  data.py                   <- dataset + CSV loaders + augmentation
  vit.py                    <- plain-PyTorch CLIP-compatible ViT (weight-compatible w/ open_clip)
  metrics.py                <- official-style track metrics (mADM, mAP, mINP, Rank-*)
weights/
  student_model.pt          <- EXACT weights of the standing submission (12MB, epoch 4, arch=large)
  train_prior.npy           <- train-split attribute prior (fallback asset)
submission_zip/
  task2_v6_distill.zip      <- the standing Codabench submission zip (byte-identical)
validate_zip.py             <- validates a submission zip exactly like the ingestion program
Dockerfile                  <- reproducible inference environment (pytorch 2.4.1, as on Codabench)
```

## Reproducing inference

1. Download the datasets (needed for the val split images; see `training/download_datasets.py`)
   or place your own pedestrian crops + annotation CSVs under `data/` following the
   UPAR 2027 starter-kit layout.
2. Weights: `weights/student_model.pt` (also inside the submission zip at
   `assets/student_model.pt`).
3. Quick check (same code path as the Codabench ingestion):

```bash
python validate_zip.py task2 submission_zip/task2_v6_distill.zip --limit 64
```

4. Full val-split evaluation:

```bash
python validate_zip.py task2 submission_zip/task2_v6_distill.zip
```

Expected val results (33,407 gallery images): mADM 0.5413, mAP 0.0916, mINP 0.0301,
Rank-1 0.1306, Rank-5 0.2317, Rank-10 0.2796.

The student is CPU-friendly by design: ~17 ms/image fp32 on 4 cores, so the full
gallery finishes in ~10 minutes even without a GPU (the evaluation container may be
CPU-only). A train-prior fallback guarantees a valid ranking matrix in degraded
environments.

## Reproducing training

Environment: Python 3.11, PyTorch >= 2.0 + CUDA GPU (a single 32 GB GPU was used),
`open_clip_torch` + `torchvision` (one-time weight downloads), Pillow, NumPy, tqdm, gdown<6.

Two stages:

```bash
cd training
# Stage 1: fine-tune the CLIP ViT-B/16 teacher (identical to the Track 1 recipe)
python download_datasets.py                    # Market1501, PA100k, PETA -> ../data/
python prepare_clip_init.py                    # CLIP ViT-B/16 (openai) -> training/weights/
python train.py --epochs 6 --batch-size 256    # -> training/runs/clip_vitb16/ep{N}.pt

# Stage 2: distill into the MobileNetV3-Large student
python distill_t2.py --arch large --epochs 8 --batch-size 230 \
    --teacher-ckpt runs/clip_vitb16/ep4.pt --out runs/distill_mbv3l
python export_student_v6.py --ckpt runs/distill_mbv3l/ep4.pt --out ../weights/student_model.pt
```

`build_submission.sh` (from the release root) then packs `run.py` + `metadata.yaml` +
`weights/` into `submission_zip/task2_v6_distill.zip`.

Key hyper-parameters:
- Teacher (as in Track 1): weighted BCE, per-attribute pos-weight `(neg/pos)^0.5`;
  AdamW wd 0.05; backbone LR 2e-5, head LR 1e-3; warmup 500 + cosine; 6 epochs; batch 256;
  bf16 AMP; hflip p=0.5 + random erasing p=0.25; letterbox 224 + CLIP normalization
- Distillation: `BCE(z, teacher_probs) + 0.5 * BCE(z, labels)`; AdamW lr 3e-4 wd 0.05;
  warmup 500 + cosine; batch 230; 8 epochs; student init = torchvision MobileNetV3-Large
  ImageNet weights; checkpoint = epoch 4 by validation mADM
- Inference: plain L1 distance (no weighting, no calibration — all such variants
  reduced validation mADM: weighting −13%, bias calibration −16%)

## Docker (recommended by the organizers)

```bash
docker build -t upar2027-track2 .
# inside the container, the val split must be mounted at /data
docker run --gpus all -v /path/to/upar-data:/data upar2027-track2
```

The container builds on `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` — the same base
image the Codabench evaluation uses — and runs `validate_zip.py` on the submission zip
so the organizers can verify the Codabench result locally without adjustments.

## Notes

- The exact standing submission weights (`weights/student_model.pt`) are byte-identical
  to `assets/student_model.pt` inside `submission_zip/task2_v6_distill.zip`.
- `training/` is self-contained (no imports from the surrounding repo); paths resolve
  relative to the release layout (`data/` next to the release root, run outputs under
  `training/runs/`).
- The evaluation-time inference code (letterbox, chunked fp32 inference, L1 ranking,
  prior fallback) is the `run.py` at the release root — the same file shipped in the zip.
