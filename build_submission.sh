#!/usr/bin/env bash
# Build the task2 v6 (distilled student) submission zip.
# Layout mirrors the organizers' sample: run.py + metadata.yaml at root,
# assets/ with student_model.pt + train_prior.npy.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT"
OUT="$ROOT"

[ -f "$SRC/run.py" ] || { echo "missing $SRC/run.py"; exit 1; }
[ -f "$SRC/metadata.yaml" ] || { echo "missing metadata.yaml"; exit 1; }
[ -f "$SRC/weights/student_model.pt" ] || { echo "run training/export_student_v6.py first"; exit 1; }
[ -f "$SRC/weights/train_prior.npy" ] || { echo "missing train_prior.npy"; exit 1; }

STAGE="$(mktemp -d)"
mkdir -p "$STAGE/assets" "$OUT/submission_zip"
cp "$SRC/run.py" "$STAGE/run.py"
cp "$SRC/metadata.yaml" "$STAGE/"
cp "$SRC/weights/student_model.pt" "$SRC/weights/train_prior.npy" "$STAGE/assets/"

ZIP="$OUT/submission_zip/task2_v6_distill.zip"
rm -f "$ZIP"
(cd "$STAGE" && zip -q -r "$ZIP" run.py metadata.yaml assets -x "*__pycache__*")
rm -rf "$STAGE"
echo "built $ZIP ($(du -h "$ZIP" | cut -f1))"
