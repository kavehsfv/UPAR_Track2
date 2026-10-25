"""Validate a built submission zip the way the ingestion program does.

Usage (pt env):
  python validate_zip.py <track1|track2> <zip> [--limit N] [--skip-t2-full]

- extracts zip to a fresh temp dir, imports run.py from there
- T1: predict_batch over the val split (or --limit N images), scores with
  official track1 metrics vs gt
- T2: rank_gallery on the val split (or --limit N gallery images / all
  3462 queries with --skip-t2-full subsampling like the official metric)
Exits non-zero on any contract violation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent          # release root (data/ lives next to it)
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "training"))
from data import (  # noqa: E402
    _open_csv,
    load_gt,
    load_ids,
    load_queries,
)
from metrics import track1_metrics, track2_metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("track", choices=["task1", "task2"])
    ap.add_argument("zip", type=str)
    ap.add_argument("--limit", type=int, default=0, help="limit gallery/val images")
    args = ap.parse_args()

    zpath = Path(args.zip)
    assert zpath.exists(), f"missing {zpath}"
    tmp = tempfile.mkdtemp(prefix="upar_val_")

    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        assert "run.py" in names, "run.py must be at zip ROOT"
        assert "metadata.yaml" in names, "metadata.yaml must be at zip ROOT"
        zf.extractall(tmp)

    spec = importlib.util.spec_from_file_location("run", str(Path(tmp) / "run.py"))
    assert spec is not None and spec.loader is not None, "could not load run.py spec"
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)

    val_paths, val_gt = load_gt(DATA / "annotations/task1/val/gt.csv")
    limit = args.limit or len(val_paths)
    paths, gt = val_paths[:limit], val_gt[:limit]

    # attribute order from gt header (canonical)
    with _open_csv(DATA / "annotations/task1/val/gt.csv") as fh:
        header = fh.readline().strip().lstrip("# ").split(",")
    attr_names = header[1:]

    if args.track == "task1":
        if hasattr(run, "load_model"):
            run.load_model()
        samples = [{"image_path": str(DATA / p), "attribute_names": attr_names}
                   for p in paths]
        if hasattr(run, "predict_batch"):
            P = np.asarray(run.predict_batch(samples), dtype=np.float64)
        else:
            P = np.asarray([run.predict_image(s) for s in samples], dtype=np.float64)
        assert P.shape == (len(paths), 40), f"bad probs shape {P.shape}"
        assert P.min() >= 0.0 and P.max() <= 1.0, "probs out of [0,1]"
        m = track1_metrics(gt, P)
        print(json.dumps({k: round(v, 4) for k, v in m.items()}, indent=2))
    else:
        if hasattr(run, "load_model"):
            run.load_model()
        ids = load_ids(DATA / "annotations/task2/val/ids.csv")
        Q = load_queries(DATA / "annotations/task2/val/queries.csv")
        gallery = [{"image_path": str(DATA / p)} for p in paths]
        sample = {"queries": Q, "gallery": gallery, "attribute_names": attr_names}
        out = run.rank_gallery(sample)
        assert isinstance(out, dict) and len(out) == 1, "must return single-key dict"
        key = next(iter(out))
        assert key in ("distances", "similarities", "ranking"), f"bad key {key}"
        D = np.asarray(out[key])
        assert D.shape == (len(Q), len(gallery)), f"bad matrix shape {D.shape}"
        if key in ("similarities", "ranking"):
            D = -D  # higher = better -> convert to distance
        qids = np.asarray([ids[p] for p in paths], dtype=np.int64)
        m = track2_metrics(D, qids, gt.astype(np.int8))
        print(json.dumps({k: round(v, 4) for k, v in m.items()
                          if k != "n_eval"}, indent=2))
    print(f"VALIDATION OK: {zpath.name} ({args.track})")


if __name__ == "__main__":
    main()
