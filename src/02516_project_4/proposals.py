"""Task 4.1 (2)+(3): extract Selective Search proposals and evaluate them.

Extract (run on a compute node, see jobs/proposals.sh):
    uv run python src/02516_project_4/proposals.py extract --mode fast --workers 8
Evaluate on the training split:
    uv run python src/02516_project_4/proposals.py evaluate

Proposals are stored in outputs/proposals_ss_<mode>.npz as one int16 array of
shape (K, 4) per image, keyed by filename, in [x1, y1, x2, y2] coordinates of
the ORIGINAL image, in Selective Search's own ranking order.
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dataset import get_splits, load_dataframe  # noqa: E402

OUT_DIR = "outputs"
FIG_DIR = "figures"
MAX_SIDE = 500  # images are at most ~540 px wide, so this rarely resizes anything


def proposals_path(mode):
    return os.path.join(OUT_DIR, f"proposals_ss_{mode}.npz")


def selective_search(image_path, mode="fast", max_side=MAX_SIDE):
    cv2.setNumThreads(1)
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    if mode == "fast":
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()
    rects = ss.process().astype(np.float32)  # x, y, w, h
    boxes = np.stack([rects[:, 0], rects[:, 1], rects[:, 0] + rects[:, 2], rects[:, 1] + rects[:, 3]], 1)
    boxes = np.round(boxes / scale)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes.astype(np.int16)


def _worker(args):
    filename, path, mode = args
    t = time.time()
    return filename, selective_search(path, mode), time.time() - t


def extract(mode, workers):
    df = load_dataframe()
    jobs = [(r.filename, r.image_path, mode) for r in df.itertuples()]
    results, times = {}, []
    with Pool(workers) as pool:
        for i, (fname, boxes, dt) in enumerate(pool.imap_unordered(_worker, jobs, chunksize=4), 1):
            results[fname] = boxes
            times.append(dt)
            if i % 50 == 0 or i == len(jobs):
                print(f"[{mode}] {i}/{len(jobs)}  mean {np.mean(times):.2f}s/img", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(proposals_path(mode), **results)
    n = np.array([len(b) for b in results.values()])
    print(f"[{mode}] saved {proposals_path(mode)}: proposals/img mean {n.mean():.0f}, "
          f"min {n.min()}, max {n.max()}; time/img {np.mean(times):.2f}s")


def load_proposals(mode):
    data = np.load(proposals_path(mode))
    return {k: data[k].astype(np.float32) for k in data.files}


def box_iou(a, b):
    """IoU matrix between boxes a (N, 4) and b (M, 4) in x1y1x2y2."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod((rb - lt).clip(0), axis=2)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def best_iou_per_gt(proposals, gt, ks):
    """For each k in ks: best IoU of every GT box among the first k proposals. Shape (len(ks), n_gt)."""
    if len(proposals) == 0:
        return np.zeros((len(ks), len(gt)))
    iou = box_iou(proposals, gt)              # (P, G)
    running_max = np.maximum.accumulate(iou, axis=0)  # best IoU using the first i+1 proposals
    idx = np.minimum(np.array(ks), len(proposals)) - 1
    return running_max[idx]


KS = [1, 5, 10, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000]


def evaluate(modes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_df, _ = get_splits()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for mode in modes:
        if not os.path.exists(proposals_path(mode)):
            print(f"skipping {mode}: {proposals_path(mode)} not found")
            continue
        props = load_proposals(mode)
        best = np.concatenate([best_iou_per_gt(props[r.filename], r.boxes, KS) for r in train_df.itertuples()], 1)
        n_props = np.array([len(props[f]) for f in train_df.filename])
        rec50 = (best >= 0.5).mean(1)
        rec70 = (best >= 0.7).mean(1)
        mabo = best.mean(1)
        print(f"\n=== Selective Search {mode} (train split: {len(train_df)} images, {best.shape[1]} GT boxes) ===")
        print(f"proposals/img: mean {n_props.mean():.0f}, median {np.median(n_props):.0f}, max {n_props.max()}")
        print(f"{'k':>6} {'recall@0.5':>11} {'recall@0.7':>11} {'MABO':>6}")
        for k, r5, r7, m in zip(KS, rec50, rec70, mabo):
            print(f"{k:>6} {r5:>11.3f} {r7:>11.3f} {m:>6.3f}")
        axes[0].plot(KS, rec50, marker=".", label=f"SS {mode}")
        axes[1].plot(KS, rec70, marker=".", label=f"SS {mode}")
        axes[2].plot(KS, mabo, marker=".", label=f"SS {mode}")
    for ax, title in zip(axes, ["Recall @ IoU 0.5", "Recall @ IoU 0.7", "MABO"]):
        ax.set_xscale("log")
        ax.set_xlabel("number of proposals per image (k)")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "proposal_recall.png"), dpi=130)
    print(f"\nSaved {FIG_DIR}/proposal_recall.png")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--mode", choices=["fast", "quality"], default="fast")
    e.add_argument("--workers", type=int, default=int(os.environ.get("LSB_DJOB_NUMPROC", 1)))
    v = sub.add_parser("evaluate")
    v.add_argument("--modes", nargs="+", default=["fast", "quality"])
    args = p.parse_args()
    if args.cmd == "extract":
        extract(args.mode, args.workers)
    else:
        evaluate(args.modes)


if __name__ == "__main__":
    main()
