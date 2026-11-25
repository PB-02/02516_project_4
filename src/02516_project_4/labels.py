"""Task 4.1 (4): assign a class (pothole) or background label to each proposal.

    uv run python src/02516_project_4/labels.py --mode fast --k 2000

For every image, the first k Selective Search proposals are matched to the
ground truth by IoU (R-CNN style):
    IoU >= POS_IOU            -> label 1 (pothole)
    IoU <  NEG_IOU            -> label 0 (background)
    NEG_IOU <= IoU < POS_IOU  -> ignored (ambiguous, not used for training)
The ground-truth boxes themselves are added as extra positives (is_gt=True).

Output: outputs/labeled_proposals_ss_<mode>_k<k>.pkl, a pandas DataFrame with
one row per kept box: filename, split, x1, y1, x2, y2, iou, gt_idx, label, is_gt.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from dataset import get_splits  # noqa: E402
from proposals import FIG_DIR, OUT_DIR, box_iou, load_proposals  # noqa: E402

POS_IOU = 0.5
NEG_IOU = 0.3
MIN_PROPOSAL_SIDE = 4  # px; drop slivers that cannot be classified meaningfully


def label_image(proposals, gt, k):
    props = proposals[:k]
    w = props[:, 2] - props[:, 0]
    h = props[:, 3] - props[:, 1]
    props = props[(w >= MIN_PROPOSAL_SIDE) & (h >= MIN_PROPOSAL_SIDE)]
    props = np.unique(props, axis=0)  # SS returns some duplicates after rescaling

    iou = box_iou(props, gt)
    best = iou.max(1)
    gt_idx = iou.argmax(1)
    label = np.full(len(props), -1, np.int8)
    label[best >= POS_IOU] = 1
    label[best < NEG_IOU] = 0
    keep = label >= 0

    boxes = np.concatenate([props[keep], gt.astype(np.float32)])
    return pd.DataFrame({
        "x1": boxes[:, 0].astype(np.int16), "y1": boxes[:, 1].astype(np.int16),
        "x2": boxes[:, 2].astype(np.int16), "y2": boxes[:, 3].astype(np.int16),
        "iou": np.concatenate([best[keep], np.ones(len(gt))]).astype(np.float32),
        "gt_idx": np.concatenate([gt_idx[keep], np.arange(len(gt))]).astype(np.int16),
        "label": np.concatenate([label[keep], np.ones(len(gt), np.int8)]),
        "is_gt": np.concatenate([np.zeros(keep.sum(), bool), np.ones(len(gt), bool)]),
    }), int((~keep).sum())


def build(mode, k):
    props = load_proposals(mode)
    train_df, test_df = get_splits()
    frames, ignored = [], 0
    for split, df in (("train", train_df), ("test", test_df)):
        for r in df.itertuples():
            f, n_ign = label_image(props[r.filename], r.boxes, k)
            f.insert(0, "split", split)
            f.insert(0, "filename", r.filename)
            frames.append(f)
            ignored += n_ign
    out = pd.concat(frames, ignore_index=True)
    out["filename"] = out["filename"].astype("category")
    out["split"] = out["split"].astype("category")
    path = os.path.join(OUT_DIR, f"labeled_proposals_ss_{mode}_k{k}.pkl")
    out.to_pickle(path)

    print(f"Saved {path} ({len(out)} rows, {ignored} ambiguous proposals dropped)")
    for split in ("train", "test"):
        s = out[out.split == split]
        pos_ss = ((s.label == 1) & ~s.is_gt).sum()
        n_img = s.filename.nunique()
        print(f"[{split}] {n_img} images | background {(s.label == 0).sum()} | pothole {(s.label == 1).sum()} "
              f"({pos_ss} from SS + {s.is_gt.sum()} GT) | bg:pothole = {(s.label == 0).sum() / (s.label == 1).sum():.1f}:1 "
              f"| per image: {(s.label == 0).sum() / n_img:.0f} bg, {(s.label == 1).sum() / n_img:.1f} pos")
    return out


def show_example(labeled, filename, out_path, n_bg=40, seed=0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    from dataset import IMAGE_DIR

    s = labeled[labeled.filename == filename]
    bg = s[s.label == 0].sample(n=min(n_bg, (s.label == 0).sum()), random_state=seed)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(Image.open(os.path.join(IMAGE_DIR, filename)).convert("RGB"))
    for rows, color, lw in ((bg, "red", 0.8), (s[(s.label == 1) & ~s.is_gt], "yellow", 1.0), (s[s.is_gt], "lime", 2.5)):
        for r in rows.itertuples():
            ax.add_patch(Rectangle((r.x1, r.y1), r.x2 - r.x1, r.y2 - r.y1, fill=False, edgecolor=color, linewidth=lw))
    ax.set_title(f"{filename}\nGT (green), pothole (yellow), {len(bg)} sampled background (red)", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fast", "quality"], default="fast")
    p.add_argument("--k", type=int, required=True, help="number of proposals kept per image")
    args = p.parse_args()
    labeled = build(args.mode, args.k)
    os.makedirs(FIG_DIR, exist_ok=True)
    show_example(labeled, "potholes217.png", os.path.join(FIG_DIR, "labeled_proposals_example.png"))


if __name__ == "__main__":
    main()
