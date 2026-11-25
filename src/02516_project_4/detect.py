"""Task 4.3: apply the trained CNN to all proposals of an image, NMS, and Average Precision.

    # GPU: score the top-2000 Selective Search proposals of the val and test images
    uv run python src/02516_project_4/detect.py infer --split val test
    # CPU: NMS + AP for several variants (NMS threshold chosen on val), figures
    uv run python src/02516_project_4/detect.py evaluate
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from dataset import IMAGE_DIR, get_train_val_test, load_dataframe  # noqa: E402
from proposals import FIG_DIR, OUT_DIR, box_iou, load_proposals  # noqa: E402
from rcnn import MIN_PROPOSAL_SIDE, ProposalClassifier, crop_box, decode, load_images, predict_crops  # noqa: E402

K = 2000
MODEL_PATH = os.path.join(OUT_DIR, "rcnn_resnet50.pt")


def raw_path(split):
    return os.path.join(OUT_DIR, f"raw_detections_{split}.npz")


# --------------------------------------------------------------------------- inference
class ImageCrops(torch.utils.data.Dataset):
    def __init__(self, images, items):
        self.images, self.items = images, items  # items: list of (filename, box)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, box = self.items[i]
        return torch.from_numpy(crop_box(self.images[f], box))


def select_proposals(props):
    p = props[:K]
    keep = ((p[:, 2] - p[:, 0]) >= MIN_PROPOSAL_SIDE) & ((p[:, 3] - p[:, 1]) >= MIN_PROPOSAL_SIDE)
    return np.unique(p[keep], axis=0)


def infer(splits, workers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProposalClassifier(pretrained=False).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    props = load_proposals("quality")
    _, val_df, test_df = get_train_val_test()
    for split in splits:
        df = {"val": val_df, "test": test_df}[split]
        images = load_images(df)
        per_image = {f: select_proposals(props[f]) for f in df.filename}
        items = [(f, b) for f in df.filename for b in per_image[f]]
        loader = torch.utils.data.DataLoader(ImageCrops(images, items), batch_size=256, num_workers=workers)
        scores, deltas = predict_crops(model, loader, device)
        out, start = {}, 0
        for f in df.filename:
            n = len(per_image[f])
            out[f"{f}/boxes"] = per_image[f]
            out[f"{f}/scores"] = scores[start:start + n]
            out[f"{f}/deltas"] = deltas[start:start + n]
            start += n
        np.savez_compressed(raw_path(split), **out)
        print(f"[{split}] scored {len(items)} proposals from {len(df)} images -> {raw_path(split)}", flush=True)


def load_raw(split):
    data = np.load(raw_path(split))
    files = sorted({k.rsplit("/", 1)[0] for k in data.files})
    return {f: {k: data[f"{f}/{k}"] for k in ("boxes", "scores", "deltas")} for f in files}


# --------------------------------------------------------------------------- NMS
def nms(boxes, scores, iou_threshold):
    """Greedy non-maximum suppression. Returns indices of kept boxes, highest score first."""
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_threshold]  # drop everything that overlaps the kept box too much
    return np.array(keep, dtype=np.int64)


# --------------------------------------------------------------------------- AP
def average_precision(detections, gt_boxes, iou_threshold=0.5):
    """Average Precision for a single class, as defined in the lecture (L4.2).

    detections: list of (filename, score, box); gt_boxes: {filename: (G, 4)}.
    Detections are sorted by decreasing score. A detection is a true positive if
    it matches a not yet used GT box with IoU >= iou_threshold (the GT box is then
    removed), otherwise a false positive. AP is the area under the resulting
    precision-recall curve: the precision at every true positive, summed and
    divided by the number of GT boxes (lecture example: (1 + 1 + 0.6) / 3).
    """
    n_gt = sum(len(b) for b in gt_boxes.values())
    detections = sorted(detections, key=lambda d: -d[1])
    matched = {f: np.zeros(len(b), bool) for f, b in gt_boxes.items()}
    tp = np.zeros(len(detections))
    for i, (f, _, box) in enumerate(detections):
        gt = gt_boxes[f]
        if len(gt) == 0:
            continue
        ious = box_iou(box[None], gt)[0]
        ious[matched[f]] = -1  # GT boxes already matched are removed
        j = ious.argmax()
        if ious[j] >= iou_threshold:
            matched[f][j] = True
            tp[i] = 1
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    ap = float(precision[tp == 1].sum() / n_gt)
    return ap, precision, recall


def make_detections(raw, use_regression, nms_threshold, score_threshold=0.0, max_per_image=100):
    dets = {}
    for f, r in raw.items():
        boxes = decode(r["boxes"].astype(np.float32), r["deltas"].copy()) if use_regression else r["boxes"].astype(np.float32)
        scores = r["scores"]
        keep = scores >= score_threshold
        boxes, scores = boxes[keep], scores[keep]
        if nms_threshold is not None:
            idx = nms(boxes, scores, nms_threshold)
        else:
            idx = np.argsort(-scores)
        idx = idx[:max_per_image] if max_per_image else idx
        dets[f] = (boxes[idx], scores[idx])
    return dets


def flat(dets):
    return [(f, s, b) for f, (bs, ss) in dets.items() for b, s in zip(bs, ss)]


def coco_style_ap(dets, gt):
    return float(np.mean([average_precision(flat(dets), gt, t)[0] for t in np.arange(0.5, 0.96, 0.05)]))


def evaluate():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torchvision

    df = load_dataframe()
    gt_all = dict(zip(df.filename, df.boxes))
    results = {}

    # 0) sanity check: our NMS == torchvision's NMS
    raw_test = load_raw("test")
    for f in list(raw_test)[:10]:
        b, s = raw_test[f]["boxes"].astype(np.float32), raw_test[f]["scores"]
        ours = nms(b, s, 0.5)
        ref = torchvision.ops.nms(torch.from_numpy(b), torch.from_numpy(s.astype(np.float32)), 0.5).numpy()
        assert np.array_equal(np.sort(ours), np.sort(ref)), f"NMS mismatch on {f}"
    print("NMS check: identical to torchvision.ops.nms on 10 test images")

    # 1) choose the NMS IoU threshold on the validation set
    raw_val = load_raw("val")
    gt_val = {f: gt_all[f] for f in raw_val}
    val_rows = []
    for reg in (False, True):
        for thr in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            ap = average_precision(flat(make_detections(raw_val, reg, thr)), gt_val, 0.5)[0]
            val_rows.append({"regression": reg, "nms": thr, "val_ap50": ap})
            print(f"[val] regression={reg!s:5} NMS {thr:.1f}: AP@0.5 = {ap:.4f}")
    results["val_nms_sweep"] = val_rows
    best = {reg: max((r for r in val_rows if r["regression"] == reg), key=lambda r: r["val_ap50"])["nms"]
            for reg in (False, True)}
    print(f"best NMS threshold on val: {best}")

    # 2) test set: variants
    gt_test = {f: gt_all[f] for f in raw_test}
    # upper bound: fraction of test GT boxes reachable by any of the K proposals
    reach = np.concatenate([box_iou(raw_test[f]["boxes"], gt_test[f]).max(0) for f in raw_test])
    results["test_proposal_recall@0.5"] = float((reach >= 0.5).mean())
    variants = {
        "no NMS, no box regression": (False, None, None),
        f"NMS {best[False]}, no box regression": (False, best[False], 100),
        f"NMS {best[True]}, with box regression": (True, best[True], 100),
    }
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    results["test"] = {}
    for name, (reg, thr, max_n) in variants.items():
        dets = make_detections(raw_test, reg, thr, max_per_image=max_n)
        ap50, prec, rec = average_precision(flat(dets), gt_test, 0.5)
        ap75 = average_precision(flat(dets), gt_test, 0.75)[0]
        apcoco = coco_style_ap(dets, gt_test)
        n_det = sum(len(v[1]) for v in dets.values())
        results["test"][name] = {"AP50": ap50, "AP75": ap75, "AP50:95": apcoco, "max_recall": float(rec[-1]),
                                 "detections_per_image": n_det / len(dets)}
        print(f"[test] {name:40s} AP@0.5 {ap50:.4f}  AP@0.75 {ap75:.4f}  AP@[.5:.95] {apcoco:.4f}  "
              f"max recall {rec[-1]:.3f}  dets/img {n_det / len(dets):.0f}")
        ax.plot(rec, prec, label=f"{name} (AP50={ap50:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("Test set precision-recall (IoU 0.5)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "test_pr_curve.png"), dpi=150)

    with open(os.path.join(OUT_DIR, "detection_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved {OUT_DIR}/detection_results.json and {FIG_DIR}/test_pr_curve.png")

    # 3) qualitative examples (final model: regression + val-chosen NMS), score >= 0.5
    show_detections(make_detections(raw_test, True, best[True]), gt_test, os.path.join(FIG_DIR, "test_detections.png"))


def show_detections(dets, gt, out_path, files=None, score_threshold=0.5):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    files = files or sorted(dets)[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, f in zip(axes.ravel(), files):
        ax.imshow(Image.open(os.path.join(IMAGE_DIR, f)).convert("RGB"))
        for x1, y1, x2, y2 in gt[f]:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2))
        boxes, scores = dets[f]
        for (x1, y1, x2, y2), s in zip(boxes, scores):
            if s < score_threshold:
                continue
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=1.5, linestyle="--"))
            ax.text(x1, y1 - 2, f"{s:.2f}", color="white", fontsize=7, bbox=dict(facecolor="red", pad=0.5, lw=0))
        ax.set_title(f, fontsize=9)
        ax.axis("off")
    fig.suptitle("Test detections after NMS (red, score >= 0.5) and ground truth (green)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("infer")
    i.add_argument("--split", nargs="+", default=["val", "test"])
    i.add_argument("--workers", type=int, default=int(os.environ.get("LSB_DJOB_NUMPROC", 4)))
    sub.add_parser("evaluate")
    args = p.parse_args()
    if args.cmd == "infer":
        infer(args.split, args.workers)
    else:
        evaluate()


if __name__ == "__main__":
    main()
