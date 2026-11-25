"""Task 4.2: fine-tune a CNN to classify proposals (pothole vs background) + box regression.

    uv run python src/02516_project_4/train.py            (on a GPU node, see jobs/train.sh)

Class imbalance (~25 background : 1 pothole) is handled by the sampler: every
mini-batch is drawn with a fixed fraction of positives (POS_FRACTION, R-CNN uses
25%), with replacement, so the network sees all positives many times while the
huge background pool is subsampled differently in every batch.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from dataset import get_train_val_test, load_dataframe  # noqa: E402
from rcnn import CROP_SIZE, ProposalClassifier, crop_box, encode, load_images, predict_crops, to_input  # noqa: E402

OUT_DIR = "outputs"
LABELS_PATH = os.path.join(OUT_DIR, "labeled_proposals_ss_quality_k2000.pkl")


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, table, images, gt_boxes, train=False):
        self.boxes = table[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        self.files = table["filename"].astype(str).to_numpy()
        self.labels = table["label"].to_numpy(np.int64)
        self.images = images
        self.train = train
        # regression targets for positives (towards the GT box they overlap most)
        self.targets = np.zeros((len(table), 4), np.float32)
        pos = np.flatnonzero(self.labels == 1)
        gt = np.stack([gt_boxes[f][g] for f, g in zip(self.files[pos], table["gt_idx"].to_numpy()[pos])])
        self.targets[pos] = encode(self.boxes[pos], gt.astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        crop = crop_box(self.images[self.files[i]], self.boxes[i])
        target = self.targets[i].copy()
        if self.train and np.random.rand() < 0.5:  # horizontal flip; mirror the x-offset target too
            crop = crop[:, ::-1]
            target[0] = -target[0]
        return torch.from_numpy(np.ascontiguousarray(crop)), self.labels[i], torch.from_numpy(target)


class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, labels, batch_size, pos_fraction, n_iter, seed=0):
        self.pos = np.flatnonzero(labels == 1)
        self.neg = np.flatnonzero(labels == 0)
        self.n_pos = int(round(batch_size * pos_fraction))
        self.n_neg = batch_size - self.n_pos
        self.n_iter = n_iter
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        for _ in range(self.n_iter):
            yield np.concatenate([self.rng.choice(self.pos, self.n_pos), self.rng.choice(self.neg, self.n_neg)]).tolist()


def classification_metrics(probs, labels, threshold=0.5):
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp)),
        "pothole_recall": tp / (tp + fn),
        "background_recall": tn / (tn + fp),
        "pothole_precision": tp / max(tp + fp, 1),
        "proposal_ap": average_precision_score(labels, probs),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n": int(len(labels)), "n_pos": int(labels.sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--iters", type=int, default=300, help="iterations per epoch")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--pos-fraction", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--box-weight", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=int(os.environ.get("LSB_DJOB_NUMPROC", 8)))
    p.add_argument("--out", default=os.path.join(OUT_DIR, "rcnn_resnet50.pt"))
    args = p.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} {torch.cuda.get_device_name() if device.type == 'cuda' else ''}", flush=True)

    df = load_dataframe()
    train_imgs, val_imgs, _ = get_train_val_test(df)
    gt_boxes = dict(zip(df.filename, df.boxes))
    table = pd.read_pickle(LABELS_PATH)
    table = table[table.split == "train"]
    train_tab = table[table.filename.isin(set(train_imgs.filename))].reset_index(drop=True)
    val_tab = table[table.filename.isin(set(val_imgs.filename))].reset_index(drop=True)
    for name, t in (("train", train_tab), ("val", val_tab)):
        print(f"{name}: {t.filename.nunique()} images, {len(t)} crops, {(t.label == 1).sum()} pothole, "
              f"{(t.label == 0).sum()} background", flush=True)

    images = load_images(pd.concat([train_imgs, val_imgs]))
    train_ds = CropDataset(train_tab, images, gt_boxes, train=True)
    val_ds = CropDataset(val_tab, images, gt_boxes)

    # validation during training: all positives + a fixed random 15k background subset (full set at the end)
    rng = np.random.default_rng(0)
    val_sub_idx = np.concatenate([np.flatnonzero(val_ds.labels == 1),
                                  rng.choice(np.flatnonzero(val_ds.labels == 0), 15000, replace=False)])
    loader_kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_sampler=BalancedBatchSampler(train_ds.labels, args.batch, args.pos_fraction, args.iters), **loader_kw)
    val_sub_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(val_ds, val_sub_idx), batch_size=256, **loader_kw)

    model = ProposalClassifier().to(device)
    head_params = list(model.cls_head.parameters()) + list(model.box_head.parameters())
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": args.lr},
                             {"params": head_params, "lr": args.lr * 10}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * 10],
                                                total_steps=args.epochs * args.iters, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history, best_ap = [], -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, sums, n = time.time(), np.zeros(3), 0
        for x, y, target in train_loader:
            x, y, target = to_input(x, device), y.to(device), target.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits, deltas = model(x)
            logits, deltas = logits.float(), deltas.float()
            cls_loss = F.cross_entropy(logits, y)
            pos = y == 1
            box_loss = F.smooth_l1_loss(deltas[pos], target[pos], beta=1 / 9) if pos.any() else deltas.sum() * 0
            loss = cls_loss + args.box_weight * box_loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            acc = (logits.argmax(1) == y).float().mean().item()
            sums += [cls_loss.item(), box_loss.item(), acc]
            n += 1
        probs, _ = predict_crops(model, val_sub_loader, device)
        m = classification_metrics(probs, val_ds.labels[val_sub_idx])
        rec = {"epoch": epoch, "train_cls_loss": sums[0] / n, "train_box_loss": sums[1] / n,
               "train_acc_balanced_batches": sums[2] / n, **{f"val_{k}": v for k, v in m.items() if k != "confusion"},
               "time_s": time.time() - t0}
        history.append(rec)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
        if m["proposal_ap"] > best_ap:
            best_ap = m["proposal_ap"]
            torch.save(model.state_dict(), args.out)

    # final evaluation of the best checkpoint on the FULL validation set (natural class balance)
    model.load_state_dict(torch.load(args.out, map_location=device))
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, **loader_kw)
    probs, deltas = predict_crops(model, val_loader, device)
    final = {}
    for thr in (0.5, 0.7, 0.9):
        final[str(thr)] = classification_metrics(probs, val_ds.labels, thr)
    # does box regression improve localisation of the validation positives?
    from rcnn import decode, paired_iou
    pos = np.flatnonzero((val_ds.labels == 1) & ~val_tab.is_gt.to_numpy())
    gt = np.stack([gt_boxes[f][g] for f, g in zip(val_ds.files[pos], val_tab.gt_idx.to_numpy()[pos])]).astype(np.float32)
    before = paired_iou(val_ds.boxes[pos], gt)
    after = paired_iou(decode(val_ds.boxes[pos], deltas[pos]), gt)
    final["box_regression_val_positives"] = {"mean_iou_before": float(before.mean()), "mean_iou_after": float(after.mean()),
                                             "frac_iou>=0.7_before": float((before >= 0.7).mean()),
                                             "frac_iou>=0.7_after": float((after >= 0.7).mean())}
    print(json.dumps(final, indent=1, default=float), flush=True)
    with open(os.path.join(OUT_DIR, "train_results.json"), "w") as f:
        json.dump({"args": vars(args), "history": history, "final_val": final}, f, indent=1, default=float)
    print(f"Saved {args.out} and {OUT_DIR}/train_results.json")


if __name__ == "__main__":
    main()
