"""Shared pieces for the R-CNN style detector: model, crops, box encoding."""

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, os.path.dirname(__file__))

CROP_SIZE = 224
CONTEXT = 0.1           # enlarge each proposal by 10% of its size on every side (R-CNN style context)
MIN_PROPOSAL_SIDE = 4
BOX_STD = np.array([0.1, 0.1, 0.2, 0.2], np.float32)  # normalisation of regression targets
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


def load_images(df):
    """Read all images of a split into memory as RGB uint8 arrays (dataset is ~300 MB in total)."""
    return {r.filename: cv2.cvtColor(cv2.imread(r.image_path), cv2.COLOR_BGR2RGB) for r in df.itertuples()}


def crop_box(image, box, size=CROP_SIZE, context=CONTEXT):
    """Warp a proposal (plus some context) to a size x size patch, as in R-CNN."""
    H, W = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    cw, ch = context * (x2 - x1), context * (y2 - y1)
    x1, x2 = int(max(0, np.floor(x1 - cw))), int(min(W, np.ceil(x2 + cw)))
    y1, y2 = int(max(0, np.floor(y1 - ch))), int(min(H, np.ceil(y2 + ch)))
    x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
    return cv2.resize(image[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_LINEAR)


def encode(proposals, gt):
    """R-CNN box-regression targets (tx, ty, tw, th), normalised by BOX_STD."""
    pw, ph = proposals[:, 2] - proposals[:, 0], proposals[:, 3] - proposals[:, 1]
    px, py = proposals[:, 0] + 0.5 * pw, proposals[:, 1] + 0.5 * ph
    gw, gh = gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]
    gx, gy = gt[:, 0] + 0.5 * gw, gt[:, 1] + 0.5 * gh
    t = np.stack([(gx - px) / pw, (gy - py) / ph, np.log(gw / pw), np.log(gh / ph)], 1)
    return (t / BOX_STD).astype(np.float32)


def decode(proposals, deltas):
    """Apply predicted (normalised) deltas to proposals. Both are numpy (N, 4)."""
    t = deltas * BOX_STD
    t[:, 2:] = np.clip(t[:, 2:], -4.0, 4.0)
    pw, ph = proposals[:, 2] - proposals[:, 0], proposals[:, 3] - proposals[:, 1]
    px, py = proposals[:, 0] + 0.5 * pw, proposals[:, 1] + 0.5 * ph
    cx, cy = px + t[:, 0] * pw, py + t[:, 1] * ph
    w, h = pw * np.exp(t[:, 2]), ph * np.exp(t[:, 3])
    return np.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], 1)


def paired_iou(a, b):
    """IoU between a[i] and b[i] for every i."""
    lt = np.maximum(a[:, :2], b[:, :2])
    rb = np.minimum(a[:, 2:], b[:, 2:])
    inter = np.prod((rb - lt).clip(0), 1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a) + area(b) - inter + 1e-9)


class ProposalClassifier(nn.Module):
    """ImageNet-pretrained ResNet-50 with an (N+1)-way classifier and a class-specific box regressor."""

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = torchvision.models.resnet50(weights=weights)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.cls_head = nn.Linear(feat_dim, num_classes)
        self.box_head = nn.Linear(feat_dim, 4)  # only one foreground class -> one set of deltas
        nn.init.normal_(self.box_head.weight, std=0.001)
        nn.init.zeros_(self.box_head.bias)

    def forward(self, x):
        f = self.backbone(x)
        return self.cls_head(f), self.box_head(f)


def to_input(batch_uint8, device):
    """(B, H, W, 3) uint8 tensor -> normalised float (B, 3, H, W) on device."""
    x = batch_uint8.to(device, non_blocking=True).permute(0, 3, 1, 2).float()
    return (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)


@torch.no_grad()
def predict_crops(model, crops_loader, device):
    """Returns pothole probabilities and box deltas for every crop in the loader."""
    model.eval()
    probs, deltas = [], []
    for batch in crops_loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits, d = model(to_input(x, device))
        probs.append(torch.softmax(logits.float(), 1)[:, 1].cpu())
        deltas.append(d.float().cpu())
    return torch.cat(probs).numpy(), torch.cat(deltas).numpy()
