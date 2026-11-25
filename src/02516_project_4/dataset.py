"""Potholes dataset: XML parsing, shared train/test split and visualisation.

Run as a script to print dataset statistics and save example images with
ground-truth boxes:
    uv run python src/02516_project_4/dataset.py
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

DATA_ROOT = "/dtu/datasets1/02516/potholes"
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
ANNOTATION_DIR = os.path.join(DATA_ROOT, "annotations")

TEST_SIZE = 0.1
RANDOM_STATE = 42  # shared across all groups (see Slack)


def parse_annotation(xml_path):
    """Parse one PascalVOC XML file.

    Returns a dict with filename, width, height and boxes as an (N, 4) int
    array in [xmin, ymin, xmax, ymax] pixel coordinates.
    """
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        boxes.append([int(float(bb.find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax")])
    return {
        "filename": root.find("filename").text,
        "width": int(size.find("width").text),
        "height": int(size.find("height").text),
        "boxes": np.array(boxes, dtype=np.int64).reshape(-1, 4),
    }


MIN_BOX_SIDE = 3  # px; drops the two near-empty annotations (potholes58, potholes483)


def clean_boxes(boxes, width, height):
    """Clip boxes to the image and drop degenerate ones."""
    boxes = boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    keep = ((boxes[:, 2] - boxes[:, 0]) >= MIN_BOX_SIDE) & ((boxes[:, 3] - boxes[:, 1]) >= MIN_BOX_SIDE)
    return boxes[keep]


def load_dataframe(clean=True):
    """One row per image, sorted by XML filename so every group gets the same order."""
    rows = []
    for xml_name in sorted(os.listdir(ANNOTATION_DIR)):
        if not xml_name.endswith(".xml"):
            continue
        ann = parse_annotation(os.path.join(ANNOTATION_DIR, xml_name))
        if clean:
            ann["boxes"] = clean_boxes(ann["boxes"], ann["width"], ann["height"])
        ann["image_path"] = os.path.join(IMAGE_DIR, ann["filename"])
        ann["n_boxes"] = len(ann["boxes"])
        rows.append(ann)
    return pd.DataFrame(rows)


def get_splits(df=None):
    """Train/test split agreed across groups: test_size=0.1, random_state=42."""
    if df is None:
        df = load_dataframe()
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_train_val_test(df=None, val_size=0.1):
    """Shared train/test split, then a validation set carved out of the training part (image level)."""
    train_df, test_df = get_splits(df)
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=RANDOM_STATE)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df


class PotholesDataset(torch.utils.data.Dataset):
    """Full images with their ground-truth boxes (for proposals / evaluation)."""

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        boxes = torch.as_tensor(row["boxes"], dtype=torch.float32)
        if self.transform is not None:
            image = self.transform(image)
        return image, boxes


def show_examples(df, n=6, out_path="figures/gt_examples.png", seed=0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    sample = df.sample(n=n, random_state=seed)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    for ax, (_, row) in zip(axes.ravel(), sample.iterrows()):
        ax.imshow(Image.open(row["image_path"]).convert("RGB"))
        for x1, y1, x2, y2 in row["boxes"]:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2))
        ax.set_title(f"{row['filename']} ({row['n_boxes']} potholes)")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    df = load_dataframe()
    train_df, test_df = get_splits(df)
    print(f"Images: {len(df)}  train: {len(train_df)}  test: {len(test_df)}")
    print(f"Boxes: {df['n_boxes'].sum()}  per image: mean {df['n_boxes'].mean():.2f}, "
          f"min {df['n_boxes'].min()}, max {df['n_boxes'].max()}")
    all_boxes = np.concatenate(df["boxes"].to_list())
    w = all_boxes[:, 2] - all_boxes[:, 0]
    h = all_boxes[:, 3] - all_boxes[:, 1]
    print(f"Box width  median {np.median(w):.0f}px (min {w.min()}, max {w.max()})")
    print(f"Box height median {np.median(h):.0f}px (min {h.min()}, max {h.max()})")
    print(f"Image sizes: {df.groupby(['width', 'height']).size().sort_values(ascending=False).head().to_dict()}")
    print("First test images:", test_df["filename"].head().to_list())
    show_examples(train_df)
