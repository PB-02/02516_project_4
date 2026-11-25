"""Plot training/validation curves from outputs/train_results.json."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

h = json.load(open("outputs/train_results.json"))["history"]
ep = [r["epoch"] for r in h]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
axes[0].plot(ep, [r["train_cls_loss"] for r in h], marker=".", label="train classification loss")
axes[0].plot(ep, [r["train_box_loss"] for r in h], marker=".", label="train box-regression loss")
axes[0].set_xlabel("epoch")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)
for key, name in (("val_proposal_ap", "proposal AP"), ("val_balanced_accuracy", "balanced accuracy"),
                  ("val_pothole_recall", "pothole recall"), ("val_pothole_precision", "pothole precision")):
    axes[1].plot(ep, [r[key] for r in h], marker=".", label=name)
axes[1].set_xlabel("epoch")
axes[1].set_title("validation subset (all positives + 15k background)", fontsize=9)
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figures/training_curves.png", dpi=150)
print("Saved figures/training_curves.png")
