# 02516 Project 4 – Pothole detection (R-CNN style)

Setup: `uv sync` (torch is pinned to the CUDA 12.6 build so it runs on V100/A100/L40s).

| Step | Command | Where |
|---|---|---|
| 4.1 data stats + GT figure | `uv run python src/02516_project_4/dataset.py` | login node |
| 4.1 Selective Search, evaluation, labels | `bsub < jobs/proposals.sh` | CPU job (~10 min) |
| 4.1 only re-label (other k / IoU) | `bsub < jobs/labels.sh` | CPU job |
| 4.2 train classifier + box regressor | `bsub < jobs/train.sh` (or `.venv/bin/python src/02516_project_4/train.py` in `voltash`) | GPU (~9 min on V100) |
| 4.3 score val/test proposals | `.venv/bin/python src/02516_project_4/detect.py infer` | GPU |
| 4.3 NMS + AP + figures | `uv run python src/02516_project_4/detect.py evaluate` | login node |
| report | `cd report && latexmk -pdf report.tex` | login node |

Split: images sorted by filename, `train_test_split(test_size=0.1, random_state=42)` (598 train / 67 test);
validation = 10 % of train with the same seed (538 / 60).
Large outputs (proposals, labels, model weights) go to `outputs/` and are not committed.
