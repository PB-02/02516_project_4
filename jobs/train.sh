#!/bin/bash
#BSUB -J rcnn_train
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 02:00
#BSUB -o logs/rcnn_train_%J.out
#BSUB -e logs/rcnn_train_%J.err
cd ~/02516_project_4
.venv/bin/python src/02516_project_4/train.py "$@"
