#!/bin/bash
#BSUB -J ss_labels
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 00:30
#BSUB -o logs/ss_labels_%J.out
#BSUB -e logs/ss_labels_%J.err
cd ~/02516_project_4
.venv/bin/python src/02516_project_4/labels.py --mode quality --k 2000
