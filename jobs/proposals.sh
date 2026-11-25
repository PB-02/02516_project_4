#!/bin/bash
#BSUB -J ss_proposals
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 01:30
#BSUB -o logs/ss_proposals_%J.out
#BSUB -e logs/ss_proposals_%J.err
cd ~/02516_project_4
export OMP_NUM_THREADS=1
.venv/bin/python src/02516_project_4/proposals.py extract --mode fast --workers 8
.venv/bin/python src/02516_project_4/proposals.py extract --mode quality --workers 8
.venv/bin/python src/02516_project_4/proposals.py evaluate
.venv/bin/python src/02516_project_4/labels.py --mode quality --k 2000
