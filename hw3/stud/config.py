"""
Global configuration variables for the project.
"""

from pathlib import Path
import os
# import sys

ROOT: Path = Path(__file__).parent.parent.parent
DATA: Path = ROOT / 'data'
MODEL: Path = ROOT / 'model'

HW3: Path = ROOT / 'hw3'
TRAIN: Path = DATA / 'train.tsv'
DEV: Path = DATA / 'dev.tsv'

CKP: Path = ROOT / 'checkpoints'

VOCAB: Path = MODEL / 'vocab-glove.pkl'
NERMODEL: Path = MODEL / '7597-stacked-100h-crf.pth'
CRFMODEL: Path = MODEL / 'crf-7597.pth'

SEED: int = 42

# set current directory to ROOT
os.chdir(HW3)
# sys.path.append(str(ROOT))
