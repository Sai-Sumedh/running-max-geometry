"""Output locations.

Everything is written under $RMG_OUT, which defaults to ./outputs at the root of this repository:
    data/<dataset>/{train,val}.pt      running-max datasets
    checkpoints/<run>/ep<k>.pt         model checkpoints, one per epoch
    results/                           analysis results (.npz arrays, .json summaries)
    figures/                           figures (.pdf) and tables (.csv, .md)
"""

import os

ROOT = os.environ.get('RMG_OUT', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs'))
DATA = os.path.join(ROOT, 'data')
CKPT = os.path.join(ROOT, 'checkpoints')
RESULTS = os.path.join(ROOT, 'results')
FIGURES = os.path.join(ROOT, 'figures')
