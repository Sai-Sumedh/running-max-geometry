"""Theorem 1 curves of Fig. 2(c, d) for chosen anchor numbers.

    python -m analysis.anchors --runs baseline_seed0 baseline_seed1 baseline_seed2 baseline_seed3

Anchors are token values (number = token + 1); the default 299 and 699 are the numbers 300 and 700.
Writes outputs/results/<run>_anchors.npz with keys anchors, cosEE_a, lbEE_a, cosEU_a, ubEU_a
(shape: n_anchors x V).
"""

import argparse
import os

import numpy as np

from analysis.analyze import OUT, load_model, log, table


def anchor_curves(E, U, anchors):
    """cos(E_i, E_j) with its Eq. 4 lower bound, and cos(E_i, U_j) with its Eq. 5 upper bound."""
    nE, nU = np.linalg.norm(E, axis=1), np.linalg.norm(U, axis=1)
    En, Un = E / nE[:, None], U / nU[:, None]
    m = np.asarray(anchors)
    cosEE = En[m] @ En.T
    lbEE = np.minimum(nE[m][:, None], nE[None]) / np.maximum(nE[m][:, None], nE[None])
    cosEU = En[m] @ Un.T
    ubEU = cosEU[np.arange(len(m)), m][:, None] * nU[m][:, None] / nU[None]
    return dict(anchors=m, cosEE_a=cosEE, lbEE_a=lbEE, cosEU_a=cosEU, ubEU_a=ubEU)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', default=['baseline_seed0'])
    ap.add_argument('--anchors', nargs='+', type=int, default=[299, 699])   # numbers 300 and 700
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for run in args.runs:
        model = load_model(run)[0]
        out = anchor_curves(table(model, 'We'), table(model, 'Wd'), args.anchors)
        np.savez_compressed(os.path.join(OUT, f'{run}_anchors.npz'), **out)
        log(f'{run}: anchors {args.anchors} written')
    log('done')
