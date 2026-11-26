"""Every two-token sequence: accuracy, attention on the maximum, and the margins of Eqs. 1, 7 and 8.

    python -m analysis.pairs --runs baseline_seed0 baseline_seed1 baseline_seed2 baseline_seed3

For all 499,500 pairs of distinct numbers and both input orders: larger first, where the query (the second,
smaller token) must retrieve the earlier, larger token; and larger second, where the query is the maximum.
In the code, q is the larger and m the smaller number of a pair. Writes outputs/results/<run>_pairs.npz
(per-pair arrays) and <run>_pairs.json (summary). GPU if visible.
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from analysis.analyze import DEV, OUT, fit_pca, log, load_model, logits_from_embed, table, to_json


@torch.no_grad()
def second_position(model, first, second, bs=50_000):
    """Prediction and attention weight on position 0, both at position 1 of the sequence (first, second)."""
    preds, w0 = [], []
    d = model.We.weight.shape[1]
    for i in range(0, len(first), bs):
        ids = torch.stack([first[i:i + bs], second[i:i + bs]], 1).to(DEV)
        h = model.We(ids)
        preds.append(logits_from_embed(model, h)[:, 1].argmax(-1).cpu())
        s = (h[:, 1:2] @ h.transpose(1, 2)).squeeze(1) / d ** 0.5           # query = token at position 1
        w0.append(torch.softmax(s, -1)[:, 0].cpu())
    return torch.cat(preds).numpy(), torch.cat(w0).numpy()


def auc(y, score):
    return float(roc_auc_score(y, score)) if 0 < y.mean() < 1 else None


def analyse(run):
    model = load_model(run)[0]
    assert not model.dist_attn and model.decoder_type == 'linear'
    E = table(model)
    V, d = E.shape
    p = fit_pca(E)[1][0]
    a = E @ p
    R = E - np.outer(a, p)
    S, Gr = E @ E.T, R @ R.T
    q, m = np.tril_indices(V, -1)                                          # all pairs q > m
    qt, mt = torch.tensor(q), torch.tensor(m)
    arr, summ = dict(q=q.astype(np.int16), m=m.astype(np.int16)), dict(run=run, n_pairs=int(len(q)))
    for order, (first, second, query) in (('lf', (qt, mt, m)), ('ls', (mt, qt, q))):
        pred, w0 = second_position(model, first, second)
        C = a[query] * (a[q] - a[m])                                       # PC1 term of Eq. 7
        I = Gr[query, q] - Gr[query, m]                                    # residual term of Eq. 7
        delta = (S[query, q] - S[query, m]) / np.sqrt(d)                   # score(max) - score(other)
        assert np.allclose(C + I, S[query, q] - S[query, m], atol=1e-6)
        p_max = w0 if order == 'lf' else 1 - w0                            # the max sits at position 0 for 'lf'
        assert np.allclose(p_max, 1 / (1 + np.exp(-delta)), atol=1e-4)     # two-token attention = sigmoid(delta)
        ok, eq1, eq8, gap = pred == q, delta > 0, C > np.abs(I), q - m
        arr.update({f'{order}_delta': delta.astype(np.float32), f'{order}_C': C.astype(np.float32),
                    f'{order}_I': I.astype(np.float32), f'{order}_correct': ok, f'{order}_pred_m': pred == m})
        s = dict(acc=ok.mean(), acc_gap1=ok[gap == 1].mean(), acc_gap_le5=ok[gap <= 5].mean(),
                 acc_gap_ge20=ok[gap >= 20].mean(), frac_eq1=eq1.mean(), frac_eq8=eq8.mean(),
                 frac_eq8_gap1=eq8[gap == 1].mean(), mean_pmax=p_max.mean(),
                 frac_eq1_but_wrong=(eq1 & ~ok).mean(), frac_not_eq1_but_correct=(~eq1 & ok).mean(),
                 acc_given_eq1=ok[eq1].mean(), acc_given_not_eq1=ok[~eq1].mean() if (~eq1).any() else None,
                 frac_errors_output_m=(pred == m)[~ok].mean() if (~ok).any() else None,
                 auc_delta=auc(ok, delta), auc_gap=auc(ok, gap), auc_eq8=auc(ok, eq8.astype(float)),
                 auc_pc1_term=auc(ok, C))
        for lo, hi, lab in ((0, 333, 'low'), (333, 666, 'mid'), (666, V, 'high')):
            sel = (q >= lo) & (q < hi)
            s[f'acc_max_{lab}'] = ok[sel].mean()
            s[f'acc_gap1_max_{lab}'] = ok[sel & (gap == 1)].mean()
        summ[order] = {k: (float(v) if v is not None else None) for k, v in s.items()}
        log(f'{run} {order}: acc {s["acc"]:.4f}, gap 1 {s["acc_gap1"]:.3f}, Eq1 {s["frac_eq1"]:.4f}, '
            f'Eq8 {s["frac_eq8"]:.4f}, AUC delta {summ[order]["auc_delta"]}, AUC gap {summ[order]["auc_gap"]}')
    np.savez_compressed(os.path.join(OUT, f'{run}_pairs.npz'), **arr)
    with open(os.path.join(OUT, f'{run}_pairs.json'), 'w') as f:
        json.dump(to_json(summ), f, indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', default=['baseline_seed0'])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for run in args.runs:
        analyse(run)
    log('done')
