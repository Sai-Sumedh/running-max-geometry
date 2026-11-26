"""Analyses of one trained run: geometry, Theorem 1 conditions, patching, decoy interchange, attention
saturation, interference budget, order directions without PCA, and translation symmetry.

    python -m analysis.analyze --run baseline_seed0 --epochs --extras
    python -m analysis.analyze --run baseline_seed1

Runs on a GPU when one is visible (forward passes), CPU otherwise. Writes outputs/results/<run>.npz
(arrays for the figures) and outputs/results/<run>.json (scalars for the tables). --epochs adds the
per-epoch metrics (needs all epoch checkpoints); --extras adds checks on the two architectural variants
(needs the runs dist_attention and dist_readout).
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_score

from rmg import paths
from rmg.model import TxformerSimple

OUT = paths.RESULTS
DATA = 'running_max_500k'
ANCHORS = [99, 499, 899]          # token values; shown as m = 100, 500, 900 (number = token + 1)
M0 = 499                          # anchor i = 500 of Fig. 3(d, e)
T_OVER, GAP1, GAP2, POS = 20, 5, 5, 1
N_STEER, N_IIA, N_SAT, N_RAND, N_BOOT = 500, 1000, 2000, 100, 10000
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_data(name, split='val'):
    return torch.load(os.path.join(paths.DATA, name, f'{split}.pt'), weights_only=True)


def load_model(run, epoch=None):
    loc = os.path.join(paths.CKPT, run)
    if epoch is None:
        epoch = max(int(f[2:-3]) for f in os.listdir(loc) if f.endswith('.pt'))
    ck = torch.load(os.path.join(loc, f'ep{epoch}.pt'), map_location='cpu', weights_only=False)
    cfg, sd = ck['config'], ck['model_state_dict']
    model = TxformerSimple(sd['We.weight'].shape[0], cfg['embed_dim'], decoder_type=cfg['decoder_type'],
                           dist_attn=cfg['dist_attn'])
    model.load_state_dict(sd)
    model.to(DEV).eval()
    return model, cfg, epoch


def table(model, which='We'):
    """Embedding (We) or unembedding (Wd) table as a float64 numpy array."""
    return getattr(model, which).weight.detach().cpu().double().numpy()


def logits_from_embed(model, h):
    """Token embeddings (B, T, d) -> logits (B, T, K); the forward pass after the lookup."""
    return model.decode(model.attn(h))


@torch.no_grad()
def predict(model, ids, bs=500):
    return torch.cat([logits_from_embed(model, model.We(ids[i:i + bs].to(DEV))).argmax(-1).cpu()
                      for i in range(0, len(ids), bs)])


def running_max(x):
    return np.maximum.accumulate(x, axis=-1)


def boot_idx(n, seed=0):
    return np.random.default_rng(seed).integers(0, n, (N_BOOT, n))


def boot(num, den, idx):
    """Pooled ratio sum(num)/sum(den) and its 95% percentile CI, resampling sequences."""
    pt = num.sum() / den.sum()
    bs = num[idx].sum(1) / den[idx].sum(1)
    return [float(pt), *map(float, np.percentile(bs, [2.5, 97.5]))]


def boot_diff(n1, d1, n0, d0, idx):
    pt = n1.sum() / d1.sum() - n0.sum() / d0.sum()
    bs = n1[idx].sum(1) / d1[idx].sum(1) - n0[idx].sum(1) / d0[idx].sum(1)
    return [float(pt), *map(float, np.percentile(bs, [2.5, 97.5]))]


# ------------------------------------------------------------------ geometry (Figs. 2, 3)

def fit_pca(E):
    """Full-solver PCA of E; PC1 oriented so that its score rises with number."""
    pca = PCA(svd_solver='full').fit(E)
    comps = pca.components_.copy()
    scores = (E - pca.mean_) @ comps.T
    if np.corrcoef(scores[:, 0], np.arange(len(E)))[0, 1] < 0:
        comps[0], scores[:, 0] = -comps[0], -scores[:, 0]
    return pca, comps, scores


def geometry(E, U):
    pca, comps, scores = fit_pca(E)
    p = comps[0]
    a, b = E @ p, U @ p                           # uncentered coordinates along PC1 (Theorem 3)
    R, S = E - np.outer(a, p), U - np.outer(b, p)  # transverse parts r_i, s_i
    nE, nU, nR = (np.linalg.norm(X, axis=1) for X in (E, U, R))
    En, Un = E / nE[:, None], U / nU[:, None]
    m = np.array(ANCHORS)
    cosEE = En[m] @ En.T
    lbEE = np.minimum(nE[m][:, None], nE[None]) / np.maximum(nE[m][:, None], nE[None])
    cosEU = En[m] @ Un.T
    ubEU = cosEU[np.arange(len(m)), m][:, None] * nU[m][:, None] / nU[None]
    G = E @ E.T
    Gp = np.outer(a, a)
    ss_tot = ((G - G.mean(1, keepdims=True)) ** 2).sum(1)
    out = dict(
        var_ratio=pca.explained_variance_ratio_, pc_scores=scores[:, :5], p=p, a=a, b=b,
        normE=nE, normU=nU, normR=nR, cosEE=cosEE, lbEE=lbEE, cosEU=cosEU, ubEU=ubEU,
        dotEE_full=G[M0], dotEE_pc1=a[M0] * a, dotEE_low=R[M0] @ R.T,
        dotEU_full=E[M0] @ U.T, dotEU_pc1=a[M0] * b, dotEU_low=R[M0] @ U.T,
        cosdiag_full=(E * U).sum(1) / (nE * nU), cosdiag_pc1=np.sign(a) * b / nU,
        cosdiag_low=(R * U).sum(1) / (nR * nU),
        fve_pc1=1 - ((G - Gp) ** 2).sum(1) / ss_tot, fve_low=1 - (Gp ** 2).sum(1) / ss_tot,
    )
    # Theorem 1 conditions per anchor i over all j != i (Fig. 6); Eq. 4 uses the symmetric bound min/max of the norms
    V = len(E)
    off = ~np.eye(V, dtype=bool)
    cosEE_all = En @ En.T
    lb_all = np.minimum(nE[:, None], nE[None]) / np.maximum(nE[:, None], nE[None])
    MU = E @ U.T
    upper = np.triu(np.ones((V, V), dtype=bool), 1)
    out['frac_eq3'] = np.array([(nE[i + 1:] > nE[i]).mean() if i < V - 1 else np.nan for i in range(V)])
    out['frac_eq4'] = np.where(off, cosEE_all > lb_all, False).sum(1) / (V - 1)
    out['frac_eq5'] = np.where(off, MU < np.diag(MU)[:, None], False).sum(1) / (V - 1)
    summ = dict(var_ratio_pc1=float(out['var_ratio'][0]), var_ratio_pc2=float(out['var_ratio'][1]),
                spearman_pc1=float(spearmanr(scores[:, 0], np.arange(V))[0]),
                frac_eq3_pairs=float((nE[None] > nE[:, None])[upper].mean()),
                frac_eq4_pairs=float((cosEE_all > lb_all)[upper].mean()),
                frac_eq5_pairs=float((MU < np.diag(MU)[:, None])[off].mean()),
                a_min=float(a.min()), a_max=float(a.max()),
                fve_pc1_median=float(np.median(out['fve_pc1'])),
                fve_low_median=float(np.median(out['fve_low'])),
                cosdiag_low_median=float(np.median(out['cosdiag_low'])),
                cosdiag_full_median=float(np.median(out['cosdiag_full'])),
                cosdiag_pc1_median=float(np.median(out['cosdiag_pc1'])))
    return out, summ, comps, R, S


# ------------------------------------------------------------------ accuracy

def accuracy(model, data, n=None):
    ids, tg = data['inputs'][:n], data['targets'][:n]
    corr = (predict(model, ids) == tg).numpy()
    tgf = tg.numpy().ravel()
    V = model.We.weight.shape[0]
    tot = np.bincount(tgf, minlength=V)
    cor = np.bincount(tgf[corr.ravel()], minlength=V)
    per_seq = corr.mean(1)
    se = per_seq.std(ddof=1) / np.sqrt(len(per_seq))
    acc = float(corr.mean())
    return dict(acc=acc, acc_ci=[acc - 1.96 * se, acc + 1.96 * se], n_seq=int(len(per_seq))), tot, cor


# ------------------------------------------------------------------ patching toward a new value (Figs. 5, 7)

def construct_steer_targets(dataset, n_examples, t_overtake=20, gap1=5, gap2=5, pos_steer=1,
                            min_below_frac=1 / 3):
    """Patching examples and targets from held-out sequences (App. A)."""
    V = dataset['vocab_size']
    inputs = dataset['inputs'].numpy()
    n_required = max(1, round(min_below_frac * t_overtake))
    found = []
    for seq_idx in range(len(inputs)):
        if len(found) >= n_examples:
            break
        x = inputs[seq_idx].copy()
        v_orig = int(x[pos_steer])
        if v_orig == 0:
            continue
        x[0] = 0
        required_idx = np.arange(pos_steer + 1, min(pos_steer + 1 + n_required, t_overtake))
        violate = x[required_idx] >= v_orig
        x[required_idx[violate]] = 0
        base_max = int(max(x[pos_steer], x[pos_steer + 1:t_overtake].max()))
        new_v = base_max + gap1
        overtake_val = new_v + gap2
        if overtake_val > V - 1:
            continue
        x[t_overtake] = overtake_val
        found.append(dict(seq_idx=seq_idx, input=x, pos_steer=pos_steer, v_orig=v_orig,
                          t_overtake=t_overtake, overtake_val=overtake_val, new_v=new_v))
    return found


def steer_setup(data):
    ex = construct_steer_targets(data, N_STEER + 1, T_OVER, GAP1, GAP2, POS)
    ids = np.stack([e['input'] for e in ex])
    v_orig = np.array([e['v_orig'] for e in ex])
    v_new = np.array([e['new_v'] for e in ex])
    y_true = running_max(ids)
    ids_p = ids.copy()
    ids_p[:, POS] = v_new
    y_patch = running_max(ids_p)
    assert np.array_equal(y_patch[:, T_OVER:], y_true[:, T_OVER:])
    T = ids.shape[1]
    t = np.arange(T)[None]
    win = (t >= POS) & (t <= T_OVER) & np.ones_like(ids, dtype=bool)
    st = dict(ids=ids, v_orig=v_orig, v_new=v_new, y_true=y_true, y_patch=y_patch, win=win,
              A=win & (y_true == v_orig[:, None]),                    # unpatched max is still v_orig
              B=win & (y_true != v_orig[:, None]) & (t < T_OVER),     # natural overtake, before T_OVER
              targets=dict(whole=y_patch,
                           pc1=np.where(y_patch == v_new[:, None], v_orig[:, None], y_patch),
                           comp=np.where(y_true == v_orig[:, None], v_new[:, None], y_true)))
    # demo = first example (as in the paper), the rest are held out for the aggregate
    return {k: (v[1:] if isinstance(v, np.ndarray) else {kk: vv[1:] for kk, vv in v.items()})
            for k, v in st.items()}, {k: v[0] for k, v in st.items() if isinstance(v, np.ndarray)}


@torch.no_grad()
def patched_logits(model, ids, pos, src, u, mode):
    """Patch the embedding at position pos[b] of row b. src: (B, d) source embeddings.
    whole: h' = s; dir: h' = h + u u^T (s - h); comp: h' = s + u u^T (h - s)."""
    h = model.We(ids).clone()
    rows = torch.arange(len(ids), device=ids.device)
    hp = h[rows, pos]
    if mode == 'whole':
        new = src
    elif mode == 'dir':
        new = hp + ((src - hp) @ u)[:, None] * u[None]
    elif mode == 'comp':
        new = src + ((hp - src) @ u)[:, None] * u[None]
    else:
        raise ValueError(mode)
    h[rows, pos] = new
    return logits_from_embed(model, h), (src - hp) @ u if mode != 'whole' else None


def steer_eval(P, st):
    """Per-sequence (num, den) counts of every metric for predictions P (N, T)."""
    vn, vo, yt = st['v_new'][:, None], st['v_orig'][:, None], st['y_true']
    A, B, win = st['A'], st['B'], st['win']
    res = {}
    for name, T in st['targets'].items():
        ok = P == T
        res[f'win_{name}'] = ((ok & win).sum(1), win.sum(1))
        ch = win & (T != yt)
        res[f'chg_{name}'] = ((ok & ch).sum(1), ch.sum(1))
    res['order'] = ((((P == vn) | (P == vo)) & B).sum(1), B.sum(1))
    res['identity'] = (((P == vn) & A).sum(1), A.sum(1))
    for lab, c in (('vnew', P == vn), ('vorig', P == vo), ('ytrue', P == yt)):
        res[f'A_{lab}'] = ((c & A).sum(1), A.sum(1))
        res[f'B_{lab}'] = ((c & B).sum(1), B.sum(1))
    return res


def steering(model, E, U, comps, data, Gr, a):
    st, demo = steer_setup(data)
    ids = torch.tensor(st['ids'], device=DEV)
    N = len(ids)
    pos = torch.full((N,), POS, device=DEV)
    Et = torch.tensor(E, dtype=torch.float32, device=DEV)
    src = Et[torch.tensor(st['v_new'], device=DEV)]
    p1 = torch.tensor(comps[0], dtype=torch.float32, device=DEV)
    preds = {'none': predict(model, ids).numpy()}
    for mode, key in (('whole', 'whole'), ('dir', 'pc1'), ('comp', 'comp')):
        preds[key] = patched_logits(model, ids, pos, src, p1, mode)[0].argmax(-1).cpu().numpy()
    idx = boot_idx(N)
    ev = {k: steer_eval(P, st) for k, P in preds.items()}
    summ = {}
    for k in preds:
        for met, (num, den) in ev[k].items():
            summ[f'{k}/{met}'] = boot(num, den, idx)
            if k != 'none':
                summ[f'{k}/{met}/effect'] = boot_diff(num, den, *ev['none'][met], idx)
    summ['n_examples'] = N
    summ['n_A_positions'], summ['n_B_positions'] = int(st['A'].sum()), int(st['B'].sum())
    ok_A = preds['none'] == st['v_orig'][:, None]            # no patch, stretch A, by query token
    for lab, qsel in (('query0', st['ids'] == 0), ('query_nonzero', st['ids'] != 0)):
        summ[f'none_A_vorig_{lab}'] = float(ok_A[st['A'] & qsel].mean())
        summ[f'n_A_{lab}'] = int((st['A'] & qsel).sum())
    # PC1-only in stretch B: does the hybrid vector a_new p + r_orig still win attention?
    Bm = st['B']
    bi, ti = np.nonzero(Bm)
    hyb = E[st['v_orig']] - np.outer(a[st['v_orig']], comps[0]) + np.outer(a[st['v_new']], comps[0])
    mq = st['ids'][bi, ti]                                   # query token m at position t
    s_hyb = (E[mq] * hyb[bi]).sum(1)
    pref = st['ids'].copy()
    pref[:, POS] = -1                                        # competitors: every unpatched prefix token
    S = E @ E.T
    s_comp = np.full(len(bi), -np.inf)
    for j in range(st['ids'].shape[1]):
        live = (j <= ti) & (pref[bi, j] >= 0)
        s_comp[live] = np.maximum(s_comp[live], S[mq[live], pref[bi[live], j]])
    u_nat = st['y_true'][bi, ti]                             # natural running max (main competitor)
    arr = dict(
        steer_B_gap=st['v_new'][bi] - u_nat, steer_B_agap=a[st['v_new'][bi]] - a[u_nat],
        steer_B_pc1_vorig=preds['pc1'][bi, ti] == st['v_orig'][bi],
        steer_B_pc1_ytrue=preds['pc1'][bi, ti] == u_nat,
        steer_B_whole_vnew=preds['whole'][bi, ti] == st['v_new'][bi],
        steer_B_hybrid_wins=s_hyb > s_comp,
        steer_B_eq8_hybrid=(a[mq] * (a[st['v_new'][bi]] - a[u_nat])
                            > np.abs(Gr[mq, st['v_orig'][bi]] - Gr[mq, u_nat])),
        steer_B_seq=bi, steer_B_t=ti,
        demo_input=demo['ids'], demo_ytrue=demo['y_true'], demo_vorig=demo['v_orig'],
        demo_vnew=demo['v_new'],
        first30_input=st['ids'][:30], first30_ytrue=st['y_true'][:30], first30_vorig=st['v_orig'][:30],
        first30_vnew=st['v_new'][:30], **{f'first30_{k}': P[:30] for k, P in preds.items()},
    )
    demo_ids = torch.tensor(demo['ids'], device=DEV)[None]
    dsrc = Et[torch.tensor([demo['v_new']], device=DEV)]
    arr['demo_none'] = predict(model, demo_ids).numpy()[0]
    for mode, key in (('whole', 'whole'), ('dir', 'pc1'), ('comp', 'comp')):
        arr[f'demo_{key}'] = patched_logits(model, demo_ids, torch.tensor([POS], device=DEV), dsrc, p1,
                                            mode)[0].argmax(-1).cpu().numpy()[0]
    hyb_top = (hyb @ U.T).argmax(1)                         # identity read out if attention picked the hybrid
    arr['steer_hybrid_decodes_vorig'] = hyb_top == st['v_orig']
    arr['steer_B_hybrid_decodes_vorig'] = arr['steer_hybrid_decodes_vorig'][bi]
    summ['frac_hybrid_decodes_vorig'] = float(arr['steer_hybrid_decodes_vorig'].mean())
    summ['frac_B_hybrid_wins'] = float(arr['steer_B_hybrid_wins'].mean())
    summ['pc1_ok_given_hybrid_wins'] = float(arr['steer_B_pc1_vorig'][arr['steer_B_hybrid_wins']].mean())
    summ['pc1_ok_given_hybrid_loses'] = float(arr['steer_B_pc1_vorig'][~arr['steer_B_hybrid_wins']].mean())
    # control directions: PC2 and N_RAND random unit vectors, same two rank-1 patch types
    rng = np.random.default_rng(0)
    rnd = rng.standard_normal((N_RAND, E.shape[1]))
    dirs = np.vstack([comps[0], comps[1], rnd / np.linalg.norm(rnd, axis=1, keepdims=True)])
    ctrl = np.zeros((len(dirs), 2, 5))   # dir x {dir, comp} x {order, identity, chg_whole, chg_pc1, chg_comp}
    mag = np.zeros(len(dirs))
    for i, u in enumerate(dirs):
        ut = torch.tensor(u, dtype=torch.float32, device=DEV)
        for j, mode in enumerate(('dir', 'comp')):
            lg, du = patched_logits(model, ids, pos, src, ut, mode)
            ev_u = steer_eval(lg.argmax(-1).cpu().numpy(), st)
            ctrl[i, j] = [ev_u[m][0].sum() / ev_u[m][1].sum()
                          for m in ('order', 'identity', 'chg_whole', 'chg_pc1', 'chg_comp')]
            mag[i] = du.abs().mean().item()
    arr.update(ctrl=ctrl, ctrl_mag=mag)
    return summ, arr


# ------------------------------------------------------------------ interchange with a decoy (Fig. 8)

def build_examples(dataset, n_examples, min_start_pos=5, min_dominance=3, read_offset=2, decoy_margin=10):
    """Clean and corrupted examples for the interchange with a decoy (App. A)."""
    inputs = dataset['inputs'].numpy()
    targets = dataset['targets'].numpy()
    T = inputs.shape[1]
    examples = []
    for seq_idx in range(len(inputs)):
        if len(examples) >= n_examples:
            break
        ytrue = targets[seq_idx]
        for t in range(min_start_pos + 1, T - 1):
            if ytrue[t] == ytrue[t - 1]:
                continue
            v_next = int(ytrue[t])
            v = int(ytrue[t - 1])
            t_v_start = int(np.where(ytrue == v)[0][0])
            if t_v_start < min_start_pos:
                continue
            if (t - t_v_start) < min_dominance:
                continue
            t_read = t_v_start + read_offset
            if t_read >= t:
                continue
            v_prev = int(ytrue[t_v_start - 1])
            v_c = max(0, v_prev - decoy_margin)
            examples.append(dict(seq_idx=seq_idx, v=v, v_next=v_next, v_prev=v_prev, v_c=v_c,
                                 t_v_start=t_v_start, t_trans=t, t_read=t_read))
            break
    return examples


@torch.no_grad()
def iia(model, E, p, data):
    ex = build_examples(data, N_IIA + 1)[1:]                 # first one is the paper's demo
    g = lambda k: torch.tensor([e[k] for e in ex], device=DEV)  # noqa: E731
    v, vp, vc, pp, pr = g('v'), g('v_prev'), g('v_c'), g('t_v_start'), g('t_read')
    ids_c = torch.stack([data['inputs'][e['seq_idx']] for e in ex]).to(DEV)
    ids_r = ids_c.clone()
    rows = torch.arange(len(ex), device=DEV)
    ids_r[rows, pp] = vc
    Et = torch.tensor(E, dtype=torch.float32, device=DEV)
    pt = torch.tensor(p, dtype=torch.float32, device=DEV)

    def read(lg):
        return lg[rows, pr]

    lc, lr = read(logits_from_embed(model, model.We(ids_c))), read(logits_from_embed(model, model.We(ids_r)))
    ld = lambda L: L[rows, v] - L[rows, vp]                  # noqa: E731
    ld_c, ld_r = ld(lc), ld(lr)
    idx = boot_idx(len(ex))
    one = np.ones(len(ex))
    summ = dict(n_examples=len(ex))
    arr = dict(iia_ld_clean=ld_c.cpu().numpy(), iia_ld_corr=ld_r.cpu().numpy(), iia_v=v.cpu().numpy(),
               iia_vprev=vp.cpu().numpy(), iia_vc=vc.cpu().numpy())
    for key, L in [('none', lr)] + [(k, read(patched_logits(model, ids_r, pp, Et[v], pt, m)[0]))
                                    for m, k in (('whole', 'whole'), ('dir', 'pc1'), ('comp', 'comp'))]:
        top = L.argmax(-1)
        for lab, tok in (('v', v), ('vc', vc), ('vprev', vp)):
            summ[f'{key}/top_{lab}'] = boot((top == tok).cpu().numpy().astype(float), one, idx)
        arr[f'iia_top_{key}'] = top.cpu().numpy()
        arr[f'iia_ld_{key}'] = ld(L).cpu().numpy()
        den = (ld_c - ld_r).cpu().numpy()
        ok = np.abs(den) > 1e-3
        rec = (ld(L) - ld_r).cpu().numpy()[ok] / den[ok]
        summ[f'{key}/recovery_median'] = float(np.median(rec))
        summ[f'{key}/recovery_iqr'] = list(map(float, np.percentile(rec, [25, 75])))
    summ['n_recovery_degenerate'] = int((~ok).sum())
    return summ, arr


# ------------------------------------------------------------------ saturation and Eqs. 1 and 8 on data (Fig. 9)

@torch.no_grad()
def saturation(model, E, a, Gr, data):
    x, y = data['inputs'][:N_SAT], data['targets'][:N_SAT]
    h = model.We(x.to(DEV))
    n, T, d = h.shape
    scores = h @ h.transpose(1, 2) / d ** 0.5
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=DEV))
    A = torch.softmax(scores.masked_fill(~causal, -torch.inf), -1)
    assert torch.allclose(A @ h, model.attn(h), atol=1e-4)
    pred = model.decode(A @ h).argmax(-1).cpu().numpy()
    A = A.cpu().numpy()
    xn, yn = x.numpy(), y.numpy()
    cz = causal.cpu().numpy()[None]
    is_max = (xn[:, None, :] == yn[:, :, None]) & cz
    smaller = (xn[:, None, :] < yn[:, :, None]) & cz
    mass = (A * is_max).sum(-1)
    gap = yn - np.where(smaller, xn[:, None, :], -1).max(-1)
    gap[~smaller.any(-1)] = -1
    S = E @ E.T
    m = np.broadcast_to(xn[:, :, None], smaller.shape)
    k = np.broadcast_to(xn[:, None, :], smaller.shape)
    q = np.broadcast_to(yn[:, :, None], smaller.shape)
    eq1 = S[xn, yn] > np.where(smaller, S[m, k], -np.inf).max(-1)
    eq8 = np.where(smaller, a[m] * (a[q] - a[k]) > np.abs(Gr[m, q] - Gr[m, k]), True).all(-1)
    keep = gap > 0
    t = np.broadcast_to(np.arange(T)[None], xn.shape)
    arr = dict(sat_mass=mass[keep].astype(np.float32), sat_gap=gap[keep], sat_q=yn[keep], sat_m=xn[keep],
               sat_t=t[keep], sat_correct=(pred == yn)[keep], sat_eq1=eq1[keep], sat_eq8=eq8[keep])
    summ = dict(sat_n=int(keep.sum()), sat_mass_median=float(np.median(mass[keep])),
                sat_frac_mass_gt_0p9=float((mass[keep] > 0.9).mean()),
                sat_frac_mass_gt_0p99=float((mass[keep] > 0.99).mean()),
                sat_frac_eq1=float(eq1[keep].mean()), sat_frac_eq8=float(eq8[keep].mean()),
                sat_acc=float((pred == yn)[keep].mean()),
                sat_acc_eq1_true=float((pred == yn)[keep & eq1].mean()),
                sat_acc_eq1_false=float((pred == yn)[keep & ~eq1].mean()) if (keep & ~eq1).any() else None)
    return summ, arr


# ------------------------------------------------------------------ Eqs. 1 and 8 and the 2R^2 bound over all vocabulary triples;
# ------------------------------------------------------------------ Eq. 2 and the readout condition of Theorem 3 (Fig. 10)

def vocab_conditions(E, U, a, b, R, S_):
    V = len(E)
    S = E @ E.T
    Gr = R @ R.T
    R2 = 2 * np.max(np.sum(R ** 2, 1))                       # the 2 R^2 bound (App. B)
    qbin = np.minimum(np.arange(V) // 100, 9)
    tot, ok1, ok8, ok56 = (np.zeros((10, V)) for _ in range(4))
    rng = np.random.default_rng(0)
    samp = []
    for q in range(1, V):
        mm, kk = np.arange(q + 1), np.arange(q)
        C = a[mm, None] * (a[q] - a[None, kk])
        I = Gr[mm, q][:, None] - Gr[np.ix_(mm, kk)]
        gaps = q - kk
        tot[qbin[q], gaps] += len(mm)
        ok1[qbin[q], gaps] += (S[mm, q][:, None] > S[np.ix_(mm, kk)]).sum(0)
        ok8[qbin[q], gaps] += (C > np.abs(I)).sum(0)
        ok56[qbin[q], gaps] += (C > R2).sum(0)
        i, j = rng.integers(0, q + 1, 100), rng.integers(0, q, 100)
        samp.append(np.stack([np.full(100, q), q - j, C[i, j], I[i, j]], 1))
    samp = np.concatenate(samp)
    # identity readout (Eq. 2) and its sufficient condition (Theorem 3)
    MU = E @ U.T
    RS = R @ S_.T
    off = ~np.eye(V, dtype=bool)
    eq2 = (np.diag(MU)[:, None] > MU) & off
    eq53 = ((np.diag(RS)[:, None] - RS) > np.abs(a[:, None] * (b[:, None] - b[None]))) & off
    arr = dict(tri_tot=tot, tri_ok1=ok1, tri_ok8=ok8, tri_ok56=ok56, tri_samp=samp.astype(np.float32),
               id_frac2=eq2.sum(1) / (V - 1), id_frac53=eq53.sum(1) / (V - 1))
    summ = dict(tri_frac_eq1=float(ok1.sum() / tot.sum()), tri_frac_eq8=float(ok8.sum() / tot.sum()),
                tri_frac_eq56=float(ok56.sum() / tot.sum()),
                tri_frac_eq1_gap1=float(ok1[:, 1].sum() / tot[:, 1].sum()),
                tri_frac_eq8_gap1=float(ok8[:, 1].sum() / tot[:, 1].sum()),
                id_frac_eq2=float(eq2.sum() / off.sum()), id_frac_eq53=float(eq53.sum() / off.sum()),
                id_frac_q_all_eq2=float((eq2.sum(1) == V - 1).mean()),
                id_frac_q_all_eq53=float((eq53.sum(1) == V - 1).mean()),
                b_cv=float(b.std() / np.abs(b).mean()))
    return summ, arr


# ------------------------------------------------------------------ PC1 against order directions found without PCA

def pc1_validation(E, p, n_boot=500):
    V = len(E)
    idx = np.arange(V, dtype=float)

    def dirs(X, y):
        dm = X[y >= V / 2].mean(0) - X[y < V / 2].mean(0)
        w = Ridge(alpha=alpha).fit(X, y).coef_
        pc = PCA(svd_solver='full').fit(X).components_[0]
        return pc, dm / np.linalg.norm(dm), w / np.linalg.norm(w)

    alpha = RidgeCV(alphas=np.logspace(-3, 4, 15)).fit(E, idx).alpha_
    cv_r2 = cross_val_score(Ridge(alpha=alpha), E, idx, cv=KFold(5, shuffle=True, random_state=0), scoring='r2')
    _, dm, w = dirs(E, idx)
    cos = lambda u, v_: float(abs(u @ v_))                   # noqa: E731
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(n_boot):
        r = rng.integers(0, V, V)
        pc_b, dm_b, w_b = dirs(E[r], idx[r])
        bs.append([cos(pc_b, dm_b), cos(pc_b, w_b), cos(pc_b, p)])
    bs = np.array(bs)
    Ec = E - E.mean(0)
    at = Ec @ dm
    rt = Ec - np.outer(at, dm)
    lam = np.linalg.eigvalsh(np.cov(rt.T)).max()
    c = (at[:, None] * rt).mean(0)
    return dict(ridge_alpha=float(alpha), ridge_cv_r2=float(cv_r2.mean()),
                cos_pc1_dm=cos(p, dm), cos_pc1_dm_ci=list(np.percentile(bs[:, 0], [2.5, 97.5])),
                cos_pc1_ridge=cos(p, w), cos_pc1_ridge_ci=list(np.percentile(bs[:, 1], [2.5, 97.5])),
                cos_pc1_resampled_ci=list(np.percentile(bs[:, 2], [2.5, 97.5])),
                spearman_dm=float(spearmanr(E @ dm, idx)[0]), spearman_ridge=float(spearmanr(E @ w, idx)[0]),
                prop6_var_a=float(at.var()), prop6_lam_max_r=float(lam),
                prop6_ratio=float(at.var() / lam),
                prop6_corr=float(np.linalg.norm(c) / (at.std() * np.sqrt(lam))))


# ------------------------------------------------------------------ Toeplitz test of the readout components (Fig. 12)

def toeplitz_r2(G):
    n = len(G)
    D = np.abs(np.arange(n)[:, None] - np.arange(n)[None])
    off = D > 0
    cnt = np.bincount(D[off], minlength=n)
    g = np.bincount(D[off], weights=G[off], minlength=n) / np.maximum(cnt, 1)
    res = (G - g[D])[off]
    return float(1 - (res ** 2).sum() / ((G[off] - G[off].mean()) ** 2).sum()), g


def toeplitz(E, p, n_null=20):
    Ec = E - E.mean(0)
    rt = Ec - np.outer(Ec @ p, p)
    rng = np.random.default_rng(0)
    summ, arr = {}, {}
    for kind, X in (('raw', rt), ('cos', rt / np.linalg.norm(rt, axis=1, keepdims=True))):
        G = X @ X.T
        for rng_name, hi in (('all', len(E)), ('lt800', 800)):
            Gs = G[:hi, :hi]
            r2, g = toeplitz_r2(Gs)
            null = [toeplitz_r2(Gs[np.ix_(pm, pm)])[0] for pm in (rng.permutation(hi) for _ in range(n_null))]
            summ[f'toep_{kind}_{rng_name}_r2'] = r2
            summ[f'toep_{kind}_{rng_name}_null_max'] = float(np.max(null))
            arr[f'toep_{kind}_{rng_name}_g'] = g
        if kind == 'cos':
            arr['toep_cos_gram'] = G.astype(np.float16)
            arr['toep_cos_blocks'] = np.array([[np.diagonal(G[s:s + 250, s:s + 250], dl).mean()
                                                for dl in range(100)] for s in (0, 250, 500, 750)])
    return summ, arr


# ------------------------------------------------------------------ main

def analyse(run):
    model, cfg, ep = load_model(run)
    log(f'{run}: ep{ep}, dist_attn={cfg.get("dist_attn", False)}, decoder={cfg.get("decoder_type", "linear")}')
    E, U = table(model, 'We'), table(model, 'Wd')
    data = load_data(DATA)
    summ = dict(run=run, epoch=ep)
    arr, s, comps, R, S_ = geometry(E, U)
    summ.update(s)
    log('geometry done')
    s, tot, cor = accuracy(model, data)
    summ.update(s)
    arr.update(acc_tot=tot, acc_cor=cor)
    log(f'accuracy {summ["acc"]:.5f} on {summ["n_seq"]} val sequences')
    Gr = R @ R.T
    s, a2 = steering(model, E, U, comps, data, Gr, arr['a'])
    summ['steer'] = s
    arr.update(a2)
    log('steering done')
    summ['iia'], a2 = iia(model, E, comps[0], data)
    arr.update(a2)
    log('iia done')
    s, a2 = saturation(model, E, arr['a'], Gr, data)
    summ.update(s)
    arr.update(a2)
    log('saturation done')
    s, a2 = vocab_conditions(E, U, arr['a'], arr['b'], R, S_)
    summ.update(s)
    arr.update(a2)
    log('vocabulary conditions done')
    summ.update(pc1_validation(E, comps[0]))
    log('pc1 validation done')
    s, a2 = toeplitz(E, comps[0])
    summ.update(s)
    arr.update(a2)
    log('toeplitz done')
    return summ, arr


def epochs(run):
    data = load_data(DATA)
    st, _ = steer_setup(data)
    ids = torch.tensor(st['ids'], device=DEV)
    pos = torch.full((len(ids),), POS, device=DEV)
    loc = os.path.join(paths.CKPT, run)
    eps = sorted(int(f[2:-3]) for f in os.listdir(loc) if f.endswith('.pt'))
    p_final = fit_pca(table(load_model(run, eps[-1])[0]))[1][0]
    keys = ('var1', 'spearman', 'cos_final', 'acc', 'whole_order', 'whole_identity', 'pc1_order',
            'pc1_identity', 'comp_order', 'comp_identity')
    out = {k: [] for k in keys}
    for ep in eps:
        model = load_model(run, ep)[0]
        E = table(model)
        pca, comps, scores = fit_pca(E)
        out['var1'].append(pca.explained_variance_ratio_[0])
        out['spearman'].append(spearmanr(scores[:, 0], np.arange(len(E)))[0])
        out['cos_final'].append(abs(comps[0] @ p_final))
        out['acc'].append(accuracy(model, data, N_SAT)[0]['acc'])
        src = torch.tensor(E, dtype=torch.float32, device=DEV)[torch.tensor(st['v_new'], device=DEV)]
        u = torch.tensor(comps[0], dtype=torch.float32, device=DEV)
        for mode, key in (('whole', 'whole'), ('dir', 'pc1'), ('comp', 'comp')):
            ev = steer_eval(patched_logits(model, ids, pos, src, u, mode)[0].argmax(-1).cpu().numpy(), st)
            for met in ('order', 'identity'):
                out[f'{key}_{met}'].append(ev[met][0].sum() / ev[met][1].sum())
        log(f'epoch {ep}: acc {out["acc"][-1]:.4f}, PC1 {out["var1"][-1]:.3f}')
    return {f'ep_{k}': np.array(v) for k, v in out.items()} | {'ep_index': np.array(eps)}


def extras():
    res = {}
    E = table(load_model('dist_attention')[0]).astype(np.float32)
    res['distattn_pc123_randomized_3comp'] = [PCA(n_components=3).fit(E).explained_variance_ratio_.tolist()
                                              for _ in range(5)]
    res['distattn_pc123_full'] = PCA(svd_solver='full').fit(E).explained_variance_ratio_[:3].tolist()
    for run, dname in (('dist_attention', DATA), ('dist_readout', 'running_max_50k')):
        res[f'acc_{run}'] = accuracy(load_model(run)[0], load_data(dname))[0]
    tr = load_data(DATA, 'train')['inputs'].numpy()
    seen = {r.tobytes() for r in tr}
    v50 = load_data('running_max_50k')['inputs'].numpy()
    res['frac_50k_val_in_500k_train'] = float(np.mean([r.tobytes() in seen for r in v50]))
    return res


def to_json(o):
    if isinstance(o, dict):
        return {k: to_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_json(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='baseline_seed0')
    ap.add_argument('--epochs', action='store_true')
    ap.add_argument('--extras', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    summ, arr = analyse(args.run)
    if args.epochs:
        arr.update(epochs(args.run))
        log('epochs done')
    if args.extras:
        summ['extras'] = extras()
        log('extras done')
    np.savez_compressed(os.path.join(OUT, f'{args.run}.npz'), **arr)
    with open(os.path.join(OUT, f'{args.run}.json'), 'w') as f:
        json.dump(to_json(summ), f, indent=1)
    log(f'saved {OUT}/{args.run}.npz and .json')
