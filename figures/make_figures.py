"""Figures and tables of the paper, made from the analysis results in outputs/results.

    python -m figures.make_figures

Needs the results of analysis.analyze (all four baseline seeds; --epochs --extras for baseline_seed0),
analysis.pairs and analysis.anchors. Writes PDFs and tables to outputs/figures. CPU only.
"""

import matplotlib
matplotlib.use('Agg')


# %% setup
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from rmg import paths

RES, FIGS = paths.RESULTS, paths.FIGURES
os.makedirs(FIGS, exist_ok=True)
LABEL_FS, TICK_FS, LEGEND_FS, PANEL_FS, ANNOT_FS = 18, 15, 14, 20, 12
plt.rcParams.update({'font.size': TICK_FS, 'axes.linewidth': 1.0, 'pdf.fonttype': 42,
                     'mathtext.fontset': 'dejavusans'})

C_FULL, C_PC1, C_LOW, C_NONE, C_GRAY = '#457B9D', '#F4A261', '#9B5DE5', '#9A9A9A', '#6E6E6E'
COND = {'none': ('no patch', C_NONE), 'whole': ('whole vector', C_FULL),
        'pc1': ('order patch', C_PC1), 'comp': ('readout patch', C_LOW)}
NUM_CM = LinearSegmentedColormap.from_list('ylgn', plt.cm.YlGn(np.linspace(0.40, 1.0, 256)))  # numbers: green, disjoint from the component colours
NUM_NORM = Normalize(1, 1000)
SEQ_CM = LinearSegmentedColormap.from_list('ylgnbu', plt.cm.YlGnBu(np.linspace(0.05, 0.9, 256)))
DIV_CM = LinearSegmentedColormap.from_list('puor', plt.cm.PuOr_r(np.linspace(0.08, 0.92, 256)))
ANCH = [299, 699]   # Fig. 2(c, d) anchors: numbers 300 and 700 (curves from analysis.anchors)
ACOL = [NUM_CM(NUM_NORM(m + 1)) for m in ANCH]
NUM = np.arange(1, 1001)

BASE = 'baseline_seed0'
RUNS = [BASE] + [f'baseline_seed{s}' for s in (1, 2, 3) if os.path.exists(f'{RES}/baseline_seed{s}.json')]
D = {r: dict(np.load(f'{RES}/{r}.npz')) for r in RUNS}
J = {r: json.load(open(f'{RES}/{r}.json')) for r in RUNS}
B, JB = D[BASE], J[BASE]
print('runs:', RUNS)


def style(ax, xlabel=None, ylabel=None):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def letter(ax, s, x=-0.2, y=1.03):
    ax.text(x, y, f'({s})', transform=ax.transAxes, fontsize=PANEL_FS, fontweight='bold',
            ha='left', va='bottom')


def legend(ax, **kw):
    kw.setdefault('fontsize', LEGEND_FS)
    kw.setdefault('frameon', False)
    return ax.legend(**kw)


def save(fig, name):
    path = os.path.join(FIGS, f'{name}.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'saved {path}')


def stack(key, f=lambda x: x):
    return np.stack([f(D[r][key]) for r in RUNS])


def band(ax, x, Y, color, lw=1.6, alpha=0.25, **kw):
    """Mean of Y (runs x points) as a line, ± SD over runs as a band."""
    m = Y.mean(0)
    line, = ax.plot(x, m, color=color, lw=lw, **kw)
    if len(Y) > 1:
        sd = Y.std(0, ddof=1)
        ax.fill_between(x, m - sd, m + sd, color=color, alpha=alpha, linewidth=0)
    return line


def anchor_marks(ax, labels=True):
    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for m, c in zip(ANCH, ACOL):
        ax.axvline(m + 1, color=c, ls='--', lw=1, alpha=0.6)
        if labels:
            ax.text(m + 1, 1.01, f'i={m + 1}', transform=tr, color=c, ha='center', va='bottom',
                    fontsize=ANNOT_FS, fontweight='bold')


def ci(key, run=BASE, block='steer'):
    return J[run][block][key]


def seed_vals(key, block='steer'):
    return [J[r][block][key][0] for r in RUNS if r != BASE]


def md_table(df, path):
    cols = [str(c) for c in df.columns]
    lines = ['| ' + ' | '.join([df.index.name or ''] + cols) + ' |',
             '|' + '---|' * (len(cols) + 1)]
    lines += ['| ' + ' | '.join([str(i)] + [str(v) for v in row]) + ' |' for i, row in df.iterrows()]
    open(path, 'w').write('\n'.join(lines) + '\n')

# %% Fig. 2: learned embeddings and the conditions of Theorem 1
for r in RUNS:   # anchor curves for numbers 300 and 700 (analysis.anchors)
    D[r].update({k: v for k, v in np.load(f'{RES}/{r}_anchors.npz').items() if k != 'anchors'})

fig = plt.figure(figsize=(16.5, 4.2))
gs = fig.add_gridspec(1, 4, width_ratios=[1.45, 1, 1, 1], wspace=0.42)

ax = fig.add_subplot(gs[0, 0], projection='3d')
P, vr = B['pc_scores'], B['var_ratio']
sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=NUM, cmap=NUM_CM, norm=NUM_NORM, s=10, rasterized=True)
ax.view_init(elev=40, azim=-110)
ax.set_box_aspect(None, zoom=0.95)
for setl, i in ((ax.set_xlabel, 0), (ax.set_ylabel, 1), (ax.set_zlabel, 2)):
    setl(f'PC{i + 1} ({vr[i]:.1%})', fontsize=LABEL_FS - 2, labelpad=8)
ax.tick_params(labelsize=TICK_FS - 4)
pos = ax.get_position()
cax = fig.add_axes([pos.x0 + 0.2 * pos.width, pos.y0 - 0.19, 0.6 * pos.width, 0.03])
cb = fig.colorbar(sc, cax=cax, orientation='horizontal')
cb.set_label('number', fontsize=LABEL_FS - 2)
cb.ax.tick_params(labelsize=TICK_FS - 3)
ax.text2D(-0.05, 1.0, '(a)', transform=ax.transAxes, fontsize=PANEL_FS, fontweight='bold', va='bottom')

ax = fig.add_subplot(gs[0, 1])
Y = stack('normE')
m = Y.mean(0)
if len(Y) > 1:
    sd = Y.std(0, ddof=1)
    ax.fill_between(NUM, m - sd, m + sd, color=C_GRAY, alpha=0.3, linewidth=0)
seg = np.stack([np.c_[NUM[:-1], m[:-1]], np.c_[NUM[1:], m[1:]]], 1)
lc = LineCollection(seg, cmap=NUM_CM, norm=NUM_NORM, linewidth=2.2)
lc.set_array(NUM[:-1])
ax.add_collection(lc)
ax.set_yscale('log')
ax.set_xlim(1, 1000)
ax.set_ylim(m.min() * 0.85, m.max() * 1.15)
ax.set_yticks([6, 10, 20, 40], ['6', '10', '20', '40'])
ax.minorticks_off()
style(ax, 'number', r'$\|E_i\|$')
letter(ax, 'b', x=-0.22)

ax = fig.add_subplot(gs[0, 2])
for i, c in enumerate(ACOL):
    band(ax, NUM, stack('lbEE_a', lambda a: a[i]), c, lw=1.0, alpha=0.15, zorder=1, ls=':')
    band(ax, NUM, stack('cosEE_a', lambda a: a[i]), c, lw=1.1, zorder=2)
anchor_marks(ax)
ax.set_ylim(0.875, 1.02)
style(ax, '$j$', r'$\cos(E_i, E_j)$')
letter(ax, 'c', x=-0.22)

ax = fig.add_subplot(gs[0, 3])
for i, c in enumerate(ACOL):
    band(ax, NUM, stack('ubEU_a', lambda a: a[i]), c, lw=1.2, alpha=0.15, zorder=1, ls=':')
    band(ax, NUM, stack('cosEU_a', lambda a: a[i]), c, lw=0.9, zorder=2)
anchor_marks(ax)
style(ax, '$j$', r'$\cos(E_i, U_j)$')
letter(ax, 'd', x=-0.22)
fig.legend(handles=[Line2D([], [], color=C_GRAY, lw=1.8, label='cosine'),
                    Line2D([], [], color=C_GRAY, lw=1.4, ls=':', label='Theorem 1 bound: lower in (c), upper in (d)')],
           loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=LEGEND_FS, frameon=False)
save(fig, 'fig2_geometry')
print({k: round(JB[k], 4) for k in ('frac_eq3_pairs', 'frac_eq4_pairs', 'frac_eq5_pairs')})

# %% Fig. 3: order and readout components
fig, axs = plt.subplots(2, 3, figsize=(10.4, 5.8))
plt.subplots_adjust(wspace=0.95, hspace=0.62)

ax = axs[0, 0]
Y = stack('var_ratio')
k = np.arange(1, Y.shape[1] + 1)
band(ax, k[1:], Y[:, 1:], C_LOW, lw=1.2, marker='o', ms=2.5)
ax.errorbar([1], [Y[:, 0].mean()], yerr=[[Y[:, 0].std(ddof=1)] if len(Y) > 1 else [0]] * 2,
            fmt='o', color=C_PC1, ms=9, zorder=5)
ax.annotate(f'{Y[:, 0].mean():.1%}', (1, Y[:, 0].mean()), xytext=(8, -4), textcoords='offset points',
            color=C_PC1, fontsize=ANNOT_FS + 2, fontweight='bold')
ax.set_yscale('log')
style(ax, 'PC index', 'variance ratio')
axc = ax.twinx()
axc.plot(k, np.cumsum(Y, 1).mean(0), color=C_GRAY, lw=1.5)
axc.set_ylim(0, 1.05)
axc.set_ylabel('cumulative', fontsize=LABEL_FS - 2, color=C_GRAY)
axc.tick_params(labelsize=TICK_FS - 2, colors=C_GRAY)
axc.spines['top'].set_visible(False)
letter(ax, 'a', x=-0.32)

ax = axs[0, 1]
Y = stack('pc_scores', lambda a: a[:, 0])
if len(Y) > 1:
    ax.fill_between(NUM, Y.mean(0) - Y.std(0, ddof=1), Y.mean(0) + Y.std(0, ddof=1), color=C_GRAY,
                    alpha=0.3, linewidth=0)
ax.scatter(NUM, Y.mean(0), c=NUM, cmap=NUM_CM, norm=NUM_NORM, s=6, rasterized=True)
style(ax, 'number', 'PC1 score')
letter(ax, 'b', x=-0.3)

ax = axs[0, 2]
P = B['pc_scores']
sc = ax.scatter(P[:, 2], P[:, 3], c=NUM, cmap=NUM_CM, norm=NUM_NORM, s=8, rasterized=True)
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label('number', fontsize=LABEL_FS - 2)
cb.ax.tick_params(labelsize=TICK_FS - 3)
style(ax, 'PC3', 'PC4')
letter(ax, 'c', x=-0.3)

for ax, pre, lab, l in ((axs[1, 0], 'dotEE', r'$E_i \cdot E_j$', 'd'), (axs[1, 1], 'dotEU', r'$E_i \cdot U_j$', 'e')):
    for part, c, name in (('full', C_FULL, 'full'), ('pc1', C_PC1, 'order (PC1)'), ('low', C_LOW, 'readout (other PCs)')):
        band(ax, NUM, stack(f'{pre}_{part}'), c, lw=1.0 if pre == 'dotEU' else 1.6, label=name)
    ax.axvline(500, color=C_GRAY, ls='--', lw=1, alpha=0.6)
    ax.text(500, 1.01, 'i=500', transform=mtransforms.blended_transform_factory(ax.transData, ax.transAxes),
            color=C_GRAY, ha='center', va='bottom', fontsize=ANNOT_FS)
    style(ax, '$j$', lab)
    letter(ax, l, x=-0.3)

ax = axs[1, 2]
bins = np.linspace(-0.3, 0.9, 61)
for key, c, name in (('cosdiag_full', C_FULL, 'full'), ('cosdiag_pc1', C_PC1, 'order (PC1)'), ('cosdiag_low', C_LOW, 'readout (other PCs)')):
    ax.hist(B[key], bins=bins, color=c, alpha=0.85, density=True, label=name)
style(ax, r'$\cos(E_i, U_i)$', 'density')
letter(ax, 'f', x=-0.3)
h, l = axs[1, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=LEGEND_FS, frameon=False)
save(fig, 'fig3_components')
print('FVE of E_i.E_j by PC1, median over m and runs:', np.median(stack('fve_pc1')).round(4))
for r in RUNS:
    print(r, {k: round(J[r][k], 3) for k in ('var_ratio_pc1', 'spearman_pc1', 'fve_pc1_median',
                                             'cosdiag_full_median', 'cosdiag_pc1_median', 'cosdiag_low_median')})

# %% Fig. 5: patching the order and readout components
fig = plt.figure(figsize=(11, 3.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 1], wspace=0.36)

ax = fig.add_subplot(gs[0])
T = 23
t = np.arange(T)
yt, vo = B['demo_ytrue'][:T], int(B['demo_vorig'])
nat = np.where((t >= 1) & (t < 20) & (yt != vo))[0]
if len(nat):
    ax.axvspan(nat[0] - 0.5, 19.5, color='#EDEDED', zorder=0)
ax.scatter(t, B['demo_input'][:T], color='#BDBDBD', s=18, zorder=1, label='input')
ax.plot(t, yt, color='k', ls='--', lw=1.8, zorder=2, label='running max')
for k, mk, dx in (('none', 'o', -0.15), ('whole', 's', -0.05), ('pc1', '^', 0.05), ('comp', 'v', 0.15)):
    ax.plot(t + dx, B[f'demo_{k}'][:T], marker=mk, ms=6, lw=1.4, color=COND[k][1], zorder=3,
            label=COND[k][0], mfc=COND[k][1] if k != 'none' else 'white')
for x0 in (1, 20):
    ax.axvline(x0, color=C_GRAY, ls=':', lw=1.2)
style(ax, 'position', 'value')
ax.set_ylim(-20, 1050)
letter(ax, 'a', x=-0.14)
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6, fontsize=LEGEND_FS, frameon=False, columnspacing=1.2, handletextpad=0.4)

ax = fig.add_subplot(gs[1])
groups = ['whole', 'pc1', 'comp']
x = np.arange(3)
for gi, g in enumerate(groups):
    for j, (key, alpha) in enumerate(((f'{g}/chg_{g}', 0.95), (f'none/chg_{g}', 0.35))):
        pt, lo, hi = ci(key)
        xp = x[gi] + (j - 0.5) * 0.38
        ax.bar(xp, pt, 0.36, color=COND[g][1], alpha=alpha, edgecolor='white')
        ax.errorbar(xp, pt, yerr=[[pt - lo], [hi - pt]], color='k', lw=1.1, capsize=2.5)
        sv = seed_vals(key)
        if sv:
            ax.scatter(xp + np.linspace(-0.07, 0.07, len(sv)), sv, s=14, color='k', alpha=0.6, zorder=4)
ax.set_xticks(x, ['whole', 'order', 'readout'], fontsize=TICK_FS - 2)
ax.set_ylim(0, 1.05)
style(ax, None, 'accuracy')
letter(ax, 'b', x=-0.25)

ax = fig.add_subplot(gs[2])
conds = ['none', 'whole', 'pc1', 'comp']
w = 0.2
for gi, g in enumerate(('order', 'identity')):
    for ci_, c in enumerate(conds):
        pt, lo, hi = ci(f'{c}/{g}')
        xp = gi + (ci_ - 1.5) * w
        ax.bar(xp, pt, w * 0.92, color=COND[c][1], edgecolor='white')
        ax.errorbar(xp, pt, yerr=[[pt - lo], [hi - pt]], color='k', lw=1.1, capsize=2.5)
        sv = seed_vals(f'{c}/{g}')
        if sv:
            ax.scatter(xp + np.linspace(-0.04, 0.04, len(sv)), sv, s=14, color='k', alpha=0.6, zorder=4)
ax.set_xticks([0, 1], ['order\ntransfer', 'identity\ntransfer'], fontsize=TICK_FS - 1)
ax.set_ylim(0, 1.05)
style(ax, None, 'fraction')
letter(ax, 'c', x=-0.25)
save(fig, 'fig5_patching')
for k in ('whole/chg_whole', 'pc1/chg_pc1', 'comp/chg_comp', 'none/chg_pc1', 'none/chg_comp',
          'whole/order', 'whole/identity', 'pc1/order', 'pc1/identity', 'comp/order', 'comp/identity'):
    print(f'{k:18s}', np.round(ci(k), 3), 'seeds:', np.round(seed_vals(k), 3))
for r in RUNS:
    print(r, 'no patch, stretch A, output v_orig: query 0 ->', round(J[r]['steer'].get('none_A_vorig_query0', np.nan), 3),
          f"(n={J[r]['steer'].get('n_A_query0')}), other queries ->", round(J[r]['steer'].get('none_A_vorig_query_nonzero', np.nan), 3))

# %% Fig. 7: patching controls
fig, axs = plt.subplots(2, 2, figsize=(14, 11.5))
plt.subplots_adjust(wspace=0.38, hspace=0.6)

ax = axs[0, 0]
rows, cols = ['none', 'whole', 'pc1', 'comp'], ['whole', 'pc1', 'comp']
M = np.array([[ci(f'{r}/chg_{c}')[0] for c in cols] for r in rows])
im = ax.imshow(M, cmap=SEQ_CM, vmin=0, vmax=1, aspect='auto')
for i in range(len(rows)):
    for j in range(len(cols)):
        lo, hi = ci(f'{rows[i]}/chg_{cols[j]}')[1:]
        ax.text(j, i, f'{M[i, j]:.2f}\n[{lo:.2f}, {hi:.2f}]', ha='center', va='center', fontsize=ANNOT_FS,
                color='white' if M[i, j] > 0.6 else 'k')
ax.set_xticks(range(3), ['whole-vector\ntarget', 'order-patch\ntarget', 'readout-patch\ntarget'], fontsize=TICK_FS - 2)
ax.set_yticks(range(4), [COND[r][0] for r in rows], fontsize=TICK_FS - 2)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label('accuracy (changed positions)', fontsize=LABEL_FS - 4)
letter(ax, 'a', x=-0.38)

ax = axs[0, 1]
cats = [('vnew', r'$v_{new}$', C_FULL), ('vorig', r'$v_{orig}$', C_PC1), ('ytrue', 'natural max', C_NONE)]
xt, xl = [], []
for si, stretch in enumerate(('A', 'B')):
    for ci_, c in enumerate(['none', 'whole', 'pc1', 'comp']):
        xp = si * 5 + ci_
        bottom = 0
        for cat, lab, col in cats:
            if stretch == 'A' and cat == 'ytrue':
                continue
            v = ci(f'{c}/{stretch}_{cat}')[0]
            ax.bar(xp, v, 0.8, bottom=bottom, color=col, edgecolor='white',
                   label=lab if (si == 1 and ci_ == 0) else None)
            bottom += v
        ax.bar(xp, 1 - bottom, 0.8, bottom=bottom, color='#E6E6E6', edgecolor='white',
               label='other' if (si == 1 and ci_ == 0) else None)
        xt.append(xp)
        xl.append(COND[c][0])
ax.set_xticks(xt, xl, fontsize=TICK_FS - 3, rotation=50, ha='right', rotation_mode='anchor')
for si, name in enumerate(('max still $v_{orig}$', 'natural max $> v_{orig}$')):
    ax.text(si * 5 + 1.5, 1.04, name, ha='center', va='bottom', fontsize=ANNOT_FS + 1)
ax.set_ylim(0, 1.12)
style(ax, None, 'fraction of predictions')
legend(ax, loc='upper center', bbox_to_anchor=(0.5, -0.36), ncol=4, fontsize=LEGEND_FS - 3)
letter(ax, 'b', x=-0.25)

ax = axs[1, 0]
edges = np.array([5, 11, 21, 51, 101, 201, 1000])
gap = B['steer_B_gap']
seq = B['steer_B_seq'] if 'steer_B_seq' in B else None
rng = np.random.default_rng(0)
for key, c, name in (('steer_B_pc1_vorig', C_PC1, r'order patch: output $v_{orig}$'),
                     ('steer_B_hybrid_wins', C_GRAY, 'patched position has top score')):
    y = B[key].astype(float)
    pts, los, his = [], [], []
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        sel = (gap >= lo_) & (gap < hi_)
        pts.append(y[sel].mean())
        if seq is not None:   # 95% CI, resampling sequences (positions in a sequence are correlated)
            u = np.unique(seq[sel])
            cnt = np.bincount(seq[sel], minlength=seq.max() + 1)
            num = np.bincount(seq[sel], weights=y[sel], minlength=seq.max() + 1)
            picks = rng.choice(u, (1000, len(u)))
            bs = num[picks].sum(1) / cnt[picks].sum(1)
            los.append(np.percentile(bs, 2.5)); his.append(np.percentile(bs, 97.5))
    xc = np.sqrt(edges[:-1] * (edges[1:] - 1))
    ax.plot(xc, pts, marker='o', color=c, lw=1.8, label=name)
    if los:
        ax.fill_between(xc, los, his, color=c, alpha=0.25, linewidth=0)
ax.set_xscale('log')
ax.set_ylim(0, 1.02)
style(ax, r'gap $v_{new}$ − natural max', 'fraction of positions')
legend(ax, loc='upper left', fontsize=LEGEND_FS - 3)
letter(ax, 'c', x=-0.25)

ax = axs[1, 1]
ctrl = B['ctrl']
ax.scatter(ctrl[2:, 0, 0], ctrl[2:, 0, 1], s=22, color=C_NONE, alpha=0.6, marker='o', label='random direction only')
ax.scatter(ctrl[2:, 1, 0], ctrl[2:, 1, 1], s=26, color=C_NONE, alpha=0.6, marker='x', label='all but random direction')
ax.scatter(*ctrl[1, 0, :2], s=90, color=C_FULL, marker='o', edgecolor='k', label='PC2 only')
ax.scatter(*ctrl[1, 1, :2], s=90, color=C_FULL, marker='X', edgecolor='k', label='all but PC2')
ax.scatter(*ctrl[0, 0, :2], s=110, color=C_PC1, marker='o', edgecolor='k', label='PC1 only (order patch)')
ax.scatter(*ctrl[0, 1, :2], s=110, color=C_LOW, marker='X', edgecolor='k', label='all but PC1 (readout patch)')
ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.04)
style(ax, 'order taken from source', 'identity taken from source')
legend(ax, loc='center right', fontsize=LEGEND_FS - 4)
letter(ax, 'd', x=-0.25)
save(fig, 'app_steering_controls')
print('random dir only, order: max', ctrl[2:, 0, 0].max().round(3), ' all-but-random, identity: min', ctrl[2:, 1, 1].min().round(3))
print('PC2 only', ctrl[1, 0, :2].round(3), ' all but PC2', ctrl[1, 1, :2].round(3))
print('mean |patch| along direction: PC1', B['ctrl_mag'][0].round(2), ' PC2', B['ctrl_mag'][1].round(2),
      ' random', B['ctrl_mag'][2:].mean().round(2))

# %% Fig. 8: interchange with a decoy
fig, ax = plt.subplots(figsize=(8.5, 4.6))
outs = [('v', 'source $v$', C_FULL), ('vc', 'decoy $v_c$', C_PC1), ('vprev', r'previous max $v_{prev}$', C_LOW)]
conds = ['none', 'whole', 'pc1', 'comp']
w = 0.26
for oi, (o, lab, col) in enumerate(outs):
    for ci_, c in enumerate(conds):
        pt, lo, hi = J[BASE]['iia'][f'{c}/top_{o}']
        xp = ci_ + (oi - 1) * w
        ax.bar(xp, pt, w * 0.92, color=col, edgecolor='white', label=lab if ci_ == 0 else None)
        ax.errorbar(xp, pt, yerr=[[pt - lo], [hi - pt]], color='k', lw=1.0, capsize=2)
        sv = [J[r]['iia'][f'{c}/top_{o}'][0] for r in RUNS if r != BASE]
        if sv:
            ax.scatter(xp + np.linspace(-0.05, 0.05, len(sv)), sv, s=12, color='k', alpha=0.6, zorder=4)
ax.set_xticks(range(4), [COND[c][0] for c in conds], fontsize=TICK_FS - 1)
ax.set_ylim(0, 1.05)
style(ax, None, 'fraction of top-1 outputs')
legend(ax, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=LEGEND_FS - 2)
save(fig, 'app_iia_outcomes')
print({k: v for k, v in J[BASE]['iia'].items() if 'recovery' in k or 'n_' in k})

# %% Fig. 9: attention saturation
edges = np.array([1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 1000])
xc = np.sqrt(edges[:-1] * (edges[1:] - 1).clip(1))
terc = [(0, 333, 'max < 334'), (333, 666, '334 ≤ max < 667'), (666, 1000, 'max ≥ 667')]
fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
plt.subplots_adjust(wspace=0.32)

ax = axs[0]
for lo_, hi_, lab in terc:
    col = NUM_CM(NUM_NORM((lo_ + hi_) / 2))
    meds = []
    for r in RUNS:
        g, q, mass = D[r]['sat_gap'], D[r]['sat_q'], D[r]['sat_mass']
        meds.append([np.median(mass[(g >= a) & (g < b) & (q >= lo_) & (q < hi_)])
                     if ((g >= a) & (g < b) & (q >= lo_) & (q < hi_)).sum() > 20 else np.nan
                     for a, b in zip(edges[:-1], edges[1:])])
    band(ax, xc, np.array(meds), col, lw=2, marker='o', ms=4, label=lab)
ax.set_xscale('log')
ax.set_ylim(0, 1.02)
style(ax, 'gap: max − next-largest value', 'attention on max (median)')
legend(ax, loc='lower right', fontsize=LEGEND_FS - 2)
letter(ax, 'a', x=-0.2)

ax = axs[1]
for key, col, lab in (('sat_eq1', C_FULL, 'Eq. 1 holds'), ('sat_eq8', C_PC1, 'Eq. 8 holds'),
                      ('sat_correct', C_GRAY, 'correct output')):
    Y = []
    for r in RUNS:
        g, y = D[r]['sat_gap'], D[r][key].astype(float)
        Y.append([y[(g >= a) & (g < b)].mean() for a, b in zip(edges[:-1], edges[1:])])
    band(ax, xc, np.array(Y), col, lw=2, marker='o', ms=4, label=lab)
ax.set_xscale('log')
ax.set_ylim(0, 1.02)
style(ax, 'gap: max − next-largest value', 'fraction of positions')
legend(ax, loc='lower right', fontsize=LEGEND_FS - 2)
letter(ax, 'b', x=-0.2)
save(fig, 'app_saturation')
for r in RUNS:
    print(r, {k: round(J[r][k], 4) for k in ('sat_mass_median', 'sat_frac_mass_gt_0p9', 'sat_frac_eq1',
                                             'sat_frac_eq8', 'sat_acc_eq1_true', 'sat_acc_eq1_false')})

# %% Fig. 10: interference budget over all vocabulary triples
gedges = np.array([1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513, 1000])
gx = np.sqrt(gedges[:-1] * (gedges[1:] - 1).clip(1))
def gbin(A):   # (10, V) counts by gap -> (10, n_bins)
    return np.stack([A[:, a:b].sum(1) for a, b in zip(gedges[:-1], gedges[1:])], 1)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(wspace=0.35, hspace=0.42)

ax = axs[0, 0]
F = gbin(B['tri_ok8']) / gbin(B['tri_tot'])
im = ax.imshow(F, cmap=SEQ_CM, vmin=0, vmax=1, aspect='auto', origin='lower')
ax.set_xticks(range(len(gx)), [f'{a}' for a in gedges[:-1]], fontsize=TICK_FS - 3, rotation=45)
ax.set_yticks(range(10), [f'{100 * i + 1}' for i in range(10)], fontsize=TICK_FS - 3)
style(ax, r'gap $m-k$ (bin start)', 'max $m$ (bin start)')
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label('fraction with Eq. 8', fontsize=LABEL_FS - 4)
letter(ax, 'a', x=-0.3)

ax = axs[0, 1]
for key, col, lab in (('tri_ok1', C_FULL, 'Eq. 1 (selection)'), ('tri_ok8', C_PC1, 'Eq. 8 (order term dominates)')):
    Y = np.array([gbin(D[r][key]).sum(0) / gbin(D[r]['tri_tot']).sum(0) for r in RUNS])
    band(ax, gx, Y, col, lw=2, marker='o', ms=4, label=lab)
ax.set_xscale('log')
ax.set_ylim(0, 1.02)
style(ax, r'gap $m-k$', 'fraction of triples')
legend(ax, loc='lower right', fontsize=LEGEND_FS - 2)
letter(ax, 'b', x=-0.25)

ax = axs[1, 0]
for key, col, lab in (('neg', C_PC1, r'order reversed: $a_m \leq a_k$'),
                      ('big', C_LOW, 'interference exceeds order term')):
    Y = []
    for r in RUNS:
        s_ = D[r]['tri_samp']
        g, C, I = s_[:, 1], s_[:, 2], s_[:, 3]
        bad = (C <= 0) if key == 'neg' else ((C > 0) & (np.abs(I) >= C))
        Y.append([bad[(g >= a) & (g < b)].mean() for a, b in zip(gedges[:-1], gedges[1:])])
    band(ax, gx, np.array(Y), col, lw=2, marker='o', ms=4, label=lab)
ax.set_xscale('log')
ax.set_ylim(0, 0.6)
style(ax, r'gap $m-k$', 'fraction of triples')
legend(ax, loc='upper right', fontsize=LEGEND_FS - 3)
letter(ax, 'c', x=-0.25)

ax = axs[1, 1]
band(ax, NUM, stack('id_frac2'), C_FULL, lw=1.4, label='Eq. 2 (readout)')
band(ax, NUM, stack('id_frac53'), C_LOW, lw=1.4, label='readout component decides (Thm. 3)')
ax.set_ylim(0.94, 1.003)
style(ax, 'number $m$', r'fraction of $j \neq m$')
legend(ax, loc='lower left', fontsize=LEGEND_FS - 2)
letter(ax, 'd', x=-0.25)
save(fig, 'app_interference')
for r in RUNS:
    print(r, {k: round(J[r][k], 4) for k in ('tri_frac_eq1', 'tri_frac_eq8', 'tri_frac_eq1_gap1',
                                             'tri_frac_eq8_gap1', 'tri_frac_eq56', 'id_frac_eq2',
                                             'id_frac_eq53', 'id_frac_q_all_eq53', 'b_cv')})

# %% Fig. 6: Theorem 1 conditions per anchor
fig, ax = plt.subplots(figsize=(8, 4.6))
for key, col, lab in (('frac_eq3', C_GRAY, r'Eq. 3: $\|E_j\| > \|E_i\|$, $j>i$'),
                      ('frac_eq4', C_FULL, 'Eq. 4: cosine above lower bound'),
                      ('frac_eq5', C_LOW, 'Eq. 5: cosine below upper bound')):
    band(ax, NUM, stack(key), col, lw=1.4, label=lab)
ax.set_ylim(0.8, 1.005)
style(ax, 'anchor $i$', 'fraction of $j$ satisfied')
legend(ax, loc='lower right', fontsize=LEGEND_FS - 3)
save(fig, 'app_theorem1')
for r in RUNS:
    print(r, {k: round(J[r][k], 4) for k in ('frac_eq3_pairs', 'frac_eq4_pairs', 'frac_eq5_pairs')})

# %% Fig. 11 and Table 2: two-token inputs
PR = [r for r in RUNS if os.path.exists(f'{RES}/{r}_pairs.npz')]
P2 = {r: dict(np.load(f'{RES}/{r}_pairs.npz')) for r in PR}
JP = {r: json.load(open(f'{RES}/{r}_pairs.json')) for r in PR}
qv, mv = P2[BASE]['q'].astype(int), P2[BASE]['m'].astype(int)
gp = qv - mv
pedges = np.array([1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513, 1000])
px = np.sqrt(pedges[:-1] * (pedges[1:] - 1).clip(1))
gsel = [(gp >= a) & (gp < b) for a, b in zip(pedges[:-1], pedges[1:])]
sig = lambda x: 1 / (1 + np.exp(-np.asarray(x, dtype=float)))   # noqa: E731

def pair_vals(r, order):
    Dl, C, I = P2[r][f'{order}_delta'], P2[r][f'{order}_C'], P2[r][f'{order}_I']
    return {'eq8': C > np.abs(I), 'eq1': Dl > 0, 'pmax': sig(Dl), 'acc': P2[r][f'{order}_correct']}

fig, axs = plt.subplots(2, 2, figsize=(14, 11))
plt.subplots_adjust(wspace=0.38, hspace=0.62)
CUR = (('eq8', C_PC1, 'Eq. 8 holds', '-'), ('eq1', C_FULL, 'Eq. 1 holds', '--'),
       ('pmax', C_GRAY, 'attention on max (mean)', ':'), ('acc', 'k', 'correct output', '-'))
for ax, order, l, tag, tx, ty, ha, va in ((axs[0, 0], 'lf', 'a', 'input $(m, k)$: larger first', 0.03, 0.97, 'left', 'top'),
                                          (axs[0, 1], 'ls', 'b', 'input $(k, m)$: larger second', 0.97, 0.05, 'right', 'bottom')):
    for k, col, lab, ls in CUR:
        Y = np.array([[pair_vals(r, order)[k][s].mean() for s in gsel] for r in PR])
        band(ax, px, Y, col, lw=2, ls=ls, marker='o', ms=4, label=lab)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.02)
    ax.text(tx, ty, tag, transform=ax.transAxes, ha=ha, va=va, fontsize=ANNOT_FS + 2)
    style(ax, 'gap $m-k$', 'fraction of pairs')
    letter(ax, l, x=-0.22)
h, l_ = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l_, loc='center', bbox_to_anchor=(0.5, 0.505), ncol=4, fontsize=LEGEND_FS - 2, frameon=False)

ax = axs[1, 0]
dedges = np.linspace(-1.2, 2.5, 38)
dc = 0.5 * (dedges[:-1] + dedges[1:])
for lo_, hi_, lab in ((0, 333, 'max < 334'), (333, 666, '334 ≤ max < 667'), (666, 1000, 'max ≥ 667')):
    Y = []
    for r in PR:
        Dl, ok = P2[r]['lf_delta'], P2[r]['lf_correct']
        sel_q = (qv >= lo_) & (qv < hi_)
        Y.append([ok[sel_q & (Dl >= a) & (Dl < b)].mean() if (sel_q & (Dl >= a) & (Dl < b)).sum() >= 30 else np.nan
                  for a, b in zip(dedges[:-1], dedges[1:])])
    band(ax, dc, np.array(Y), NUM_CM(NUM_NORM((lo_ + hi_) / 2)), lw=2, label=lab)
ax.plot(dc, sig(dc), color=C_GRAY, ls=':', lw=1.8, label=r'attention on max, $\sigma(\Delta)$')
ax.axvline(0, color=C_GRAY, lw=0.8)
ax.set_ylim(0, 1.02)
style(ax, r'margin $\Delta = (E_k\cdot E_m - E_k\cdot E_k)/\sqrt{d}$', 'fraction correct')
legend(ax, loc='lower right', fontsize=LEGEND_FS - 3)
ax.text(0.97, 0.52, 'input $(m, k)$', transform=ax.transAxes, ha='right', va='center', fontsize=ANNOT_FS + 2)
letter(ax, 'c', x=-0.22)

ax = axs[1, 1]
okb = P2[BASE]['lf_correct']
qdec = np.minimum(qv // 100, 9)
H = np.full((10, len(gsel)), np.nan)
for i in range(10):
    for j, s in enumerate(gsel):
        sel = s & (qdec == i)
        if sel.any():
            H[i, j] = okb[sel].mean()
im = ax.imshow(H, cmap=SEQ_CM, vmin=0, vmax=1, aspect='auto', origin='lower')
ax.set_xticks(range(len(px)), [f'{a}' for a in pedges[:-1]], fontsize=TICK_FS - 3, rotation=45)
ax.set_yticks(range(10), [f'{100 * i + 1}' for i in range(10)], fontsize=TICK_FS - 3)
style(ax, 'gap $m-k$ (bin start)', 'max $m$ (bin start)')
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label('fraction correct, input $(m, k)$', fontsize=LABEL_FS - 4)
letter(ax, 'd', x=-0.3)
save(fig, 'app_pairs')

rows = [('accuracy', 'acc'), ('accuracy, gap 1', 'acc_gap1'), ('accuracy, gap ≤ 5', 'acc_gap_le5'),
        ('accuracy, gap ≥ 20', 'acc_gap_ge20'), ('accuracy, gap 1, max < 334', 'acc_gap1_max_low'),
        ('accuracy, gap 1, max ≥ 667', 'acc_gap1_max_high'), ('Eq. 1 holds', 'frac_eq1'), ('Eq. 8 holds', 'frac_eq8'),
        ('attention on max (mean)', 'mean_pmax'), ('accuracy when Eq. 1 holds', 'acc_given_eq1'),
        ('accuracy when Eq. 1 fails', 'acc_given_not_eq1'), ('errors that output the smaller number', 'frac_errors_output_m'),
        ('AUC: margin Δ', 'auc_delta'), ('AUC: order term of Δ', 'auc_pc1_term'), ('AUC: gap', 'auc_gap'),
        ('AUC: Eq. 8 (yes/no)', 'auc_eq8')]
def msd(v):
    v = [x for x in v if x is not None]
    if not v:
        return 'n/a'
    return f'{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}' if len(v) > 1 else f'{v[0]:.4f}'
tab = pd.DataFrame({name: {lab: msd([JP[r][o][k] for r in PR]) for lab, k in rows}
                    for o, name in (('lf', 'larger first (m, k)'), ('ls', 'larger second (k, m)'))})
tab.index.name = 'quantity'
tab.to_csv(f'{FIGS}/tab_pairs.csv'); md_table(tab, f'{FIGS}/tab_pairs.md')
print(tab.to_string())
for o, lab in (('lf', 'larger first'), ('ls', 'larger second')):
    print(f'{lab}: fraction of pairs with residual term I < 0:', [round(float((P2[r][f"{o}_I"] < 0).mean()), 4) for r in PR])

# %% Fig. 4: two-token inputs, larger number first
# Main-text Fig. 4: two-token inputs, larger number first (same data as app_pairs, panels a and c)
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
plt.subplots_adjust(wspace=0.3)
ax = axs[0]
for k, col, lab, ls in (('acc', 'k', 'correct output', '-'), ('eq8', C_PC1, 'Eq. 8 holds', '-'),
                        ('eq1', C_FULL, 'Eq. 1 holds', '--'), ('pmax', C_GRAY, 'attention on max', ':')):
    Y = np.array([[pair_vals(r, 'lf')[k][s].mean() for s in gsel] for r in PR])
    band(ax, px, Y, col, lw=2, ls=ls, marker='o', ms=4, label=lab)
ax.set_xscale('log')
ax.set_ylim(0, 1.02)
ax.text(0.03, 0.97, 'larger first', transform=ax.transAxes, ha='left', va='top', fontsize=ANNOT_FS + 2)
style(ax, 'gap $m-k$', 'fraction of pairs')
legend(ax, loc='lower right', fontsize=LEGEND_FS - 2)
letter(ax, 'a', x=-0.2)
ax = axs[1]
for lo_, hi_, lab in ((0, 333, 'max < 334'), (333, 666, '334 ≤ max < 667'), (666, 1000, 'max ≥ 667')):
    Y = []
    for r in PR:
        Dl, ok = P2[r]['lf_delta'], P2[r]['lf_correct']
        sel_q = (qv >= lo_) & (qv < hi_)
        Y.append([ok[sel_q & (Dl >= a) & (Dl < b)].mean() if (sel_q & (Dl >= a) & (Dl < b)).sum() >= 30 else np.nan
                  for a, b in zip(dedges[:-1], dedges[1:])])
    band(ax, dc, np.array(Y), NUM_CM(NUM_NORM((lo_ + hi_) / 2)), lw=2, label=lab)
ax.plot(dc, sig(dc), color=C_GRAY, ls=':', lw=1.8, label=r'attention on max, $\sigma(\Delta)$')
ax.axvline(0, color=C_GRAY, lw=0.8)
ax.set_ylim(0, 1.02)
style(ax, r'margin $\Delta$', 'fraction correct')
legend(ax, loc='lower right', fontsize=LEGEND_FS - 2)
letter(ax, 'b', x=-0.2)
save(fig, 'fig4_two_token')


# %% Fig. 12: translation symmetry of the readout components
fig = plt.figure(figsize=(18, 4.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1], wspace=0.55)

ax = fig.add_subplot(gs[0])
im = ax.imshow(B['toep_cos_gram'].astype(float), cmap=DIV_CM, vmin=-1, vmax=1, extent=(0.5, 1000.5, 1000.5, 0.5),
               interpolation='nearest', rasterized=True)
style(ax, 'number $j$', 'number $i$')
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label(r'$\cos(\tilde r_i, \tilde r_j)$', fontsize=LABEL_FS - 3)
letter(ax, 'a', x=-0.3)

ax = fig.add_subplot(gs[1])
for bi_, s0 in enumerate((0, 250, 500, 750)):
    band(ax, np.arange(100), stack('toep_cos_blocks', lambda a: a[bi_]), NUM_CM(NUM_NORM(s0 + 125)), lw=1.8,
         label=f'{s0 + 1}–{s0 + 250}')
ax.axhline(0, color=C_GRAY, lw=0.8)
style(ax, r'offset $|i-j|$', 'mean cosine')
legend(ax, loc='upper right', fontsize=LEGEND_FS - 3, title='numbers', title_fontsize=LEGEND_FS - 3)
letter(ax, 'b', x=-0.25)

ax = fig.add_subplot(gs[2])
labs = [('raw', 'all', 'Gram,\nall'), ('raw', 'lt800', 'Gram,\n< 800'), ('cos', 'all', 'cosine,\nall'),
        ('cos', 'lt800', 'cosine,\n< 800')]
for i, (k, rg, lab) in enumerate(labs):
    vals = [J[r][f'toep_{k}_{rg}_r2'] for r in RUNS]
    ax.bar(i, np.mean(vals), 0.6, color=C_LOW, alpha=0.9)
    if len(vals) > 1:
        ax.scatter(i + np.linspace(-0.1, 0.1, len(vals)), vals, s=14, color='k', alpha=0.6, zorder=4)
    ax.scatter(i, max(J[r][f'toep_{k}_{rg}_null_max'] for r in RUNS), marker='_', s=300, color='k', zorder=5)
ax.set_xticks(range(4), [l[2] for l in labs], fontsize=TICK_FS - 3)
ax.set_ylim(0, 1)
style(ax, None, r'Toeplitz fit $R^2$')
legend(ax, handles=[Line2D([], [], color='k', marker='_', ms=14, lw=0, label='shuffled numbers (max)')],
       loc='upper left', fontsize=LEGEND_FS - 3)
letter(ax, 'c', x=-0.25)
save(fig, 'app_toeplitz')
for r in RUNS:
    print(r, {k: round(v, 4) for k, v in J[r].items() if k.startswith('toep_')})

# %% Fig. 13: training epochs
fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
plt.subplots_adjust(wspace=0.3)
ep = B['ep_index'] + 1
ax = axs[0]
for key, col, lab in (('ep_acc', C_GRAY, 'accuracy'), ('ep_var1', C_PC1, 'PC1 variance ratio'),
                      ('ep_cos_final', C_FULL, r'$|\cos(\mathrm{PC1}, \mathrm{PC1}_{final})|$')):
    ax.plot(ep, B[key], marker='o', ms=4, lw=1.8, color=col, label=lab)
ax.set_ylim(0.45, 1.02)
style(ax, 'epoch', 'value')
legend(ax, loc='lower right', fontsize=LEGEND_FS - 2)
letter(ax, 'a', x=-0.18)
ax = axs[1]
for key, c, ls, lab in (('whole_order', 'whole', '-', 'whole: order'), ('pc1_order', 'pc1', '-', 'order patch: order'),
                        ('comp_identity', 'comp', '-', 'readout patch: identity'),
                        ('pc1_identity', 'pc1', ':', 'order patch: identity'), ('comp_order', 'comp', ':', 'readout patch: order')):
    ax.plot(ep, B[f'ep_{key}'], marker='o', ms=4, lw=1.8, ls=ls, color=COND[c][1], label=lab)
ax.set_ylim(-0.02, 1.02)
style(ax, 'epoch', 'fraction of positions')
legend(ax, loc='center right', fontsize=LEGEND_FS - 3)
letter(ax, 'b', x=-0.18)
save(fig, 'app_epochs')

# %% Tables: patching, decoy, seeds (Table 1) and checks quoted in the text
fmt = lambda v: f'{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]'
rows = {}
for c in ('none', 'whole', 'pc1', 'comp'):
    rows[COND[c][0]] = {**{f'acc, {t} target (window)': fmt(ci(f'{c}/win_{t}')) for t in ('whole', 'pc1', 'comp')},
                        **{f'acc, {t} target (changed)': fmt(ci(f'{c}/chg_{t}')) for t in ('whole', 'pc1', 'comp')},
                        'order from source': fmt(ci(f'{c}/order')), 'identity from source': fmt(ci(f'{c}/identity'))}
tab = pd.DataFrame(rows).T
tab.index.name = 'patch'
tab.to_csv(f'{FIGS}/tab_steering.csv'); md_table(tab, f'{FIGS}/tab_steering.md')
print(tab.to_string())

rows = {}
for c in ('none', 'whole', 'pc1', 'comp'):
    rows[COND[c][0]] = {f'top-1 = {o}': fmt(J[BASE]['iia'][f'{c}/top_{o}']) for o in ('v', 'vc', 'vprev')}
    rows[COND[c][0]]['recovery (median)'] = round(J[BASE]['iia'][f'{c}/recovery_median'], 3)
tab = pd.DataFrame(rows).T
tab.index.name = 'patch'
tab.to_csv(f'{FIGS}/tab_iia.csv'); md_table(tab, f'{FIGS}/tab_iia.md')
print(tab.to_string())

keys = [('accuracy (150k val)', lambda j: j['acc']), ('PC1 variance', lambda j: j['var_ratio_pc1']),
        ('Spearman(PC1, number)', lambda j: j['spearman_pc1']), ('FVE of E_i.E_j by PC1 (median)', lambda j: j['fve_pc1_median']),
        ('cos(PC1, mean-difference dir.)', lambda j: j['cos_pc1_dm']),
        ('order patch: order from source', lambda j: j['steer']['pc1/order'][0]),
        ('order patch: identity from source', lambda j: j['steer']['pc1/identity'][0]),
        ('readout patch: order from source', lambda j: j['steer']['comp/order'][0]),
        ('readout patch: identity from source', lambda j: j['steer']['comp/identity'][0]),
        ('attention on max (median)', lambda j: j['sat_mass_median']), ('positions with Eq. 1', lambda j: j['sat_frac_eq1']),
        ('triples with Eq. 8', lambda j: j['tri_frac_eq8']), ('triples with Eq. 8, gap 1', lambda j: j['tri_frac_eq8_gap1']),
        ('pairs where the readout component decides (Thm. 3)', lambda j: j['id_frac_eq53']), ('Toeplitz R2 (cosine, < 800)', lambda j: j['toep_cos_lt800_r2'])]
tab = pd.DataFrame({r.replace('baseline_', ''): {k: round(f(J[r]), 4) for k, f in keys} for r in RUNS})
if len(RUNS) > 1:
    vals = np.array([[f(J[r]) for r in RUNS] for _, f in keys])
    tab['mean ± SD'] = [f'{m:.4f} ± {s:.4f}' for m, s in zip(vals.mean(1), vals.std(1, ddof=1))]
tab.index.name = 'quantity'
tab.to_csv(f'{FIGS}/tab_seeds.csv'); md_table(tab, f'{FIGS}/tab_seeds.md')
print(tab.to_string())

ex = JB['extras']
checks = {
    'accuracy, baseline (150k val)': f"{JB['acc']:.5f}",
    'accuracy, distance attention (150k val)': f"{ex['acc_dist_attention']['acc']:.5f}",
    'accuracy, distance readout (15k val)': f"{ex['acc_dist_readout']['acc']:.5f}",
    'dist. attention PC1-3, full solver': ', '.join(f'{v:.4f}' for v in ex.get('distattn_pc123_full', [ex.get('distattn_pc1_full', np.nan)])),
    'Theorem 1 pairs: Eq. 3 / 4 / 5': f"{JB['frac_eq3_pairs']:.4f} / {JB['frac_eq4_pairs']:.4f} / {JB['frac_eq5_pairs']:.4f}",
    'triples within the 2R^2 bound': f"{JB['tri_frac_eq56']:.4f}",
    'PCA recovery: Var(a) / lambda_max(Cov r) (mean-diff. v)': f"{JB['prop6_ratio']:.2f}",
    'PCA recovery: normalised a-r correlation': f"{JB['prop6_corr']:.3f}",
    'ridge order direction: CV R2 / cos with PC1': f"{JB['ridge_cv_r2']:.3f} / {JB['cos_pc1_ridge']:.3f}",
    'b_j coefficient of variation': f"{JB['b_cv']:.3f}",
    '50k-dataset val rows inside 500k train split': f"{ex['frac_50k_val_in_500k_train']:.3f}",
}
tab = pd.DataFrame({'value': checks})
tab.index.name = 'check'
tab.to_csv(f'{FIGS}/tab_checks.csv'); md_table(tab, f'{FIGS}/tab_checks.md')
print(tab.to_string())
