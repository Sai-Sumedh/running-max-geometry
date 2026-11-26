"""Figures of the architectural variants (Figs. 14 and 15), made from the final weight tables.

    python -m figures.make_ablation_figures

Needs the final checkpoints of baseline_seed0, dist_readout and dist_attention. Reads only the embedding and
unembedding tables (no forward pass); CPU only. Writes PDFs to outputs/figures.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from rmg import paths  # noqa: E402

FIGDIR = paths.FIGURES
os.makedirs(FIGDIR, exist_ok=True)

MODEL_LABELS = [
    ('baseline_seed0', 'baseline'),
    ('dist_readout', 'distance readout'),
    ('dist_attention', 'distance attention'),
]


def load_final_weights(run):
    """Embedding and unembedding tables (E, U) of the last checkpoint of a run."""
    loc = os.path.join(paths.CKPT, run)
    last = max(int(f[2:-3]) for f in os.listdir(loc) if f.endswith('.pt'))
    sd = torch.load(os.path.join(loc, f'ep{last}.pt'), map_location='cpu', weights_only=False)['model_state_dict']
    return sd['We.weight'], sd['Wd.weight']


weights = {run: load_final_weights(run) for run, _ in MODEL_LABELS}


# %% Fig. 14: PCA of E and U for the three models
import string

AXIS_LABEL_SIZE = 16
PANEL_LABEL_SIZE = 20
TICK_SIZE = 11

# number colour scale shared with the main figures (NUM_CM in figures/make_figures.py)
cmap = LinearSegmentedColormap.from_list('ylgn', plt.cm.YlGn(np.linspace(0.40, 1.0, 256)))
vocab_size = weights[MODEL_LABELS[0][0]][0].shape[0]
color_norm = plt.Normalize(vmin=1, vmax=vocab_size)
color_vals = np.arange(1, vocab_size + 1)

panel_letters = iter(string.ascii_lowercase)
n_rows = len(MODEL_LABELS)

fig = plt.figure(figsize=(15, 4.3 * n_rows))
gs = fig.add_gridspec(n_rows, 3, width_ratios=[1, 1, 0.85],
                       left=0.06, right=0.93, top=0.93, bottom=0.07,
                       wspace=0.02, hspace=0.3)

ax2_ref = None
for row, (exptname, label) in enumerate(MODEL_LABELS):
    We, Wd = weights[exptname]

    pca_We3 = PCA(n_components=3, svd_solver='full').fit(We)
    We_3d = pca_We3.transform(We)
    pca_Wd3 = PCA(n_components=3, svd_solver='full').fit(Wd)
    Wd_3d = pca_Wd3.transform(Wd)

    cumvar_We = np.cumsum(PCA().fit(We).explained_variance_ratio_)
    cumvar_Wd = np.cumsum(PCA().fit(Wd).explained_variance_ratio_)

    # embed (We) PCA
    ax0 = fig.add_subplot(gs[row, 0], projection='3d')
    ax0.view_init(elev=40, azim=-110)
    ax0.set_box_aspect(None, zoom=1.0)
    sc0 = ax0.scatter(We_3d[:, 0], We_3d[:, 1], We_3d[:, 2], c=color_vals, cmap=cmap, norm=color_norm, s=12)
    var0 = pca_We3.explained_variance_ratio_
    ax0.set_xlabel(f'PC1 ({var0[0]:.1%})', fontsize=AXIS_LABEL_SIZE, labelpad=8)
    ax0.set_ylabel(f'PC2 ({var0[1]:.1%})', fontsize=AXIS_LABEL_SIZE, labelpad=8)
    ax0.set_zlabel(f'PC3 ({var0[2]:.1%})', fontsize=AXIS_LABEL_SIZE, labelpad=8)
    ax0.tick_params(labelsize=TICK_SIZE)
    ax0.text2D(0.02, 0.98, f'({next(panel_letters)})', transform=ax0.transAxes,
               fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')

    # unembed (Wd) PCA
    ax1 = fig.add_subplot(gs[row, 1], projection='3d')
    ax1.view_init(elev=40, azim=-110)
    ax1.set_box_aspect(None, zoom=1.0)
    sc1 = ax1.scatter(Wd_3d[:, 0], Wd_3d[:, 1], Wd_3d[:, 2], c=color_vals, cmap=cmap, norm=color_norm, s=12)
    var1 = pca_Wd3.explained_variance_ratio_
    ax1.set_xlabel(f'PC1 ({var1[0]:.1%})', fontsize=AXIS_LABEL_SIZE, labelpad=8)
    ax1.set_ylabel(f'PC2 ({var1[1]:.1%})', fontsize=AXIS_LABEL_SIZE, labelpad=8)
    ax1.set_zlabel(f'PC3 ({var1[2]:.1%})', fontsize=AXIS_LABEL_SIZE, labelpad=8)
    ax1.tick_params(labelsize=TICK_SIZE)
    ax1.text2D(0.02, 0.98, f'({next(panel_letters)})', transform=ax1.transAxes,
               fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')

    # cumulative variance (embed vs unembed, same panel); x-axis shared across rows
    ax2 = fig.add_subplot(gs[row, 2], sharex=ax2_ref)
    ax2_ref = ax2_ref or ax2
    ax2.plot(np.arange(1, len(cumvar_We) + 1), cumvar_We, label='embedding $E$', color='#457B9D', lw=1.8)
    ax2.plot(np.arange(1, len(cumvar_Wd) + 1), cumvar_Wd, label='unembedding $U$', color='#6E6E6E', lw=1.8)
    ax2.set_ylabel('cumulative variance', fontsize=AXIS_LABEL_SIZE)
    ax2.tick_params(labelsize=TICK_SIZE)
    ax2.legend(fontsize=11, frameon=False, loc='center right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.text(-0.18, 0.98, f'({next(panel_letters)})', transform=ax2.transAxes,
              fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='right')
    if row == n_rows - 1:
        ax2.set_xlabel('number of PCs', fontsize=AXIS_LABEL_SIZE)
    else:
        ax2.tick_params(labelbottom=False)

    row_bbox = gs[row, :].get_position(fig)
    fig.text(0.012, (row_bbox.y0 + row_bbox.y1) / 2, label, rotation=90,
              fontsize=AXIS_LABEL_SIZE + 2, fontweight='bold', va='center', ha='center')

# one shared colorbar (same cmap/norm used by every embed/unembed PCA panel above)
cax = fig.add_axes([0.945, 0.15, 0.012, 0.7])
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=color_norm, cmap=cmap), cax=cax)
cbar.set_label('number', fontsize=AXIS_LABEL_SIZE)
cbar.ax.tick_params(labelsize=TICK_SIZE)

plt.savefig(f'{FIGDIR}/app_ablation_pca.pdf', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'saved {FIGDIR}/app_ablation_pca.pdf')


# %% Fig. 15: order and readout components of the distance-readout model
COLOR_PC1 = '#F4A261'
COLOR_NORM = '#457B9D'
COLOR_LOWERPCS = '#9B5DE5'


def make_pc1_decomposition_fig(We, Wd, label, savepath):
    numbers_ = np.arange(1, We.shape[0] + 1)

    pca_main = PCA(n_components=10, svd_solver='full')
    pca_main.fit(We.numpy())
    pc1_vec = torch.tensor(pca_main.components_[0], dtype=torch.float32)

    We_pc1_comp = (We @ pc1_vec).unsqueeze(-1) * pc1_vec
    We_lowerpcs = We - We_pc1_comp

    We_norm_full = F.normalize(We, dim=-1)
    We_norm_pc1 = F.normalize(We_pc1_comp, dim=-1)
    We_norm_lowerpcs = F.normalize(We_lowerpcs, dim=-1)
    Wd_norm = F.normalize(Wd, dim=-1)
    cos_full_diag = (We_norm_full * Wd_norm).sum(-1).numpy()
    cos_pc1_diag = (We_norm_pc1 * Wd_norm).sum(-1).numpy()
    cos_lowerpcs_diag = (We_norm_lowerpcs * Wd_norm).sum(-1).numpy()

    m0 = 499  # anchor number 500, as in Fig. 3

    def add_panel_label(ax, letter):
        ax.text(-0.18, 1.0, letter, transform=ax.transAxes, fontsize=24, fontweight='bold', ha='right', va='bottom')

    def add_m0_label(ax):
        ax.text(m0 + 1, 1.01, f'i={m0 + 1}', transform=mtransforms.blended_transform_factory(ax.transData, ax.transAxes),
                color='gray', ha='center', va='bottom', fontsize=11)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10))

    # (a) scree plot: full-spectrum variance ratio (log) + cumulative variance ratio
    pca_full = PCA(n_components=min(We.shape))
    scores_full = pca_full.fit_transform(We.numpy())
    pc_idx = np.arange(1, len(pca_full.explained_variance_ratio_) + 1)
    var_ratio = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)
    ax = axes[0, 0]
    ax.plot(pc_idx, var_ratio, color=COLOR_LOWERPCS, marker='o', markersize=3, linewidth=1)
    ax.plot(pc_idx[0], var_ratio[0], 'o', color=COLOR_PC1, markersize=9, zorder=5)
    ax.annotate(f'{var_ratio[0]:.1%}', xy=(pc_idx[0], var_ratio[0]), xytext=(6, 0), textcoords='offset points',
                color=COLOR_PC1, fontsize=11, fontweight='bold', ha='left', va='center')
    ax.set_yscale('log')
    ax_cum = ax.twinx()
    ax_cum.plot(pc_idx, cum_var, color='dimgray', linewidth=1.5)
    ax_cum.set_ylim(0, 1.05)
    ax_cum.spines['top'].set_visible(False)
    ax.set_xlabel('PC index', fontsize=16)
    ax.set_ylabel('variance ratio', fontsize=16)
    ax_cum.set_ylabel('cumulative', fontsize=14, color='dimgray')
    ax.spines['top'].set_visible(False)
    add_panel_label(ax, '(a)')

    # (b) PC1 score vs number, colored by number (same cmap/color_norm as the rest of this notebook)
    ax = axes[0, 1]
    ax.plot(numbers_, scores_full[:, 0], color='lightgray', linewidth=1, zorder=1)
    ax.scatter(numbers_, scores_full[:, 0], c=numbers_, cmap=cmap, norm=color_norm, s=10, zorder=2)
    ax.set_xlabel('number', fontsize=16)
    ax.set_ylabel('PC1 score', fontsize=16)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    add_panel_label(ax, '(b)')

    # (c) PC3 vs PC4: representative geometry of the lower PCs
    ax = axes[0, 2]
    sc = ax.scatter(scores_full[:, 2], scores_full[:, 3], c=numbers_, cmap=cmap, norm=color_norm, s=10)
    fig.colorbar(sc, ax=ax, label='number', pad=0.02)
    ax.set_xlabel('PC3', fontsize=16)
    ax.set_ylabel('PC4', fontsize=16)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    add_panel_label(ax, '(c)')

    # (d) E_m . E_q for fixed m=500
    dots_full = We[m0] @ We.T
    dots_pc1 = We_pc1_comp[m0] @ We_pc1_comp.T
    dots_lowerpcs = We_lowerpcs[m0] @ We_lowerpcs.T
    ax = axes[1, 0]
    ax.plot(numbers_, dots_full.numpy(), color=COLOR_NORM, label='full')
    ax.plot(numbers_, dots_pc1.numpy(), color=COLOR_PC1, label='PC1')
    ax.plot(numbers_, dots_lowerpcs.numpy(), color=COLOR_LOWERPCS, label='other PCs')
    ax.axvline(m0 + 1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    add_m0_label(ax)
    ax.set_xlabel('$j$', fontsize=16)
    ax.set_ylabel(r'$E_i \cdot E_j$', fontsize=16)
    ax.legend(fontsize=10, frameon=False, loc='upper left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    add_panel_label(ax, '(d)')

    # (e) E_m . U_q for fixed m=500
    dots_full = We[m0] @ Wd.T
    dots_pc1 = We_pc1_comp[m0] @ Wd.T
    dots_lowerpcs = We_lowerpcs[m0] @ Wd.T
    ax = axes[1, 1]
    ax.plot(numbers_, dots_full.numpy(), color=COLOR_NORM)
    ax.plot(numbers_, dots_pc1.numpy(), color=COLOR_PC1)
    ax.plot(numbers_, dots_lowerpcs.numpy(), color=COLOR_LOWERPCS)
    ax.axvline(m0 + 1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    add_m0_label(ax)
    ax.set_xlabel('$j$', fontsize=16)
    ax.set_ylabel(r'$E_i \cdot U_j$', fontsize=16)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    add_panel_label(ax, '(e)')

    # (f) histogram of cos(E_m, U_m), q=m only
    ax = axes[1, 2]
    ax.hist(cos_full_diag, bins=30, color=COLOR_NORM, alpha=0.9, label='full', density=True)
    ax.hist(cos_pc1_diag, bins=30, color=COLOR_PC1, alpha=0.9, label='PC1', density=True)
    ax.hist(cos_lowerpcs_diag, bins=30, color=COLOR_LOWERPCS, alpha=0.9, label='other PCs', density=True)
    ax.set_xlabel(r'$\cos(E_i, U_i)$', fontsize=16)
    ax.set_ylabel('density', fontsize=16)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    add_panel_label(ax, '(f)')

    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {savepath}')


We_m, Wd_m = weights['dist_readout']
make_pc1_decomposition_fig(We_m, Wd_m, 'distance readout', f'{FIGDIR}/app_ablation_dist_readout.pdf')
