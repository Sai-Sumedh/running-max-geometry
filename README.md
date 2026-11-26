# running-max-geometry

Code for the preprint **On the Dissociation of Order and Identity in Number Manifold Representations**
(arXiv link to be added).

A single attention layer, with its query, key and value maps fixed to the identity, is trained to output the
running maximum of a sequence of numbers. Its attention scores are inner products of number embeddings, so one
embedding table must serve two operations: comparison, which ranks the numbers, and identity readout, which names
the largest one. The paper derives the geometric conditions that these operations impose and shows that the
trained model splits its embeddings into an *order component* along one direction (PC1), which decides comparison,
and an orthogonal *readout component*, which decides identity. This repository contains the code that generates
the data, trains the models, runs the analyses, and makes every figure and table in the paper.

## Setup

Python 3.10. A GPU is recommended for training and for the analyses; the figure scripts run on a CPU.

```bash
pip install -r requirements.txt
```

All outputs are written to `outputs/` at the root of the repository; set the environment variable `RMG_OUT` to
use another location.

## Reproducing the paper

Run the commands from the root of the repository, in this order.

**1. Data** (CPU, about a minute). These commands reproduce the datasets used in the paper exactly.

```bash
python -m rmg.data --name running_max_500k --num_seq 500000
python -m rmg.data --name running_max_50k --num_seq 50000
```

**2. Training.** Four seeds of the baseline and two architectural variants. A baseline run takes about 15
minutes on one GPU.

```bash
for s in 0 1 2 3; do
  python -m rmg.train --run_name baseline_seed$s --dataset running_max_500k --seed $s
done
python -m rmg.train --run_name dist_attention --dataset running_max_500k --dist_attn
python -m rmg.train --run_name dist_readout --dataset running_max_50k --decoder_type dist --epochs 50
```

**3. Analyses** (GPU if available).

```bash
python -m analysis.analyze --run baseline_seed0 --epochs --extras
for s in 1 2 3; do python -m analysis.analyze --run baseline_seed$s; done
python -m analysis.pairs --runs baseline_seed0 baseline_seed1 baseline_seed2 baseline_seed3
python -m analysis.anchors --runs baseline_seed0 baseline_seed1 baseline_seed2 baseline_seed3
```

**4. Figures and tables** (CPU).

```bash
python -m figures.make_figures
python -m figures.make_ablation_figures
```

## Where each result comes from

Files are written to `outputs/figures/`. Figure 1 (the model diagram) is drawn in LaTeX.

| Paper | File | Script |
|---|---|---|
| Fig. 2 | `fig2_geometry.pdf` | `figures/make_figures.py` |
| Fig. 3 | `fig3_components.pdf` | `figures/make_figures.py` |
| Fig. 4 | `fig4_two_token.pdf` | `figures/make_figures.py` |
| Fig. 5 | `fig5_patching.pdf` | `figures/make_figures.py` |
| Fig. 6 | `app_theorem1.pdf` | `figures/make_figures.py` |
| Fig. 7 | `app_steering_controls.pdf` | `figures/make_figures.py` |
| Fig. 8 | `app_iia_outcomes.pdf` | `figures/make_figures.py` |
| Fig. 9 | `app_saturation.pdf` | `figures/make_figures.py` |
| Fig. 10 | `app_interference.pdf` | `figures/make_figures.py` |
| Fig. 11 | `app_pairs.pdf` | `figures/make_figures.py` |
| Fig. 12 | `app_toeplitz.pdf` | `figures/make_figures.py` |
| Fig. 13 | `app_epochs.pdf` | `figures/make_figures.py` |
| Fig. 14 | `app_ablation_pca.pdf` | `figures/make_ablation_figures.py` |
| Fig. 15 | `app_ablation_dist_readout.pdf` | `figures/make_ablation_figures.py` |
| Table 1 | `tab_seeds.md` | `figures/make_figures.py` |
| Table 2 | `tab_pairs.md` | `figures/make_figures.py` |
| Table 3 | training commands above; accuracies in `tab_checks.md` | `figures/make_figures.py` |
| Other numbers in the text | `tab_checks.md`, `tab_steering.md`, `tab_iia.md`, `outputs/results/*.json` | `analysis/*.py` |

## Repository layout

```
rmg/         data generation (data.py), model (model.py), training (train.py), output paths (paths.py)
analysis/    analyze.py: geometry, Theorem 1 conditions, patching, decoy interchange, attention saturation,
             interference budget, order directions without PCA, translation symmetry, training epochs
             pairs.py: all two-token inputs
             anchors.py: Theorem 1 curves for chosen anchor numbers
figures/     make_figures.py: figures and tables from outputs/results
             make_ablation_figures.py: figures of the architectural variants from the final weights
```

## Notes

- Token `v` stands for the number `v + 1`; the vocabulary has 1000 tokens and sequences have 100 positions.
- Seeds 1 to 3 use exactly the settings of the paper's runs. The paper's seed-0 baseline and the two variants
  were trained without a fixed seed; the commands above fix seed 0 for them, so their numbers can differ slightly
  from those in the paper.
- The distance readout computes the logits `(2/sqrt(d)) z.U_j - |z|^2 - |U_j|^2`, a linear readout with the class
  bias `-|U_j|^2`.

## License

MIT; see `LICENSE`.
