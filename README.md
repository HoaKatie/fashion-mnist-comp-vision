# VAE vs. GAN: Fidelity–Diversity Trade-off on Fashion-MNIST

COMP 4102 final project. We compare two generative models, a **Variational Autoencoder (VAE)**
and a **Deep Convolutional GAN (DCGAN)**, on Fashion-MNIST.

> **Research question.** With the *same* network architecture, latent size, data and training
> budget, how do a VAE and a GAN trade off **fidelity / sharpness** against **diversity / mode
> coverage**?

| | Hypothesis | Evidence |
|---|---|---|
| **H1** | The GAN produces sharper samples and a lower FID. | FID, precision, Laplacian variance, high-frequency ratio |
| **H2** | The VAE covers the data distribution better (higher recall, less mode collapse) and trains more stably. | recall, class histogram, variance over seeds, D(x)/D(G(z)) curves |
| **H3** | The VAE has a smoother latent space. | interpolation figures, interpolation step CV |

## Results at a glance

Mean over 3 seeds. Full table: [results/main_results.md](results/main_results.md). All figures
are in [`results/`](results/); start with `qualitative_samples.png` and `precision_recall.png`.

| | DCGAN | VAE | Real data |
|---|---|---|---|
| FID, classifier features ↓ | **1.29** | 20.7 | 0.03 |
| FID, Inception features ↓ | **13.3** | 56.9 | 2.4 |
| Precision / recall, classifier features ↑ | **0.85 / 0.85** | 0.70 / 0.49 | 0.90 / 0.89 |
| Sharpness: Laplacian variance (closest to real is best) | **0.146** | 0.038 | 0.150 |
| Latent smoothness: interpolation CV ↓ | 0.69 | **0.51** | – |

- **H1 supported.** DCGAN samples are as sharp as real images; VAE samples are blurry.
- **H2 not supported.** The DCGAN also covers the data better and never mode-collapsed. The
  VAE's blur costs it diversity as well as fidelity.
- **H3 supported.** The VAE's latent space is smoother, with a caveat from the β ablation
  ([results/ablation_beta.md](results/ablation_beta.md)).

## Fair comparison by construction

- **One architecture.** `ConvDecoder` is both the VAE decoder and the GAN generator, and
  `ConvEncoder` is both the VAE encoder and the GAN discriminator
  ([src/models/networks.py](src/models/networks.py)).
- **One settings file.** Dataset, latent size, batch size, epochs and seed live in
  [configs/shared.yaml](configs/shared.yaml) and are inherited by both model configs.
- **One evaluation.** Both models are scored by the same script with the same latent vectors
  (fixed `--eval-seed`).

Only the training objective and the optimizer settings recommended by each model's paper differ.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The code runs on CUDA, Apple-silicon (MPS) or CPU and picks the device automatically.
Fashion-MNIST (~30 MB) downloads on first use. The FID Inception weights (~100 MB) download on
the first full evaluation.

## Running everything: `run_all.py`

```bash
python run_all.py smoke             # ~1 min end-to-end check; writes runs_smoke/ and results_smoke/
python run_all.py                   # the full pipeline behind results/: main + ablations + analysis
```

| Command | What it runs |
|---|---|
| `python run_all.py main` | evaluation classifier + real-data baseline (once), then VAE and DCGAN for seeds 0 1 2: train, evaluate, learning curve |
| `python run_all.py main --seeds 1` | one seed only, e.g. to split the seeds across teammates (then merge the `runs/` folders) |
| `python run_all.py ablation` | latent_dim 8 / 128 for both models and VAE β 0.5 / 4, seed 0 (run after `main`) |
| `python run_all.py analysis` | rebuild the `results/` tables and figures from `runs/`, no training |
| `python run_all.py smoke` | tiny runs of every step (~1 min), to check the setup before the long runs |

Add `--dry-run` to any command to print its steps without running them, and `-h` for all options.
On an Apple-silicon laptop (MPS), `main` takes about 1 hour and `ablation` about 45 min; on macOS,
prefix long runs with `caffeinate -i` to keep the laptop awake.

Teammates who split the seeds must all use the **same** evaluation classifier
(`runs/classifier_fashion_mnist/`): copy it before running `main`, otherwise the classifier-space
metrics are not comparable.

Each step can also be run on its own. Every module takes the same `--config` / `--set` flags:

```bash
python -m src.train_classifier --config configs/classifier.yaml           # once per dataset
python -m src.real_baseline                                                # real-vs-real reference row
python -m src.train_vae --config configs/vae.yaml --set seed=1 beta=4
python -m src.train_gan --config configs/gan.yaml --set seed=1 latent_dim=8
python -m src.evaluate  --config configs/gan.yaml --set seed=1             # full metrics + figures
python -m src.evaluate  --config configs/gan.yaml --set seed=1 --all-epochs --quick   # learning curve
python -m src.analysis                                                     # runs/ -> results/
```

## Check our numbers without training

`trained_models/` has our final seed-0 VAE, DCGAN and evaluation classifier (13 MB, same folders
as `runs/`). Evaluating them takes under a minute, even on a CPU, and gives the seed-0 rows of
`results/all_runs.csv` (not the 3-seed mean above):

```bash
python -m src.evaluate --config configs/vae.yaml --set output_dir=trained_models --feature-spaces classifier
python -m src.evaluate --config configs/gan.yaml --set output_dir=trained_models --feature-spaces classifier
```

Remove `--feature-spaces classifier` to also get the Inception metrics. `run_all.py main` replaces
these files with the newly trained models.

## Key design choice: the VAE loss

`vae_loss` in [src/losses.py](src/losses.py) sums the squared error over the pixels of each
image (a Gaussian decoder with fixed variance). It does not average it. Averaging would make
the KL term about 1000× stronger (1024 pixels vs. 32 latent dimensions), which is a hidden
β ≈ 1000 and makes the samples much blurrier. With the sum, `beta` is the only weight on the
KL.

## Outputs

| Path | Content |
|---|---|
| `results/main_results.md` | main table, mean ± std over seeds, with the real-data reference row |
| `results/ablation_latent_dim.md`, `results/ablation_beta.md` | ablation tables |
| `results/all_runs.csv` | every final metric of every run |
| `results/*.png` | precision–recall, mode coverage, sharpness, learning and training curves, ablations, qualitative grids |
| `trained_models/` | final seed-0 checkpoints (committed), see above |
| `runs/<run>/` | created locally when you train (not committed): config, loss log, per-epoch samples, `eval/` metrics and figures, checkpoints |

## Repository layout

```
run_all.py          one command for the whole pipeline (see above)
configs/            shared.yaml (fairness) + vae.yaml, gan.yaml, classifier.yaml
src/
  data.py           Fashion-MNIST -> 32x32 in [-1, 1]
  models/           networks.py (shared blocks), vae.py, gan.py, classifier.py (evaluation only)
  losses.py         VAE negative ELBO and GAN losses
  train_vae.py, train_gan.py, train_classifier.py
  metrics.py        FID, precision/recall, mode coverage, sharpness, PSNR, NN, latent smoothness
  evaluate.py       all metrics + qualitative figures for one run
  real_baseline.py  same metrics on real train vs. test images (best achievable scores)
  analysis.py       aggregates all runs into tables and figures
results/            tables + figures (created by src.analysis, then committed)
trained_models/     final seed-0 VAE, DCGAN and classifier (copied from runs/ by run_all.py)
```

## References

- Kingma & Welling (2014). *Auto-Encoding Variational Bayes.* ICLR.
- Goodfellow et al. (2014). *Generative Adversarial Nets.* NeurIPS.
- Radford, Metz & Chintala (2016). *Unsupervised Representation Learning with Deep Convolutional GANs.* ICLR.
- Kingma & Ba (2015). *Adam: A Method for Stochastic Optimization.* ICLR.
- Higgins et al. (2017). *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.* ICLR.
- Heusel et al. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium* (FID). NeurIPS.
- Szegedy et al. (2016). *Rethinking the Inception Architecture for Computer Vision* (Inception-v3, the network behind FID). CVPR.
- Kynkäänniemi et al. (2019). *Improved Precision and Recall Metric for Assessing Generative Models.* NeurIPS.
- Pech-Pacheco et al. (2000). *Diatom autofocusing in brightfield microscopy: a comparative study* (variance of Laplacian). ICPR.
- White (2016). *Sampling Generative Networks* (slerp).
- Karras et al. (2019). *A Style-Based Generator Architecture for GANs* (perceptual path length). CVPR.
- Xiao, Rasul & Vollgraf (2017). *Fashion-MNIST.* arXiv:1708.07747.
- torchmetrics / torch-fidelity for the FID Inception-v3 network.
