# VAE vs DCGAN on Fashion-MNIST

This is our COMP 4102 final project. We wanted to see how a VAE and a DCGAN compare on
Fashion-MNIST, mainly which one makes sharper images and which one covers the dataset better.

To keep the comparison fair, both models are built from the same blocks. The VAE decoder is also
the GAN generator, and the VAE encoder is also the GAN discriminator. They also get the same latent
size, data and number of epochs, so the only things that change are the loss and the optimizer
settings.

We started with three hypotheses:

1. The GAN makes sharper images. This turned out to be true.
2. The VAE covers the data better and trains more stably. This was wrong, the GAN did better here too.
3. The VAE has a smoother latent space. This was true, but the β ablation shows it is not that simple.

## Results

These are averages over 3 seeds. The full table is in [results/main_results.md](results/main_results.md),
and the figures and ablation tables are in [results/](results/).

| | DCGAN | VAE | Real images |
|---|---|---|---|
| FID, classifier features ↓ | **1.29** | 20.7 | 0.03 |
| FID, Inception features ↓ | **13.3** | 56.9 | 2.4 |
| Precision / recall, classifier features ↑ | **0.85 / 0.85** | 0.70 / 0.49 | 0.90 / 0.89 |
| Sharpness, Laplacian variance (closer to real is better) | **0.146** | 0.038 | 0.150 |
| Interpolation CV, latent smoothness ↓ | 0.69 | **0.51** | – |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The code picks CUDA, Apple MPS or the CPU on its own. Fashion-MNIST (about 30 MB) is downloaded the
first time you run anything, and the Inception weights for FID (about 100 MB) the first time you run
a full evaluation.

## How to run

Everything goes through `run_all.py`. We suggest running the smoke test first. It takes about a
minute and checks that every step works.

```bash
python run_all.py smoke
python run_all.py
```

The second command runs the whole project. If you only want one part, use `main` for the VAE vs
DCGAN runs (seeds 0, 1 and 2, about an hour on an Apple-silicon MacBook), `ablation` for the latent
size and β ablations, or `analysis` to rebuild `results/` from `runs/`. Adding `--dry-run` prints
every command without running it, which is handy if you want to run one step by hand.

## Checking our numbers without training

`trained_models/` has our final seed-0 models. These two commands take less than a minute and should
give the same numbers as the seed-0 rows in `results/all_runs.csv`.

```bash
python -m src.evaluate --config configs/vae.yaml --set output_dir=trained_models --feature-spaces classifier
python -m src.evaluate --config configs/gan.yaml --set output_dir=trained_models --feature-spaces classifier
```

## A note on the VAE loss

In `src/losses.py` we sum the reconstruction error over all pixels instead of taking the mean. The mean would divide the error by the 1024 pixels but not the KL term, so the KL would end up about 1000 times stronger. That is like setting β to 1000, and the samples get a lot blurrier.

## Code

```
run_all.py            runs the whole project
configs/              shared.yaml (settings both models share) + vae, gan, classifier
src/data.py           loads Fashion-MNIST as 32x32 images in [-1, 1]
src/models/           networks.py (shared blocks), vae.py, gan.py, classifier.py (only for evaluation)
src/losses.py         VAE loss and GAN losses
src/train_*.py        training
src/metrics.py        FID, precision/recall, sharpness, latent smoothness, ...
src/evaluate.py       all metrics and figures for one run
src/real_baseline.py  same metrics on real images, the best score we can hope for
src/analysis.py       collects all runs into results/
results/              tables and figures
trained_models/       our final seed-0 models
```

## References

- Kingma & Welling (2014). Auto-Encoding Variational Bayes
- Goodfellow et al. (2014). Generative Adversarial Nets
- Radford et al. (2016). Unsupervised Representation Learning with Deep Convolutional GANs
- Kingma & Ba (2015). Adam: A Method for Stochastic Optimization
- Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework
- Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)
- Szegedy et al. (2016). Rethinking the Inception Architecture for Computer Vision (Inception-v3)
- Kynkäänniemi et al. (2019). Improved Precision and Recall Metric for Assessing Generative Models
- Pech-Pacheco et al. (2000). Diatom Autofocusing in Brightfield Microscopy (Laplacian variance)
- White (2016). Sampling Generative Networks (slerp)
- Karras et al. (2019). A Style-Based Generator Architecture for GANs (perceptual path length)
- Xiao et al. (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms
