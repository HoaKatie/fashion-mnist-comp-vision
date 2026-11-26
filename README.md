# VAE vs DCGAN on Fashion-MNIST

COMP 4102 final project. We compare two ways of generating clothing images. We wanted to see
which model makes sharper images, which covers the real data better, and what happens when we
move between two points in the latent space.

The VAE decoder and the DCGAN generator use the same convolutional layers. Both models get
the same training data, batch size, latent size, and number of epochs. The VAE learns by
reconstructing images and regularizing its latent space. The DCGAN learns by trying to fool a
discriminator.

## Run the comparison

Open a terminal in this folder.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python compare.py
```

The command loads the two saved seed-0 models and the clothing classifier from `trained_models/`.
It downloads Fashion-MNIST into `data/` on first use. For a quick look, run
`python compare.py --samples 1000`. Use `--data /path/to/data` if you already have a folder
containing `FashionMNIST/`.

## What the comparison does

We generate 10,000 images from each model using the same random noise. For the image quality
and coverage scores, we compare each generator with the real test split. We also score
10,000 real training images against the test split to get a real-image reference.

The command saves its files in `comparison/`. The links below open the results used in
the [LaTeX report](report/report.tex). To regenerate those files from the saved checkpoints,
run `python compare.py --device cpu --output results`.

| File | What to look at |
|---|---|
| [metrics.csv](results/metrics.csv) | Lower Fréchet distance means the feature distributions are closer. Precision measures how many generated images fall near real ones. Recall measures how much of the real data the generator covers. For Laplacian variance, compare the edge strength to the real row. |
| [samples.png](results/samples.png) | The first 16 test images and the first 16 samples from each model. |
| [class_balance.png](results/class_balance.png) | The actual test classes and the classifier's predicted classes for generated images. A lower class KL means the shares are closer to the real mix. |
| [successes_and_failures.png](results/successes_and_failures.png) | The eight highest- and lowest-confidence examples for each model. Look at the images to see which details are clear and which ones are hard to identify. |
| [interpolation.png](results/interpolation.png) | One 10-image path for each model. The CSV scores 100 paths. Lower interpolation CV means more even changes between frames. |

For these seed-0 checkpoints, the DCGAN reached a recall of 0.856 and an edge variance of
0.148, close to the real-image value of 0.150. The VAE's recall was 0.488, and it had the
more even latent paths (CV 0.503 against 0.696). The classifier scored 90.7% on real test
images, so treat its predictions for generated clothes as estimates.

## Where to read the code

- [compare.py](compare.py) loads the models, generates images, and runs the comparison.
- [metrics.py](src/metrics.py) computes the scores, and [plots.py](src/plots.py) saves the figures.
- [data.py](src/data.py) pads the 28×28 images to 32×32 and scales their pixel values.
- [networks.py](src/models/networks.py) has the shared layers. [vae.py](src/models/vae.py),
  [gan.py](src/models/gan.py), and [classifier.py](src/models/classifier.py) show what each
  model does with its layers.
- [losses.py](src/losses.py) has the VAE and GAN losses that are written out in the report.

The original training loops are in [train_vae.py](src/train_vae.py), [train_gan.py](src/train_gan.py),
and [train_classifier.py](src/train_classifier.py). They use the settings in [configs/](configs/)
and write to `runs/` by default. The comparison uses the checkpoints in [trained_models/](trained_models/).

## Open the report in Overleaf

Copy [report.tex](report/report.tex) into `main.tex`, fill in the team names, and upload the
four PNGs linked above to an Overleaf folder called `results`. Use pdfLaTeX. The references
are included in the source.
