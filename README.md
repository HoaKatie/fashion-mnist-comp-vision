# VAE vs DCGAN on Fashion-MNIST

COMP 4102 final project. We compare two models that make clothing images. We look at
fidelity (how close the images are to real ones), diversity (how much of the real data
they cover), and how the images change as we move between two latent codes.

## The two models

- **VAE.** The encoder uses each clothing image to find a mean and log-variance.
  We use those numbers to sample a code, and the decoder tries to rebuild the image
  from it. The loss checks how well we rebuilt the image and keeps the encoder's
  distributions close to standard normal. That lets us use random noise to make new images.
- **DCGAN.** The generator turns random noise into clothing images. The discriminator
  learns to spot the fakes. The generator learns to make images that the discriminator
  is more likely to score as real.

The VAE decoder and DCGAN generator use the same layer design. We train both models
with the same data, batch size, latent size, and number of epochs. A separate
Fashion-MNIST classifier gives us image features and predicted labels for the comparison.

## Setup

Open a terminal in this project folder. Set up the Python environment and install the
packages once.

```bash
# Create the Python environment
python3 -m venv .venv

# Activate the environment in this terminal
source .venv/bin/activate

# Install the packages for this project
python -m pip install -r requirements.txt
```

## Usage

Once the environment is active, pick the full comparison or the quick demo. Both use
the saved models in `trained_models/`.

```bash
# Full CPU run with 10,000 images per group, saves in results/
python compare.py

# Quick run with 1,000 images per group, saves in results/custom-compare/
python compare.py --samples 1000
```

If you change the sample count, device, or checkpoints, the results go to
`results/custom-compare/` by default. Use `--output results/my-comparison` to
give a run its own folder. The script reads output paths from the project folder.

The first run downloads Fashion-MNIST to `data/`. If you already have the data, add
`--data /path/to/data` to the compare command. Use the folder that contains `FashionMNIST/`.

### Training commands (For REFERENCE ONLY - since we already ran it)

**We already trained the models.** The demo and full comparison use the checkpoints
in `trained_models/`. These commands show how we trained them. Running them saves
new training outputs in `runs/`.

```bash
# Train the VAE for 30 epochs
python -m src.train_vae --config configs/vae.yaml

# Train the DCGAN for 30 epochs
python -m src.train_gan --config configs/gan.yaml

# Train the classifier for 8 epochs
python -m src.train_classifier --config configs/classifier.yaml
```

## How compare.py works

Start at `compare()` in [compare.py](compare.py). The quick and full runs use the same steps.

1. **Set up the real test images.** Load the saved classifier and get 128 features
   for each test image. Use the true labels to count the share of each clothing class.
2. **Score a real-image baseline.** Pick random images from the training split and
   score them against the test images. The `Real train` row shows what the scores
   look like when both groups are real.
3. **Make and score images from each model.** Load the VAE and DCGAN one at a time.
   Reset the random seed so both saved models get the same 32-number noise vectors.
   The VAE decoder and DCGAN generator turn the noise into images. In `compare_images()`
   we use classifier features for distance, precision, and recall, predicted labels
   for the class mix, and pixels for the edge score.
4. **Move between two images.** `interpolate()` picks 100 pairs of noise vectors and
   makes 10 frames along each path. We use the classifier features to check how
   evenly the images change from frame to frame. The figure shows one path for each model.
5. **Save the scores and figures.** Write a CSV row for the real training sample,
   the VAE, and the DCGAN. Save the image grids and class chart in the same folder.

## Results and figures

The full comparison is already in `results/`, so you can open the files below right away.
Run `python compare.py` to make the comparison CSV and four figures again.

| File | What to look at |
|---|---|
| [metrics.csv](results/metrics.csv) | Lower Fréchet distance means the two groups have more similar features overall. Precision is the share of generated images close to real ones. Recall is the share of real images covered by the generated set. Compare Laplacian variance with the real row to see how close the edge strength is. |
| [samples.png](results/samples.png) | The first 16 test images and the first 16 samples from each model. |
| [class_balance.png](results/class_balance.png) | The true class mix in the test set and the predicted mix for each model. Lower class KL means the mix is closer to the real one. |
| [successes_and_failures.png](results/successes_and_failures.png) | The 8 images the classifier is most sure about and the 8 it is least sure about for each model. Look at which clothing details are clear and which are hard to make out. |
| [interpolation.png](results/interpolation.png) | One path of 10 images for each model. The CSV score uses 100 paths. Lower interpolation CV means the images change more evenly from frame to frame. |
| [training_curves.png](results/training_curves.png) | The losses after each epoch, plus feature distance and recall every five epochs for both saved seed-0 runs. |

Run `python plot_training.py` to rebuild the training plots from the
[saved CSVs](results/training/). The checkpoint plots use 5,000 images per
group. The final scores in `metrics.csv` use 10,000 per group.

For the saved seed-0 models, the DCGAN had higher recall (0.856 vs. 0.488) and its
edge variance (0.148) was close to the real images (0.150). The VAE had more even
changes along its paths (CV of 0.503 vs. 0.696). The classifier got 90.7% on the
real test images, so we use its labels for generated clothes as estimates.

## Where to read the code

- [build.py](src/models/build.py) builds the models and saves or loads their checkpoints.
- [metrics.py](src/metrics.py) works out the scores, and [plots.py](src/plots.py) saves the figures.
- [data.py](src/data.py) pads the images from 28×28 to 32×32 and scales the pixels.
- [networks.py](src/models/networks.py) has the layer designs the two models use.
  [vae.py](src/models/vae.py), [gan.py](src/models/gan.py), and
  [classifier.py](src/models/classifier.py) show how each network works.
- [losses.py](src/losses.py) shows how we calculate the VAE and GAN training losses.

To see how the models train, read [train_vae.py](src/train_vae.py), [train_gan.py](src/train_gan.py),
and [train_classifier.py](src/train_classifier.py). They read settings from [configs/](configs/)
and save to `runs/` by default. The comparison loads its models from [trained_models/](trained_models/).
