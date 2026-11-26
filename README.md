# VAE vs DCGAN on Fashion-MNIST

COMP 4102 final project. We compare two ways of generating clothing images. We wanted to see
which model makes sharper images, which covers the real data better, and what happens when we
move between two points in the latent space.

## The two models

- **VAE.** The encoder turns a clothing image into a mean and log-variance. The decoder
  reconstructs the image from a sampled code. Training balances reconstruction error
  with a penalty that keeps the codes close to standard normal noise.
- **DCGAN.** The generator makes clothing images from random noise. The discriminator
  learns to score real and generated images, and the generator learns to raise its
  score on the generated images.

The VAE decoder and the DCGAN generator use the same convolutional layer design. Both
models get the same training data, batch size, latent size, and number of epochs. A
separate Fashion-MNIST classifier supplies features and class predictions for the comparison.

## Setup

Open a terminal in this project folder. Create the environment and install the dependencies once.

```bash
# Create a local Python environment for the project.
python3 -m venv .venv

# Activate the environment in this terminal.
source .venv/bin/activate

# Install the project's Python dependencies.
python -m pip install -r requirements.txt
```

## Usage

With the environment active, choose a quick demo or the full comparison.
Both use the saved models in `trained_models/`.

```bash
# Run the quick demo with 1,000 images per group and save it in comparison/demo/.
python compare.py --samples 1000 --output comparison/demo

# Run the full comparison with 10,000 images per group and save it in comparison/.
python compare.py
```

Fashion-MNIST downloads into `data/` on first use. Add `--data /path/to/data` to a Python
command to reuse a folder that already contains `FashionMNIST/`.

### Training commands for reference

**Training is already complete.** The demo and full comparison use the submitted checkpoints
in `trained_models/`. The commands below are for reference and save new training outputs
under `runs/`.

```bash
# Train the VAE for 30 epochs.
python -m src.train_vae --config configs/vae.yaml

# Train the DCGAN for 30 epochs.
python -m src.train_gan --config configs/gan.yaml

# Train the evaluation classifier for 8 epochs.
python -m src.train_classifier --config configs/classifier.yaml
```

## How compare.py works

Start at `compare()` in [compare.py](compare.py). Both runs follow the same steps.

1. **Prepare the real reference.** Load the saved clothing classifier and read the
   Fashion-MNIST test images. The classifier gives each image a vector of 128 features.
   The test set's actual labels supply the reference clothing-class shares.
2. **Score the real-image baseline.** Take a random sample of real training images and
   compare it with the test images. The `Real train` row shows the scores for two
   different groups of real images.
3. **Generate and score each model's images.** Load the VAE and DCGAN checkpoints in
   turn. Reset the noise seed so both models get the same 32-number vectors. The VAE
   decoder and DCGAN generator turn those vectors into images. `compare_images()`
   uses classifier features for distance, precision, and recall, predicted labels
   for the class mix, and image pixels for the edge score.
4. **Measure changes between images.** `interpolate()` picks 100 pairs of noise
   vectors and makes 10 frames along each path. It measures how evenly the classifier
   features change from one frame to the next. The figure shows one path per model.
5. **Save the outputs.** Write one CSV row each for the real training sample, VAE,
   and DCGAN. Save the image grids and class chart alongside it.

## Results and figures

The repository includes the full comparison in `results/`, so you can open the scores
and figures below before running anything. A fresh full comparison writes to `comparison/`.

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

- [models/__init__.py](src/models/__init__.py) rebuilds the networks and loads the saved weights.
- [metrics.py](src/metrics.py) computes the scores, and [plots.py](src/plots.py) saves the figures.
- [data.py](src/data.py) pads the 28×28 images to 32×32 and scales their pixel values.
- [networks.py](src/models/networks.py) has the shared layers. [vae.py](src/models/vae.py),
  [gan.py](src/models/gan.py), and [classifier.py](src/models/classifier.py) show what each
  model does with its layers.
- [losses.py](src/losses.py) defines the VAE and GAN training losses.

The original training loops are in [train_vae.py](src/train_vae.py), [train_gan.py](src/train_gan.py),
and [train_classifier.py](src/train_classifier.py). They use the settings in [configs/](configs/)
and write to `runs/` by default. The comparison uses the checkpoints in [trained_models/](trained_models/).
