# Fashion-MNIST code walkthrough

The project trains a VAE and a DCGAN on Fashion-MNIST and compares the images they generate.
A separate clothing classifier provides features and class predictions for evaluation.
The three saved checkpoints are already in `trained_models/`, so the demo can start
at the comparison step.

Run the commands below from the inner project folder that contains `compare.py`,
`configs/`, and `trained_models/`.

## Files to read in order

| Order | Files | What to look for |
|---:|---|---|
| 1 | [README.md](../README.md), [requirements.txt](../requirements.txt) | The project question, the results we report, and the Python dependencies. |
| 2 | [shared.yaml](../configs/shared.yaml), [vae.yaml](../configs/vae.yaml), [gan.yaml](../configs/gan.yaml), [classifier.yaml](../configs/classifier.yaml), [config.py](../src/config.py) | Read the shared settings first. Each model file supplies its own values. `load_config()` reads them in that order and applies any `--set` options last. |
| 3 | [data.py](../src/data.py) | See how we pad each 28 by 28 image to 32 by 32 and scale the pixels to `[-1, 1]`. Training reads mini-batches. The comparison reads image tensors from the train and test splits. |
| 4 | [networks.py](../src/models/networks.py) | Follow the shapes through `ConvDecoder` and `ConvEncoder`. The decoder grows a code into an image. The encoder shrinks an image into a short vector. |
| 5 | [vae.py](../src/models/vae.py), [gan.py](../src/models/gan.py) | The VAE encoder gives a mean and log-variance, then the decoder reconstructs the image from a sampled code. In the DCGAN, `G` makes the image and `D` scores it as real or fake. Both use the shared layer designs with their own weights. |
| 6 | [losses.py](../src/losses.py), [train_vae.py](../src/train_vae.py), [train_gan.py](../src/train_gan.py) | Read the loss functions and find the matching calls in the training loops. The VAE updates its encoder and decoder together. The DCGAN alternates between updating `D` and updating `G`. |
| 7 | [classifier.py](../src/models/classifier.py), [train_classifier.py](../src/train_classifier.py) | The classifier learns the ten clothing classes. The comparison uses its 128 image features to score the two generators and its predicted labels to draw the class chart. |
| 8 | [models/__init__.py](../src/models/__init__.py), [utils.py](../src/utils.py) | The model package builds the networks and saves or loads their weights. The utilities handle the device, random seeds, run folders, and training logs. |
| 9 | [compare.py](../compare.py), [metrics.py](../src/metrics.py), [plots.py](../src/plots.py) | Start at `compare()`. Follow it from loading checkpoints through generating images and writing scores. Open `metrics.py` when you reach a score calculation and `plots.py` when you reach a figure. |
| 10 | [results/](../results/), [report.tex](report.tex) | Look at the full-run scores and figures, then read the report's Method and Analysis sections to connect the numbers to the two models. |

[`src/__init__.py`](../src/__init__.py) makes `src` importable. [`.gitignore`](../.gitignore)
keeps local data, environments, and new run outputs out of Git.

## Set up the Python environment

Set up a fresh environment on macOS or Linux with these commands.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If you already have the project's Conda environment, activate it with
`conda activate project-comp-vision`.

## Run the demo using the saved models

```bash
python compare.py --samples 1000
```

The script uses the bundled checkpoints and writes the output to `comparison/`.
On first use, Torchvision downloads Fashion-MNIST into `data/`. To reuse an
existing dataset, pass the folder that contains `FashionMNIST/`.

```bash
python compare.py --samples 1000 --data /path/to/data
```

The comparison runs in this order.

1. Load the classifier and use it to get features for the real test images.
2. Score a sample of real training images against the test images. This gives a baseline for comparing two sets of real clothes.
3. Load each generator, give it the same noise vectors, then score its generated images against the same test images.
4. Compare the pictures between pairs of noise vectors to measure how evenly the images change.
5. Write one CSV with the scores and four PNGs.

Start with `samples.png` to show the real images, VAE images, and DCGAN images side by side.
Then open `class_balance.png` to see which clothing classes each model tends to generate.
`successes_and_failures.png` shows the highest- and lowest-confidence examples from the
classifier. `interpolation.png` shows one path per model.

The quick demo uses 1,000 images per group. Use [the saved metrics CSV](../results/metrics.csv)
when quoting figures from the report's 10,000-image comparison. In that run, feature precision
and recall were higher for the DCGAN, and its edge variance was closer to the real baseline.
The VAE's interpolation steps were more even.

To run the full comparison with the saved checkpoints, use `python compare.py`.
Run this command to regenerate the report's CSV and figures in `results/`.

```bash
python compare.py --device cpu --output results
```

## Training commands for reference

Each command trains one model and saves it under `runs/`. With these settings, the
VAE and DCGAN run for 30 epochs each and the classifier runs for 8 epochs. A rerun
with the same settings can replace that run's outputs in `runs/`.

```bash
python -m src.train_vae --config configs/vae.yaml
python -m src.train_gan --config configs/gan.yaml
python -m src.train_classifier --config configs/classifier.yaml
```

After those commands finish, pass the new checkpoint paths to the same comparison.
This example writes its results to `comparison/retrained/`.

```bash
python compare.py \
  --vae runs/vae_fashion_mnist_z32_beta1_seed0/checkpoints/epoch_030.pt \
  --gan runs/gan_fashion_mnist_z32_seed0/checkpoints/epoch_030.pt \
  --classifier runs/classifier_fashion_mnist/classifier.pt \
  --output comparison/retrained
```

## Where the bundled checkpoints came from

The older project used `bash scripts/run_main.sh` to call the three training
modules and save checkpoints in `runs/`. Before the cleanup, this project's
`python run_all.py main` also had a step that copied the selected seed-0 generator
checkpoints and classifier into `trained_models/`. The bundled files match the
corresponding older checkpoints byte for byte.

Today the training commands write to `runs/`, and `compare.py` defaults to the
bundled files in [trained_models/](../trained_models/). Each generator checkpoint
includes the original network settings. The loader uses them to build the network
before it loads the saved weights.

