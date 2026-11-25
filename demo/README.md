# Q&A demo: VAE vs DCGAN in 7 steps

A small window that walks through the whole project with the trained models

## Run it

From the repository root, with the virtual environment active (see the main README):

```bash
python demo/demo.py
```

Nothing is trained: the demo loads the three checkpoints in `trained_models/` (main run,
seed 0) and runs on the CPU. The Diversity step takes a few seconds.

| Step | What it shows |
|---|---|
| 1. Dataset | 4 real test images of each of the 10 classes |
| 2. Setup | the VAE and the GAN are built from the same encoder / decoder blocks; only the loss and the optimizer settings differ |
| 3. Training | the pipeline, plus the training and learning curves saved in `results/` |
| 4. Sharpness (H1) | the first 16 real, VAE and DCGAN images side by side, with their Laplacian variance |
| 5. Diversity (H2) | share of 5,000 generated images per class, against the real 10 % |
| 6. Latent space (H3) | a path from z₀ to z₁ decoded by both models |
| 7. Conclusion | the main-results table from `results/all_runs.csv` and the ✓ / ✗ for H1–H3, computed from it |

Keys: `←` / `→` = Back / Next, `n` = New samples (steps 4–6 only), `Esc` = close. Click a tab to
jump to that step.
The demo always starts at seed 0, so it shows the same images every time you rehearse.
The captions under each step are the `CAPTIONS` list at the top of `demo.py`.

## When the numbers change

`python run_all.py` (or `run_all.py main` with seed 0) copies the newly trained seed-0 VAE,
GAN and classifier into `trained_models/`. Run the demo to check it, then commit
`results/` and `trained_models/` **together**, so the pictures and the Conclusion table match.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Tkinter is not available` / `No module named '_tkinter'` | macOS + Homebrew: `brew install python-tk@3.12`; Ubuntu: `sudo apt install python3-tk` |
| `Checkpoint not found` | `git pull` (`trained_models/` is committed) |
| The window does not appear | it opens in front for a moment; if it is still hidden, look for the Python icon in the Dock / taskbar |
| First start is slow | Fashion-MNIST (~30 MB) downloads to `data/` once |
