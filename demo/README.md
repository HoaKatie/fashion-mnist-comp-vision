# Demo

We made this small window to walk through the project step by step. It has 7 steps: the dataset,
the setup, training, one step for each hypothesis, and the conclusion.

Run it from the repo folder:

```bash
python demo/demo.py
```

It doesn't train anything. It just loads our models from `trained_models/` and runs on the CPU.
Use the arrow keys or click the tabs to move between steps, press `n` for new samples on steps 4 to 6,
and `Esc` to close. The text under each step is in `CAPTIONS` at the top of `demo.py`.

When someone re-runs `run_all.py main`, the demo picks up the new models on its own. Just remember
to commit `results/` and `trained_models/` together so the pictures and the table still match.

If Tkinter is missing, install it with `brew install python-tk@3.12` on macOS or
`sudo apt install python3-tk` on Ubuntu.
