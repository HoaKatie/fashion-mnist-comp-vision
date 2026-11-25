"""One command for the whole VAE vs DCGAN pipeline.

    python run_all.py                     # everything behind results/: main + ablations + analysis
    python run_all.py main                # VAE vs DCGAN at the default settings, seeds 0 1 2 (H1-H3)
    python run_all.py main --seeds 1      # one seed, e.g. to split the seeds across teammates
    python run_all.py ablation            # latent_dim 8/128 (both models), VAE beta 0.5/4, seed 0
    python run_all.py analysis            # rebuild results/ from runs/, no training
    python run_all.py smoke               # ~1 min end-to-end check in runs_smoke/ and results_smoke/

Add --dry-run to any command to print its steps without running them.

Every step is a `python -m src.<module> --config ... --set ...` call, run in a fresh process with
the Python that runs this file (so activate the virtual environment first). A failing step stops
the whole run.
"""
import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ["vae", "gan"]
MAIN_SEEDS = [0, 1, 2]
ABLATION_SEEDS = [0]
DRY_RUN = False


def step(module, *args):
    """Run `python -m src.<module> <args>` from the repo root and stop everything if it fails."""
    cmd = [sys.executable, "-m", f"src.{module}", *map(str, args)]
    print("$ " + shlex.join(["python", *cmd[1:]]), flush=True)
    if DRY_RUN:
        return
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"\nStep failed (exit code {result.returncode}): {shlex.join(['python', *cmd[1:]])}")


def remove(path):
    print(f"$ rm -rf {path.relative_to(ROOT)}", flush=True)
    if not DRY_RUN:
        shutil.rmtree(path, ignore_errors=True)


def dataset_name():
    from src.config import load_config
    return load_config(ROOT / "configs" / "shared.yaml")["dataset"]


def refresh_demo_checkpoints(seeds):
    """Copy the final seed-0 VAE and GAN and the classifier into demo/checkpoints/, so the Q&A demo
    shows the newest models. Skipped when seed 0 was not trained or there is no demo/ folder."""
    demo = ROOT / "demo" / "checkpoints"
    if 0 not in seeds or not demo.parent.is_dir():
        return
    from src.config import load_config
    from src.utils import run_dir_for
    copies = []
    for model in MODELS:
        cfg = load_config(ROOT / "configs" / f"{model}.yaml", ["seed=0"])
        copies.append((ROOT / run_dir_for(cfg) / "checkpoints" / f"epoch_{cfg['epochs']:03d}.pt", demo / f"{model}.pt"))
    classifier = Path(f"classifier_{dataset_name()}") / "classifier.pt"
    copies.append((ROOT / "runs" / classifier, demo / classifier))
    for source, target in copies:
        print(f"$ cp {source.relative_to(ROOT)} {target.relative_to(ROOT)}", flush=True)
        if not DRY_RUN:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def run_main(seeds):
    """Main experiment: VAE vs DCGAN with the shared default settings, one run per seed."""
    runs = ROOT / "runs"
    dataset = dataset_name()
    classifier = runs / f"classifier_{dataset}" / "classifier.pt"
    baseline = runs / f"real_baseline_{dataset}" / "metrics.json"

    # The evaluation classifier is trained once and shared by every evaluation. Teammates who split
    # the seeds must all copy the SAME runs/classifier_<dataset>/ folder before running this,
    # otherwise their classifier-space metrics are not comparable.
    new_classifier = not classifier.is_file()
    if new_classifier:
        step("train_classifier", "--config", "configs/classifier.yaml")
        remove(baseline.parent)  # it was computed with the previous classifier
    # Real-vs-real reference row for the main table (the best score a perfect generator could get).
    if new_classifier or not baseline.is_file():
        step("real_baseline")

    for seed in seeds:
        for model in MODELS:
            config = f"configs/{model}.yaml"
            step(f"train_{model}", "--config", config, "--set", f"seed={seed}")
            step("evaluate", "--config", config, "--set", f"seed={seed}")  # metrics + figures
            step("evaluate", "--config", config, "--set", f"seed={seed}", "--all-epochs", "--quick")  # learning curve
    refresh_demo_checkpoints(seeds)


def run_ablation(seeds):
    """Ablations (run after `main`, which provides latent_dim=32 / beta=1 and the classifier):
    latent dimension 8 / 128 for both models, and beta 0.5 / 4 for the VAE."""
    for seed in seeds:
        for z in [8, 128]:
            for model in MODELS:
                config = f"configs/{model}.yaml"
                step(f"train_{model}", "--config", config, "--set", f"latent_dim={z}", f"seed={seed}")
                step("evaluate", "--config", config, "--set", f"latent_dim={z}", f"seed={seed}")
        for beta in ["0.5", "4"]:
            step("train_vae", "--config", "configs/vae.yaml", "--set", f"beta={beta}", f"seed={seed}")
            step("evaluate", "--config", "configs/vae.yaml", "--set", f"beta={beta}", f"seed={seed}")


def run_smoke():
    """Tiny runs written to runs_smoke/ and results_smoke/, so real results are never touched.
    The metric VALUES are meaningless here: this only checks that every step runs."""
    out = "output_dir=runs_smoke"
    step("train_classifier", "--config", "configs/classifier.yaml", "--set", out, "subset=10000", "epochs=1")
    step("real_baseline", "--set", out, "--n-samples", "1000", "--feature-spaces", "classifier")
    for model in MODELS:
        config = f"configs/{model}.yaml"
        step(f"train_{model}", "--config", config, "--set", out, "subset=1000", "epochs=2", "save_every=1")
        step("evaluate", "--config", config, "--set", out, "--n-samples", "1000", "--feature-spaces", "classifier")
        step("evaluate", "--config", config, "--set", out, "--all-epochs", "--quick")
    step("analysis", "--runs", "runs_smoke", "--out", "results_smoke")


def parse_args():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS,
                        help="print every step without running it")
    parser = argparse.ArgumentParser(description=__doc__, parents=[common],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", metavar="command")

    p = sub.add_parser("all", parents=[common], help="main + ablations + analysis (the default)")
    p.add_argument("--seeds", type=int, nargs="+", default=MAIN_SEEDS, help="main-experiment seeds (default: 0 1 2)")
    p.add_argument("--ablation-seeds", type=int, nargs="+", default=ABLATION_SEEDS, help="(default: 0)")
    p = sub.add_parser("main", parents=[common], help="VAE vs DCGAN at the default settings, then analysis")
    p.add_argument("--seeds", type=int, nargs="+", default=MAIN_SEEDS, help="(default: 0 1 2)")
    p = sub.add_parser("ablation", parents=[common], help="latent_dim and beta ablations, then analysis (run after main)")
    p.add_argument("--seeds", type=int, nargs="+", default=ABLATION_SEEDS, help="(default: 0)")
    p = sub.add_parser("analysis", parents=[common], help="rebuild the results/ tables and figures from runs/")
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results")
    sub.add_parser("smoke", parents=[common], help="~1 min end-to-end check (runs_smoke/, results_smoke/)")

    args = parser.parse_args()
    if args.command is None:  # no command given: run everything
        args = parser.parse_args(["all", *sys.argv[1:]])
    return args


def main():
    global DRY_RUN
    args = parse_args()
    DRY_RUN = getattr(args, "dry_run", False)
    print(f"Python: {sys.executable}" + ("  (dry run: nothing is executed)" if DRY_RUN else ""), flush=True)

    if args.command == "all":
        run_main(args.seeds)
        run_ablation(args.ablation_seeds)
        step("analysis")
    elif args.command == "main":
        run_main(args.seeds)
        step("analysis")
    elif args.command == "ablation":
        run_ablation(args.seeds)
        step("analysis")
    elif args.command == "analysis":
        step("analysis", "--runs", args.runs, "--out", args.out)
    elif args.command == "smoke":
        run_smoke()
        if not DRY_RUN:
            print("Smoke test passed. Look at results_smoke/ and runs_smoke/*/eval/.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
