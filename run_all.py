"""Run the whole pipeline. Choose one command: all (default), main, ablation, analysis or smoke
Add --dry-run to print the steps without running them"""
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
    """Run `python -m src.<module> <args>` from the repo root and stop everything if it fails"""
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

def refresh_trained_models(seeds):
    """Copy the final seed-0 VAE and GAN and the classifier from runs/ to trained_models/ (same layout),
    so the committed models always match results/. Skipped when seed 0 was not trained"""
    if 0 not in seeds:
        return
    from src.config import load_config
    from src.utils import run_name
    files = [Path(f"classifier_{dataset_name()}") / "classifier.pt"]
    for model in MODELS:
        cfg = load_config(ROOT / "configs" / f"{model}.yaml", ["seed=0"])
        files.append(Path(run_name(cfg)) / "checkpoints" / f"epoch_{cfg['epochs']:03d}.pt")
    for file in files:
        print(f"$ cp runs/{file} trained_models/{file}", flush=True)
        if not DRY_RUN:
            (ROOT / "trained_models" / file).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "runs" / file, ROOT / "trained_models" / file)

def run_main(seeds):
    """Main experiment: VAE vs DCGAN with the shared default settings, one run per seed"""
    runs = ROOT / "runs"
    dataset = dataset_name()
    classifier = runs / f"classifier_{dataset}" / "classifier.pt"
    baseline = runs / f"real_baseline_{dataset}" / "metrics.json"

    # The classifier is trained once and used by every evaluation. If you split the seeds between
    # teammates, give everyone the same runs/classifier_<dataset>/ folder, or the numbers won't match
    new_classifier = not classifier.is_file()
    if new_classifier:
        step("train_classifier", "--config", "configs/classifier.yaml")
        remove(baseline.parent)  # it was computed with the previous classifier
    # Reference row: the same metrics on real images (the best a generator could score)
    if new_classifier or not baseline.is_file():
        step("real_baseline")

    for seed in seeds:
        for model in MODELS:
            config = f"configs/{model}.yaml"
            step(f"train_{model}", "--config", config, "--set", f"seed={seed}")
            step("evaluate", "--config", config, "--set", f"seed={seed}")  # metrics + figures
            step("evaluate", "--config", config, "--set", f"seed={seed}", "--all-epochs", "--quick")  # learning curve
    refresh_trained_models(seeds)

def run_ablation(seeds):
    """Ablations (run after `main`, which provides latent_dim=32 / beta=1 and the classifier):
    latent dimension 8 / 128 for both models, and beta 0.5 / 4 for the VAE"""
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
    The metric VALUES are meaningless here: this only checks that every step runs"""
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