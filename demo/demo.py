"""VAE vs DCGAN on Fashion-MNIST"""
import sys
import tkinter as tk
from pathlib import Path
import pandas as pd
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.patches import FancyBboxPatch
from torchvision.utils import make_grid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)) 
from src.data import class_names, load_split
from src.evaluate import batched, generate
from src.metrics import laplacian_variance, slerp
from src.models import load_checkpoint, load_classifier
from src.utils import to_unit_range

CKPT_DIR = ROOT / "demo" / "checkpoints"
DATA_DIR = ROOT / "data"
RESULTS_CSV = ROOT / "results" / "all_runs.csv"
DEVICE = torch.device("cpu")
FIG_SIZE = (10, 5.2)
N_SHARP, N_DIV, N_INTERP = 1000, 5000, 8   # images for the sharpness number, the bar chart, the path
COLORS = {"VAE": "#1d4ed8", "DCGAN": "#c2410c"}
MODELS = (("VAE", "vae"), ("DCGAN", "gan"))
BG, INK, MUTED, CAPTION_BG = "#ffffff", "#111827", "#4b5563", "#f1f5f9"
TAB_ON, TAB_OFF = ("#1f2937", "#ffffff"), ("#e2e8f0", "#334155")   

CAPTIONS = [
    "Fashion-MNIST: 70,000 small black-and-white photos of clothes in 10 classes. "
    "Both models learn to make new ones from random noise, without being told the class.",
    "To keep it fair, both models are built from the same two blocks and get the same data and training time. "
    "Only the loss and the optimizer settings are different.",
    "Both models trained smoothly: the losses go down, the GAN stays balanced instead of collapsing, "
    "and the 3 seeds look almost the same. DCGAN is ahead from epoch 5 on.",
    "H1 confirmed: DCGAN images are as sharp as real ones, VAE images are blurry. "
    "When the VAE is unsure, it plays safe and draws an average, and averages look blurry.",
    "H2 not confirmed: DCGAN's mix of clothes is much closer to the real 10 % per class. "
    "The VAE draws too many of some classes and hardly any of others.",
    "H3: walking from z₀ to z₁, VAE images tend to change more smoothly. "
    "One path is just an example; the real test averages 100 paths (step 7).",
    "Averaged over all seeds, as in the report: DCGAN wins on sharpness and diversity, the VAE only on a "
    "smoother latent space. No trade-off here: blur costs the VAE both.",
]

def load_context(ckpt_dir=CKPT_DIR):
    """Load both generators, the evaluation classifier and the real test set, once at start."""
    ckpt_dir = Path(ckpt_dir)
    for path in (ckpt_dir / "vae.pt", ckpt_dir / "gan.pt", ckpt_dir / "classifier_fashion_mnist" / "classifier.pt"):
        if not path.is_file():
            raise SystemExit(f"Checkpoint not found: {path}\nRun `git pull` (the demo checkpoints are committed).")
    print("Loading the models and the test images ...")   # takes a few seconds; the window opens after
    if not (DATA_DIR / "FashionMNIST").is_dir():
        print("First run: downloading Fashion-MNIST (~30 MB) into data/ ...")
    vae, cfg_vae, _ = load_checkpoint(ckpt_dir / "vae.pt", DEVICE)
    gan, cfg_gan, _ = load_checkpoint(ckpt_dir / "gan.pt", DEVICE)
    dataset = cfg_vae["dataset"]
    clf, _ = load_classifier(dataset, ckpt_dir, DEVICE)
    real_x, real_y = load_split(dataset, "test", DATA_DIR)
    return {"vae": vae, "gan": gan, "clf": clf, "cfg_vae": cfg_vae, "cfg_gan": cfg_gan,
            "z_dim": cfg_vae["latent_dim"], "names": class_names(dataset, DATA_DIR),
            "real_x": real_x, "real_y": real_y}

def sharpness(images):
    """Mean Laplacian variance (edge strength). Blurry images -> low value."""
    return laplacian_variance(images).mean().item()

def class_shares(clf, images):
    """Fraction of the images that the classifier puts in each of the 10 classes."""
    labels = batched(clf, images, DEVICE).argmax(dim=1)
    return torch.bincount(labels, minlength=10).float() / len(labels)

@torch.no_grad()
def interpolate(model, z_dim, seed, steps=N_INTERP):
    """Decode `steps` points on the spherical path between two random latent vectors."""
    z0, z1 = torch.randn(2, 1, z_dim, generator=torch.Generator().manual_seed(seed))
    return model.decode(slerp(z0, z1, torch.linspace(0, 1, steps))[0])


def main_results(csv_path=RESULTS_CSV):
    """Mean over seeds of the main runs (z = 32, VAE beta = 1), the numbers of results/main_results.md."""
    df = pd.read_csv(csv_path)
    df = df[(df.latent_dim == 32) & ((df.model == "gan") | (df.beta == 1))]
    cols = ["fid_classifier", "precision_classifier", "recall_classifier",
            "laplacian_var", "laplacian_var_real", "interp_step_cv"]
    table = df.groupby("model")[cols].mean()
    table["n_seeds"] = df.groupby("model").seed.nunique()
    return table.rename(index={"vae": "VAE", "gan": "DCGAN"}).loc[["VAE", "DCGAN"]]


def hypotheses(df):
    """(name, claim, supported?) for H1-H3, decided from the numbers rather than typed in."""
    vae, gan = df.loc["VAE"], df.loc["DCGAN"]
    real = gan.laplacian_var_real
    return [
        ("H1", "GAN: sharper samples and lower FID", bool(
            gan.fid_classifier < vae.fid_classifier
            and abs(gan.laplacian_var - real) < abs(vae.laplacian_var - real))),
        ("H2", "VAE: better coverage (higher recall)", bool(vae.recall_classifier > gan.recall_classifier)),
        ("H3", "VAE: smoother latent space (lower interp. CV)", bool(vae.interp_step_cv < gan.interp_step_cv)),
    ]


# ----------------------------------------------------------------------------- the 6 steps
def show(ax, images, nrow=1, title="", padding=2):
    """Draw images in [-1, 1] as one grid. Nearest-neighbour scaling, so blur is the model's, not ours."""
    grid = make_grid(to_unit_range(images), nrow=nrow, padding=padding, pad_value=1.0)
    ax.imshow(grid[0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def step_dataset(fig, ctx, seed):
    axes = fig.subplots(4, 10, gridspec_kw={"wspace": 0.05, "hspace": 0.05})
    fig.subplots_adjust(left=0.05, right=0.95, top=0.86, bottom=0.02)
    for c, name in enumerate(ctx["names"]):
        examples = ctx["real_x"][ctx["real_y"] == c]
        for row in range(4):
            show(axes[row, c], examples[row:row + 1], padding=0)
        axes[0, c].set_title(name, fontsize=9)
    fig.suptitle("Fashion-MNIST: 10 classes of clothing (real test images)")

def step_setup(fig, ctx, seed):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    enc, dec = "#bbf7d0", "#fde68a"   # same colour = same block of src/models/networks.py
    rows = [(0.70, "VAE", ["image x", ("ConvEncoder\n(encoder)", enc), "μ, σ → z",
                           ("ConvDecoder\n(decoder)", dec), "reconstruction"]),
            (0.34, "DCGAN", ["noise z ~ N(0, I)", ("ConvDecoder\n(generator)", dec), "fake image",
                             ("ConvEncoder\n(discriminator)", enc), "real or fake?"])]
    xs = [0.09, 0.30, 0.50, 0.70, 0.91]
    for y, name, items in rows:
        ax.text(0.02, y + 0.13, name, color=COLORS[name], fontsize=14, weight="bold")
        for i, (x, item) in enumerate(zip(xs, items)):
            if isinstance(item, tuple):
                item, colour = item
                ax.add_patch(FancyBboxPatch((x - 0.07, y - 0.07), 0.14, 0.14, boxstyle="round,pad=0.01",
                                            fc=colour, ec="#444"))
            ax.text(x, y, item, ha="center", va="center", fontsize=10)
            if i:
                ax.annotate("", xy=(x - 0.08, y), xytext=(xs[i - 1] + 0.08, y), arrowprops={"arrowstyle": "->"})
    cfg, gan = ctx["cfg_vae"], ctx["cfg_gan"]
    ax.text(0.5, 0.03, f"Same: blocks (src/models/networks.py) · z = {cfg['latent_dim']} · {cfg['epochs']} epochs · "
            f"batch {cfg['batch_size']}\nDifferent: the loss (VAE = reconstruction + β·KL with β = {cfg['beta']:g}, "
            f"GAN = adversarial)\nand each paper's Adam settings (VAE lr {cfg['lr']:g} · GAN lr {gan['lr']:g}, "
            f"β1 {gan['adam_beta1']:g})", ha="center", fontsize=11)


def step_training(fig, ctx, seed):
    """The two training figures that `python -m src.analysis` saved in results/ (nothing is computed here)."""
    fig.suptitle("run_all.py:  classifier  →  VAE + DCGAN × 3 seeds, 30 epochs  →  evaluate  →  analysis  →  results/",
                 fontsize=11)
    for box, name in (([0.0, 0.47, 1.0, 0.45], "training_curves.png"), ([0.14, 0.0, 0.72, 0.47], "learning_curve.png")):
        ax = fig.add_axes(box)
        ax.imshow(imread(ROOT / "results" / name))
        ax.axis("off")


def step_sharpness(fig, ctx, seed):
    g = torch.Generator().manual_seed(seed)
    real = ctx["real_x"][torch.randperm(len(ctx["real_x"]), generator=g)[:N_SHARP]]
    sets = {"Real": real, **{name: generate(ctx[key], N_SHARP, ctx["z_dim"], DEVICE, seed) for name, key in MODELS}}
    real_sharp = sharpness(real)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.02, wspace=0.08)
    for ax, (name, images) in zip(fig.subplots(1, 3), sets.items()):
        s = sharpness(images)
        extra = "" if name == "Real" else f" ({s / real_sharp:.0%} of real)"
        show(ax, images[:16], nrow=4, title=f"{name}\nLaplacian var {s:.3f}{extra}")
        ax.title.set_color(COLORS.get(name, "black"))
    fig.suptitle("First 16 samples of each model, not cherry-picked  ·  Laplacian variance = edge strength "
                 "(blurry → low, closest to real is best)", fontsize=11, y=0.99)


def step_diversity(fig, ctx, seed):
    ax = fig.subplots()
    for offset, (name, key) in zip((-0.2, 0.2), MODELS):
        shares = class_shares(ctx["clf"], generate(ctx[key], N_DIV, ctx["z_dim"], DEVICE, seed))
        ax.bar([c + offset for c in range(10)], (shares * 100).tolist(), width=0.4, color=COLORS[name], label=name)
    ax.axhline(10, color="gray", ls="--", label="real data (10 % per class)")
    ax.set_xticks(range(10), ctx["names"], rotation=30)
    ax.set_ylabel("% of generated images")
    ax.set_title(f"Which class do {N_DIV:,} generated images look like? (evaluation classifier)")
    ax.legend()
    fig.subplots_adjust(bottom=0.2)


def step_latent(fig, ctx, seed):
    axes = fig.subplots(2, N_INTERP, gridspec_kw={"wspace": 0.05, "hspace": 0.1})
    fig.subplots_adjust(left=0.06, right=0.98, top=0.8, bottom=0.16)
    for row, (name, key) in enumerate(MODELS):
        frames = interpolate(ctx[key], ctx["z_dim"], seed)
        for i, ax in enumerate(axes[row]):
            show(ax, frames[i:i + 1], padding=0)
        axes[row, 0].set_ylabel(name, color=COLORS[name], fontsize=14, weight="bold")
    axes[0, 0].set_title("z₀")
    axes[0, -1].set_title("z₁")
    fig.suptitle("Walking from z₀ to z₁ in latent space (spherical interpolation)")

def step_conclusion(fig, ctx, seed):
    df = main_results()
    real = df.loc["DCGAN", "laplacian_var_real"]
    rows = [("FID, classifier features (↓)", "fid_classifier", "{:.2f}"),
            ("Precision = fidelity (↑)", "precision_classifier", "{:.2f}"),
            ("Recall = diversity (↑)", "recall_classifier", "{:.2f}"),
            (f"Sharpness, Laplacian var (real {real:.3f})", "laplacian_var", "{:.3f}"),
            ("Latent smoothness, interp. CV (↓)", "interp_step_cv", "{:.2f}")]
    ax = fig.add_axes([0.38, 0.4, 0.5, 0.5])
    ax.axis("off")
    table = ax.table(cellText=[[fmt.format(df.loc[m, col]) for m in df.index] for _, col, fmt in rows],
                     rowLabels=[label for label, _, _ in rows], colLabels=list(df.index), loc="center",
                     cellLoc="center")
    table.set_fontsize(11)
    table.scale(1, 1.6)
    fig.suptitle(f"Main results, mean over {df.n_seeds.min()} seeds (results/all_runs.csv, same as the report)")
    for i, (name, claim, ok) in enumerate(hypotheses(df)):
        fig.text(0.3, 0.27 - i * 0.08, f"{name} {'✓' if ok else '✗'}  {claim}",
                 color="#15803d" if ok else "#b91c1c", fontsize=13)

STEPS = [("Dataset", step_dataset, False), ("Setup", step_setup, False),
         ("Training", step_training, False), ("Sharpness", step_sharpness, True),
         ("Diversity", step_diversity, True), ("Latent space", step_latent, True),
         ("Conclusion", step_conclusion, False)]

class Wizard:
    """A Tk window around one matplotlib Figure. Back / Next walk through STEPS."""

    def __init__(self, root, ctx):
        self.root, self.ctx = root, ctx
        self.index, self.seeds = 0, [0] * len(STEPS)   # one seed per step; New samples adds 1
        root.title("VAE vs DCGAN demo")
        root.configure(bg=BG)
        if root.tk.call("tk", "windowingsystem") == "aqua":   # macOS: light buttons even in dark mode
            root.tk.call("::tk::unsupported::MacWindowStyle", "appearance", root, "aqua")
        header = tk.Frame(root, bg=BG)
        header.pack(fill="x", padx=16, pady=(14, 6))
        tk.Label(header, text="VAE vs DCGAN on Fashion-MNIST", font=("Helvetica", 20, "bold"),
                 bg=BG, fg=INK).pack(side="left")
        self.counter = tk.Label(header, font=("Helvetica", 13), bg=BG, fg=MUTED)
        self.counter.pack(side="right")
        tabs = tk.Frame(root, bg=BG)
        tabs.pack(fill="x", padx=16, pady=(0, 4))
        self.tabs = []
        for i, (title, _, _) in enumerate(STEPS):   # clicking a tab jumps to that step
            tab = tk.Label(tabs, text=f"{i + 1}. {title}", padx=10, pady=4)
            tab.pack(side="left", padx=(0, 6))
            tab.bind("<Button-1>", lambda event, i=i: self.show(i))
            self.tabs.append(tab)
        self.fig = Figure(figsize=FIG_SIZE, dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().configure(bg=BG, highlightthickness=0)
        self.canvas.get_tk_widget().pack()
        self.caption = tk.Label(root, wraplength=940, justify="left", anchor="w", font=("Helvetica", 15),
                                bg=CAPTION_BG, fg=INK, padx=14, pady=10)
        self.caption.pack(fill="x", padx=16, pady=(2, 10))
        buttons = tk.Frame(root, bg=BG)
        buttons.pack(fill="x", padx=16, pady=(0, 12))
        self.back_button = tk.Button(buttons, text="◀ Back", command=self.back)
        self.back_button.pack(side="left")
        self.next_button = tk.Button(buttons, text="Next ▶", command=self.next)
        self.next_button.pack(side="right")
        # Only on steps 4-6: draws new random images (next seed), to show the result is not a lucky draw.
        self.new_button = tk.Button(buttons, command=self.resample)
        root.bind("<Left>", lambda event: self.back())
        root.bind("<Right>", lambda event: self.next())
        root.bind("n", lambda event: self.resample())
        root.bind("<Escape>", lambda event: root.destroy())
        self.busy = False   # True while a step is drawing: key presses and clicks are ignored
        self.show(0)
        root.lift()   # open in front of VS Code / the terminal, with focus so the arrow keys work at once
        root.attributes("-topmost", True)
        root.after(500, root.attributes, "-topmost", False)
        root.focus_force()

    def show(self, i):
        if self.busy:
            return
        self.busy, self.index = True, i
        _, draw, can_resample = STEPS[i]
        # Labels and a placeholder first: the Diversity step takes a few seconds to draw.
        self.counter.config(text=f"Step {i + 1} of {len(STEPS)}")
        for j, tab in enumerate(self.tabs):
            bg, fg = TAB_ON if j == i else TAB_OFF
            tab.config(bg=bg, fg=fg, font=("Helvetica", 13, "bold" if j == i else "normal"))
        self.caption.config(text=CAPTIONS[i])
        self.back_button.config(state="normal" if i > 0 else "disabled")
        self.next_button.config(state="normal" if i < len(STEPS) - 1 else "disabled")
        if can_resample:
            self.new_button.config(text=f"↻ New samples (seed {self.seeds[i]})")
            self.new_button.pack()
        else:
            self.new_button.pack_forget()
        self.fig.clear()   # also resets the margins a previous step set with subplots_adjust
        self.fig.text(0.5, 0.5, "Generating images ...", ha="center", fontsize=14, color="gray")
        self.canvas.draw()
        self.root.update_idletasks()
        self.fig.clear()
        draw(self.fig, self.ctx, self.seeds[i])
        self.canvas.draw()
        self.root.update()   # handle presses made while drawing (ignored, so a double press skips nothing)
        self.busy = False

    def next(self):
        if self.index < len(STEPS) - 1:
            self.show(self.index + 1)

    def back(self):
        if self.index > 0:
            self.show(self.index - 1)

    def resample(self):
        if STEPS[self.index][2] and not self.busy:
            self.seeds[self.index] += 1
            self.show(self.index)

def main():
    ctx = load_context()
    root = tk.Tk()
    Wizard(root, ctx)
    root.mainloop()

if __name__ == "__main__":
    main()