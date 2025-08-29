"""Simple EDA: plot class distribution and a few sample epochs to verify signal shape."""
import argparse, os, json, numpy as np
import matplotlib.pyplot as plt

STAGE_NAMES = ["W","N1","N2","N3","REM"]

def stage_hist(y):
    counts = np.bincount(y, minlength=5)
    total = counts.sum()
    return counts, counts/total

def plot_stage_distribution(y_all, outdir):
    counts, frac = stage_hist(y_all)
    os.makedirs(outdir, exist_ok=True)

    # Absolute counts
    plt.figure()
    xs = np.arange(5)
    plt.bar(xs, counts)
    plt.xticks(xs, STAGE_NAMES)
    plt.ylabel("Epoch count")
    plt.title("Stage distribution (all splits)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "stage_distribution_counts.png"))
    plt.close()

    # Proportions
    plt.figure()
    plt.bar(xs, frac)
    plt.xticks(xs, STAGE_NAMES)
    plt.ylabel("Proportion")
    plt.title("Stage distribution (proportion)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "stage_distribution_fraction.png"))
    plt.close()

def plot_example_epochs(X, outdir, n=5):
    """Overlay channels for a few random epochs (visual sanity check)."""
    os.makedirs(outdir, exist_ok=True)
    C, T = X.shape[1], X.shape[2]
    for i in range(min(n, len(X))):
        plt.figure()
        for c in range(C):
            plt.plot(X[i, c], label=f"Ch{c+1}")
        plt.title(f"Example epoch #{i}")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude (z)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"example_epoch_{i}.png"))
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/processed_npy", help="Processed root with manifest.json")
    args = ap.parse_args()

    # Load all shards across splits
    mani = json.load(open(os.path.join(args.root, "manifest.json")))
    Xs, ys = [], []
    for split in ["train","val","test"]:
        for rec in mani["splits"][split]:
            npz = np.load(os.path.join(args.root, rec["path"]))
            Xs.append(npz["X"])
            ys.append(npz["y"])
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)

    plot_stage_distribution(y, outdir="figs")
    plot_example_epochs(X, outdir="figs", n=5)
    print("EDA figures saved in figs/.")

if __name__ == "__main__":
    main()
