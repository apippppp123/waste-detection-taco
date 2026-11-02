"""Plot kurva val mAP@0.5:0.95 per epoch dari log training YOLO dan Faster R-CNN.
Contoh: python src/plot_curves.py --yolo y_s_default y_s_strong --frcnn f_r50
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", nargs="*", default=[])
    ap.add_argument("--frcnn", nargs="*", default=[])
    ap.add_argument("--out", default="results/curves_map.png")
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in a.yolo:
        df = pd.read_csv(Path("runs/yolo") / n / "results.csv")
        df.columns = [c.strip() for c in df.columns]
        ax.plot(df["epoch"], df["metrics/mAP50-95(B)"], label=f"YOLO {n}")
    for n in a.frcnn:
        df = pd.read_csv(Path("runs/frcnn") / n / "results.csv")
        ax.plot(df["epoch"], df["map"], label=f"Faster R-CNN {n}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mAP@0.5:0.95 (validasi)")
    ax.set_title("Kurva validasi")
    ax.grid(alpha=0.3)
    ax.legend()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("Tersimpan:", a.out)


if __name__ == "__main__":
    main()