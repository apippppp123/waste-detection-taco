"""Statistik dataset (untuk bagian 'Details on the dataset' di progress report).
Output: results/eda/class_counts.csv, class_counts.png, bbox_size_hist.png, summary.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import load_cfg, label_path

SPLITS = ["train", "val", "test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo/data.yaml")
    ap.add_argument("--out", default="results/eda")
    a = ap.parse_args()
    cfg = load_cfg(a.data)
    names = cfg["names"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    counts = {s: Counter() for s in SPLITS}
    n_images = {}
    areas = []  # luas relatif bbox terhadap gambar
    for s in SPLITS:
        d = cfg["root"] / s / "images"
        imgs = sorted(d.glob("*.jpg"))
        n_images[s] = len(imgs)
        for p in imgs:
            lp = label_path(p)
            if not lp.exists():
                continue
            for line in lp.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                c = int(parts[0]); w, h = float(parts[3]), float(parts[4])
                counts[s][c] += 1
                areas.append(w * h)

    import pandas as pd
    df = pd.DataFrame({"class": names})
    for s in SPLITS:
        df[s] = [counts[s][i] for i in range(len(names))]
    df["total"] = df[SPLITS].sum(axis=1)
    df.to_csv(out / "class_counts.csv", index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(names)); wd = 0.27
    for k, s in enumerate(SPLITS):
        ax.bar(x + (k - 1) * wd, df[s], wd, label=s)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("jumlah bbox"); ax.set_title("Distribusi kelas per split"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "class_counts.png", dpi=150); plt.close(fig)

    areas = np.array(areas)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.log10(areas.clip(1e-6)), bins=40)
    ax.set_xlabel("log10(luas bbox / luas gambar)"); ax.set_ylabel("jumlah bbox")
    ax.set_title("Ukuran relatif bbox")
    fig.tight_layout(); fig.savefig(out / "bbox_size_hist.png", dpi=150); plt.close(fig)

    small = float((areas < 0.0025).mean())  # kira-kira < 32x32 px pada 640x640
    pos = df["train"][df["train"] > 0]
    summary = {"images": n_images, "boxes_total": int(len(areas)),
               "median_rel_area": float(np.median(areas)),
               "small_ratio_(<0.25%_luas_gambar)": small,
               "imbalance_ratio_(max/min_kelas_train)": float(df["train"].max() / max(1, pos.min()))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
