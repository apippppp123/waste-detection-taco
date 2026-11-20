"""Rata-rata dan simpangan baku antar seed untuk tiap konfigurasi.
Grup = nama run tanpa akhiran _s1, _s2, dst. (run tanpa akhiran dianggap seed 0).
Jalankan dari root repo:  python src/aggregate_seeds.py
Keluaran: results/seeds_summary.md
"""
import json
import re
from pathlib import Path

import numpy as np

METRICS = [("mAP50", "mAP@0,5"), ("mAP50-95", "mAP@0,5:0,95"), ("mAP_small", "mAP kecil"),
           ("mAP_medium", "mAP sedang"), ("mAP_large", "mAP besar")]


def main():
    groups = {}
    for p in sorted(Path("results").glob("*/test_metrics.json")):
        r = json.loads(p.read_text())
        if r["model"].startswith("fps_"):
            continue
        groups.setdefault(re.sub(r"_s\d+$", "", r["model"]), []).append(r)
    lines = ["| Konfigurasi | n | " + " | ".join(n for _, n in METRICS) + " |",
             "|---|---|" + "---|" * len(METRICS)]
    for g, rs in sorted(groups.items()):
        cells = []
        for k, _ in METRICS:
            v = np.array([r[k] for r in rs], dtype=float)
            cells.append(f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v[0]:.3f}")
        lines.append(f"| {g} | {len(rs)} | " + " | ".join(cells) + " |")
    out = "\n".join(lines) + "\n"
    Path("results/seeds_summary.md").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()