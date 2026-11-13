"""Gabungkan results/*/test_metrics.json jadi tabel markdown + CSV (results/summary.md, summary.csv)."""
import json
from pathlib import Path

import pandas as pd


def main():
    rows = [json.loads(p.read_text()) for p in sorted(Path("results").glob("*/test_metrics.json"))]
    if not rows:
        raise SystemExit("Belum ada results/*/test_metrics.json. Jalankan evaluate.py dulu.")
    main_cols = ["model", "type", "mAP50", "mAP50-95", "mAP_small", "mAP_medium", "mAP_large", ]
    df = pd.DataFrame([{k: r[k] for k in main_cols} for r in rows])
    df.to_csv("results/summary.csv", index=False)

    classes = list(rows[0]["AP_per_class"].keys())
    pc = pd.DataFrame({r["model"]: [r["AP_per_class"].get(c, -1) for c in classes] for r in rows}, index=classes)

    def fmt(x):
        if isinstance(x, float):
            return "-" if x < 0 else f"{x:.3f}"
        return str(x)

    def to_md(d, index=False):
        d = d.reset_index() if index else d
        head = "| " + " | ".join(map(str, d.columns)) + " |"
        sep = "|" + "---|" * len(d.columns)
        body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
        return "\n".join([head, sep] + body)

    md = "## Hasil utama (test set)\n\n" + to_md(df) + "\n\n## AP@0.5:0.95 per kelas\n\n" + to_md(pc, index=True) + "\n"
    Path("results/summary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()