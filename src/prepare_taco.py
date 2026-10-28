"""
Konversi TACO (format COCO) -> format YOLO, dengan:
  - pengelompokan kelas (top-k super-category + "Other")
  - koreksi orientasi EXIF (gambar Flickr sering terotasi)
  - split train/val/test per gambar (seed tetap, mengusahakan kelas langka tersebar)
  - (opsional) train_rfs.txt untuk repeat-factor sampling (pembobotan kelas langka)

Contoh:
  python src/prepare_taco.py --taco-data ../TACO/data
"""
import argparse
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageOps


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taco-data", default="../TACO/data", help="folder TACO/data (berisi annotations.json + batch_*/)")
    ap.add_argument("--ann", default="annotations.json", help="annotations.json atau annotations_unofficial.json")
    ap.add_argument("--out", default="data/yolo")
    ap.add_argument("--group", choices=["topk", "none"], default="topk",
                    help="topk = top-k super-category + Other; none = pakai kategori asli")
    ap.add_argument("--top-k", type=int, default=9)
    ap.add_argument("--merge-to-other", nargs="*", default=["Unlabeled litter"],
                    help="super-category yang selalu dilebur ke Other")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-side", type=int, default=1280, help="perkecil gambar jika sisi terpanjang > nilai ini (0 = jangan)")
    ap.add_argument("--min-box", type=float, default=2.0, help="buang bbox dengan lebar/tinggi < nilai ini (piksel)")
    ap.add_argument("--rfs-threshold", type=float, default=0.3,
                    help="threshold repeat-factor sampling; 0 = tidak membuat train_rfs.txt")
    return ap.parse_args()


def resolve_orientation(path, W, H):
    """Tentukan apakah anotasi mengikuti gambar yang sudah diputar EXIF ('transpose') atau mentah ('raw')."""
    with Image.open(path) as im:
        raw = im.size
        orient = im.getexif().get(274, 1)
    tr = (raw[1], raw[0]) if orient in (5, 6, 7, 8) else raw
    if W is None or H is None:
        return "transpose"
    if (W, H) == tr:
        return "transpose"
    if (W, H) == raw:
        return "raw"
    return None


def stratified_split(img_classes, val_r, test_r, seed):
    """Split per gambar. Kelas paling langka diproses duluan supaya muncul di val/test juga."""
    rng = random.Random(seed)
    cnt = Counter(c for cs in img_classes.values() for c in set(cs))
    assigned = {}
    for c in sorted(cnt, key=lambda k: (cnt[k], k)):
        imgs = sorted(i for i, cs in img_classes.items() if c in cs and i not in assigned)
        rng.shuffle(imgs)
        n = len(imgs)
        n_val = max(1, round(n * val_r)) if n >= 3 else 0
        n_test = max(1, round(n * test_r)) if n >= 3 else 0
        for k, i in enumerate(imgs):
            assigned[i] = "val" if k < n_val else ("test" if k < n_val + n_test else "train")
    for i in sorted(img_classes):
        assigned.setdefault(i, "train")
    return assigned


def main():
    a = parse_args()
    data_dir = Path(a.taco_data)
    ann_path = data_dir / a.ann
    if not ann_path.exists():
        raise SystemExit(f"Anotasi tidak ketemu: {ann_path}\nCek --taco-data (harus folder TACO/data).")
    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    cats = {c["id"]: c for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for an in coco["annotations"]:
        anns_by_img[an["image_id"]].append(an)
    print(f"Anotasi: {len(images)} gambar, {len(coco['annotations'])} bbox, {len(cats)} kategori")

    # 1) cek file gambar + orientasi
    mode, missing, mismatch = {}, [], []
    for iid, im in images.items():
        p = data_dir / im["file_name"]
        if not p.exists():
            missing.append(im["file_name"])
            continue
        m = resolve_orientation(p, im.get("width"), im.get("height"))
        if m is None:
            mismatch.append(im["file_name"])
            continue
        mode[iid] = m
    print(f"Gambar ada: {len(mode)} | belum terdownload: {len(missing)} | ukuran tidak cocok (dilewati): {len(mismatch)}")
    if not mode:
        raise SystemExit("Tidak ada gambar yang bisa dipakai. Sudah jalankan download.py di folder TACO?")

    # 2) mapping kelas
    def sup(cid):
        c = cats[cid]
        return c.get("supercategory") or c["name"]

    inst = Counter()
    for iid in mode:
        for an in anns_by_img[iid]:
            inst[sup(an["category_id"])] += 1
    if a.group == "none":
        ids = sorted(cats)
        names = [cats[i]["name"] for i in ids]
        cat2cls = {cid: k for k, cid in enumerate(ids)}
    else:
        cand = [s for s, _ in inst.most_common() if s not in a.merge_to_other]
        keep = cand[: a.top_k]
        names = keep + ["Other"]
        cat2cls = {cid: (keep.index(sup(cid)) if sup(cid) in keep else len(keep)) for cid in cats}
    print("Kelas akhir:", names)

    # 3) kumpulkan bbox valid per gambar
    img_boxes = {}
    for iid in mode:
        im = images[iid]
        W, H = im.get("width"), im.get("height")
        rows = []
        for an in anns_by_img[iid]:
            x, y, w, h = an["bbox"]
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = (min(W, x + w) if W else x + w), (min(H, y + h) if H else y + h)
            if (x2 - x1) < a.min_box or (y2 - y1) < a.min_box:
                continue
            rows.append((cat2cls[an["category_id"]], x1, y1, x2, y2))
        if rows:
            img_boxes[iid] = rows
    print(f"Gambar dengan bbox valid: {len(img_boxes)}")

    # 4) split
    img_classes = {iid: [r[0] for r in rows] for iid, rows in img_boxes.items()}
    split_of = stratified_split(img_classes, a.val, a.test, a.seed)

    # 5) tulis dataset
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    for s in ("train", "val", "test"):
        (out / s / "images").mkdir(parents=True)
        (out / s / "labels").mkdir(parents=True)
    per_split = {s: Counter() for s in ("train", "val", "test")}
    n_img = Counter()
    written = {s: [] for s in ("train", "val", "test")}
    for iid, rows in img_boxes.items():
        im_meta = images[iid]
        s = split_of[iid]
        src = data_dir / im_meta["file_name"]
        stem = Path(im_meta["file_name"]).with_suffix("").as_posix().replace("/", "_")
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im) if mode[iid] == "transpose" else im.copy()
            im = im.convert("RGB")
        W0, H0 = im_meta.get("width") or im.size[0], im_meta.get("height") or im.size[1]
        if a.max_side and max(im.size) > a.max_side:
            im.thumbnail((a.max_side, a.max_side), Image.LANCZOS)
        dst_img = out / s / "images" / f"{stem}.jpg"
        im.save(dst_img, quality=95)
        lines = []
        for c, x1, y1, x2, y2 in rows:
            cx, cy = (x1 + x2) / 2 / W0, (y1 + y2) / 2 / H0
            bw, bh = (x2 - x1) / W0, (y2 - y1) / H0
            lines.append(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            per_split[s][names[c]] += 1
        (out / s / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        n_img[s] += 1
        written[s].append((dst_img.resolve(), set(r[0] for r in rows)))

    # 6) data.yaml
    root = out.resolve().as_posix()
    base = {"path": root, "train": "train/images", "val": "val/images", "test": "test/images",
            "names": {i: n for i, n in enumerate(names)}}
    (out / "data.yaml").write_text(yaml.safe_dump(base, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # 7) repeat-factor sampling (opsional)
    if a.rfs_threshold > 0:
        rng = random.Random(a.seed)
        tr = written["train"]
        f = Counter()
        for _, cs in tr:
            for c in cs:
                f[c] += 1
        r_c = {c: max(1.0, math.sqrt(a.rfs_threshold / (f[c] / len(tr)))) for c in f}
        lines = []
        for p, cs in tr:
            r = max(r_c[c] for c in cs)
            rep = int(math.floor(r)) + (1 if rng.random() < r - math.floor(r) else 0)
            lines += [p.as_posix()] * max(1, rep)
        (out / "train_rfs.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        rfs = dict(base, train="train_rfs.txt")
        (out / "data_rfs.yaml").write_text(yaml.safe_dump(rfs, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"RFS: train {len(tr)} gambar -> {len(lines)} entri (data_rfs.yaml)")

    # 8) ringkasan
    stats = {"images": dict(n_img), "instances": {s: dict(per_split[s]) for s in per_split},
             "missing": missing, "mismatch": mismatch}
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    (out / "classes.json").write_text(json.dumps(
        {cats[cid]["name"]: names[k] for cid, k in cat2cls.items()}, indent=2, ensure_ascii=False))
    print("\nGambar per split:", dict(n_img))
    print(f"{'kelas':28s}{'train':>8s}{'val':>8s}{'test':>8s}")
    for n in names:
        print(f"{n:28s}{per_split['train'][n]:8d}{per_split['val'][n]:8d}{per_split['test'][n]:8d}")
    print(f"\nSelesai -> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
