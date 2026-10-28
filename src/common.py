"""Helper bersama: baca data.yaml, daftar gambar, dan label YOLO."""
from pathlib import Path
import numpy as np
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_cfg(yaml_path):
    cfg = yaml.safe_load(open(yaml_path, encoding="utf-8"))
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    cfg["names"] = names
    cfg["root"] = Path(cfg["path"])
    return cfg


def split_images(cfg, split):
    """Mengembalikan list path gambar. Entry di yaml boleh folder atau file .txt (daftar path)."""
    p = cfg["root"] / cfg[split]
    if p.is_file():
        lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        return [Path(l) for l in lines]
    return sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXT)


def label_path(img_path):
    img_path = Path(img_path)
    return img_path.parent.parent / "labels" / (img_path.stem + ".txt")


def read_yolo_boxes(img_path, W, H):
    """Baca label YOLO -> (boxes Nx4 xyxy piksel, cls N)."""
    lp = label_path(img_path)
    boxes, cls = [], []
    if lp.exists():
        for line in lp.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            c = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            boxes.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
            cls.append(c)
    return np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(cls, dtype=np.int64)
