"""
Evaluasi YOLO dan Faster R-CNN dengan metrik yang SAMA (torchmetrics mAP) + kecepatan inferensi.

Contoh:
  python src/evaluate.py --model-type yolo  --weights runs/yolo/y_s_strong/weights/best.pt --name y_s_strong
  python src/evaluate.py --model-type frcnn --weights runs/frcnn/f_r50/best.pt --name f_r50 --viz 12
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from tqdm import tqdm

from common import load_cfg, read_yolo_boxes, split_images

NMS_IOU, MAX_DET = 0.5, 300


def make_yolo(weights, imgsz):
    from ultralytics import YOLO
    m = YOLO(weights)

    def f(img):
        arr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])  # RGB -> BGR
        r = m.predict(arr, imgsz=imgsz, conf=0.001, iou=NMS_IOU, max_det=MAX_DET,
                      device=0, half=True, verbose=False)[0]
        return r.boxes.xyxy.cpu(), r.boxes.conf.cpu(), r.boxes.cls.cpu().long()
    return f


def make_frcnn(weights, device):
    from train_frcnn import build_model
    ck = torch.load(weights, map_location="cpu")
    m = build_model(ck["arch"], len(ck["names"]), ck["imgsz"], pretrained=False)
    m.load_state_dict(ck["model"])
    m.to(device).eval()
    m.roi_heads.score_thresh, m.roi_heads.nms_thresh, m.roi_heads.detections_per_img = 0.001, NMS_IOU, MAX_DET

    def f(img):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            o = m([TF.to_tensor(img).to(device)])[0]
        return o["boxes"].float().cpu(), o["scores"].float().cpu(), (o["labels"] - 1).cpu()
    return f


def draw(img, boxes, scores, labels, names, thr=0.3):
    d = ImageDraw.Draw(img)
    for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        if s < thr:
            continue
        d.rectangle(b, outline=(255, 40, 40), width=3)
        d.text((b[0] + 3, b[1] + 3), f"{names[l]} {s:.2f}", fill=(255, 255, 0))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", choices=["yolo", "frcnn"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/yolo/data.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640, help="hanya untuk YOLO")
    ap.add_argument("--name", required=True)
    ap.add_argument("--viz", type=int, default=0, help="simpan N contoh gambar hasil deteksi")
    a = ap.parse_args()

    from torchmetrics.detection import MeanAveragePrecision
    device = torch.device("cuda")
    cfg = load_cfg(a.data)
    names = cfg["names"]
    imgs = split_images(cfg, a.split)
    predict = make_yolo(a.weights, a.imgsz) if a.model_type == "yolo" else make_frcnn(a.weights, device)

    out = Path("results") / a.name
    (out / "viz").mkdir(parents=True, exist_ok=True)
    viz_idx = set(np.linspace(0, len(imgs) - 1, a.viz, dtype=int).tolist()) if a.viz else set()

    for p in imgs[:5]:  # warm-up (tidak dihitung)
        predict(Image.open(p).convert("RGB"))

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    times = []
    for k, p in enumerate(tqdm(imgs, desc=f"eval {a.split}")):
        img = Image.open(p).convert("RGB")
        W, H = img.size
        gb, gc = read_yolo_boxes(p, W, H)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        b, s, l = predict(img)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
        metric.update([{"boxes": b, "scores": s, "labels": l}],
                      [{"boxes": torch.as_tensor(gb), "labels": torch.as_tensor(gc)}])
        if k in viz_idx:
            draw(img, b, s, l, names).save(out / "viz" / f"{p.stem}.jpg", quality=90)

    r = metric.compute()
    ms = float(np.mean(times) * 1000)
    per_class = {}
    for c, ap_c in zip(r["classes"].tolist(), r["map_per_class"].tolist()):
        per_class[names[c]] = ap_c  # -1 berarti kelas tidak punya ground truth di split ini
    res = {"model": a.name, "type": a.model_type, "split": a.split, "n_images": len(imgs),
           "mAP50-95": float(r["map"]), "mAP50": float(r["map_50"]), "mAP75": float(r["map_75"]),
           "mAP_small": float(r["map_small"]), "mAP_medium": float(r["map_medium"]),
           "mAP_large": float(r["map_large"]), "mAR100": float(r["mar_100"]),
           "ms_per_image": round(ms, 2), "FPS": round(1000 / ms, 2), "AP_per_class": per_class}
    (out / f"{a.split}_metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "AP_per_class"}, indent=2))
    print("AP per kelas:", json.dumps(per_class, indent=2))
    print(f"Tersimpan di {out}/")


if __name__ == "__main__":
    main()
