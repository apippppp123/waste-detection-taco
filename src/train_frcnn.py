"""
Training Faster R-CNN (torchvision, pretrained COCO) pada TACO hasil prepare_taco.py.

Contoh:
  python src/train_frcnn.py --name f_r50 --arch resnet50 --epochs 30
  python src/train_frcnn.py --name f_mb  --arch mobilenet --epochs 40
  python src/train_frcnn.py --name f_r50_rfs --data data/yolo/data_rfs.yaml
"""
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights, FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from common import load_cfg, read_yolo_boxes, split_images


def build_model(arch, num_classes, imgsz=640, pretrained=True):
    """num_classes = jumlah kelas sampah (background ditambahkan otomatis)."""
    if arch == "resnet50":
        m = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
            weights_backbone=None, min_size=imgsz, max_size=imgsz)
    elif arch == "mobilenet":
        m = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None,
            weights_backbone=None, min_size=imgsz, max_size=imgsz)
    else:
        raise ValueError(arch)
    in_f = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes + 1)
    return m


class DetDataset(torch.utils.data.Dataset):
    def __init__(self, images, augment=False):
        self.images, self.augment = images, augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        p = self.images[i]
        img = Image.open(p).convert("RGB")
        W, H = img.size
        boxes, cls = read_yolo_boxes(p, W, H)
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(cls + 1, dtype=torch.int64)  # 0 = background
        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img)
                boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
            if random.random() < 0.3:
                img = TF.vflip(img)
                boxes[:, [1, 3]] = H - boxes[:, [3, 1]]
            img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
            img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))
            img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))
        keep = ((boxes[:, 2] - boxes[:, 0]) > 1) & ((boxes[:, 3] - boxes[:, 1]) > 1)
        return TF.to_tensor(img), {"boxes": boxes[keep], "labels": labels[keep]}


def collate(batch):  # harus di level modul supaya aman di Windows (multiprocessing)
    return tuple(zip(*batch))


@torch.no_grad()
def run_map(model, dataset, device, batch_size=4):
    from torchmetrics.detection import MeanAveragePrecision
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    old = (model.roi_heads.score_thresh, model.roi_heads.detections_per_img)
    model.roi_heads.score_thresh, model.roi_heads.detections_per_img = 0.001, 300
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate)
    for imgs, tgts in loader:
        imgs = [x.to(device) for x in imgs]
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(imgs)
        preds = [{"boxes": o["boxes"].float().cpu(), "scores": o["scores"].float().cpu(),
                  "labels": o["labels"].cpu()} for o in out]
        gts = [{"boxes": t["boxes"], "labels": t["labels"]} for t in tgts]
        metric.update(preds, gts)
    res = metric.compute()
    model.roi_heads.score_thresh, model.roi_heads.detections_per_img = old
    return {k: float(res[k]) for k in ("map", "map_50", "map_75")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo/data.yaml")
    ap.add_argument("--arch", choices=["resnet50", "mobilenet"], default="resnet50")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4, help="turunkan ke 2 kalau CUDA out of memory")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default="runs/frcnn")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("PERINGATAN: CUDA tidak terdeteksi, training akan sangat lambat.")
    cfg = load_cfg(a.data)
    names = cfg["names"]
    train_ds = DetDataset(split_images(cfg, "train"), augment=not a.no_aug)
    val_ds = DetDataset(split_images(cfg, "val"), augment=False)
    loader = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                        collate_fn=collate, drop_last=True)

    model = build_model(a.arch, len(names), a.imgsz, pretrained=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    warm = min(300, len(loader) - 1)

    out = Path(a.project) / a.name
    out.mkdir(parents=True, exist_ok=True)
    csv_f = open(out / "results.csv", "w", newline="")
    wr = csv.writer(csv_f)
    wr.writerow(["epoch", "train_loss", "map", "map_50", "map_75", "sec"])
    best, bad = -1.0, 0

    for ep in range(a.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for it, (imgs, tgts) in enumerate(loader):
            if ep == 0 and it < warm:  # warmup linear
                for g in opt.param_groups:
                    g["lr"] = a.lr * (0.001 + (1 - 0.001) * it / warm)
            imgs = [x.to(device) for x in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = sum(model(imgs, tgts).values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss)
        sched.step()
        m = run_map(model, val_ds, device)
        wr.writerow([ep + 1, tot / len(loader), m["map"], m["map_50"], m["map_75"], round(time.time() - t0)])
        csv_f.flush()
        print(f"epoch {ep+1}/{a.epochs} loss {tot/len(loader):.3f} mAP50-95 {m['map']:.4f} mAP50 {m['map_50']:.4f}")
        ck = {"model": model.state_dict(), "arch": a.arch, "names": names, "imgsz": a.imgsz}
        torch.save(ck, out / "last.pt")
        if m["map"] > best:
            best, bad = m["map"], 0
            torch.save(ck, out / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("Early stopping.")
                break
    csv_f.close()
    print(f"Selesai. Terbaik val mAP50-95 = {best:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
