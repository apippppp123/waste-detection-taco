"""
Training YOLO (Ultralytics) pada TACO hasil prepare_taco.py.

Contoh:
  python src/train_yolo.py --name y_s_default --aug default
  python src/train_yolo.py --name y_s_strong  --aug strong
  python src/train_yolo.py --name y_s_strong_rfs --aug strong --data data/yolo/data_rfs.yaml
"""
import argparse
from pathlib import Path

AUG = {
    # semua augmentasi dimatikan (pembanding ablation)
    "none": dict(mosaic=0.0, mixup=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                 fliplr=0.0, flipud=0.0, scale=0.0, translate=0.0, degrees=0.0),
    # pengaturan bawaan Ultralytics (mosaic aktif)
    "default": dict(),
    # augmentasi diperkuat
    "strong": dict(mosaic=1.0, mixup=0.15, scale=0.7, degrees=10.0, flipud=0.3, fliplr=0.5,
                   hsv_s=0.8, hsv_v=0.5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo/data.yaml")
    ap.add_argument("--model", default="yolov8s.pt", help="yolov8n.pt / yolov8s.pt / yolo11s.pt, dst")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8, help="turunkan ke 4 kalau CUDA out of memory")
    ap.add_argument("--workers", type=int, default=4, help="set 0 kalau di Windows macet/error dataloader")
    ap.add_argument("--aug", choices=list(AUG), default="default")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default="runs/yolo")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    model.train(
        data=str(Path(a.data).resolve()), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        workers=a.workers, device=0, patience=a.patience, seed=a.seed,
        project=str(Path(a.project).resolve()), name=a.name, **AUG[a.aug],
    )
    print("Selesai. Bobot terbaik: ", Path(a.project).resolve() / a.name / "weights" / "best.pt")


if __name__ == "__main__":
    main()
