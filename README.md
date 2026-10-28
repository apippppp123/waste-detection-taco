# Deteksi Sampah pada Lingkungan Nyata (TACO) — YOLO vs Faster R-CNN

Project UAS Deep Learning. Dataset: TACO (publik). Dataset TIDAK ada di repo ini; lihat `data/README.md`.

## 1. Setup (Windows + RTX 3050 6 GB)
```powershell
python -m venv .venv
.venv\Scripts\activate
# PyTorch dengan CUDA: ambil perintah dari https://pytorch.org/get-started/locally/ (pilih Pip + CUDA)
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # harus True
```

## 2. Data
```powershell
cd ..                       # keluar dari repo, taruh TACO di sebelahnya
git clone https://github.com/pedropro/TACO.git
cd TACO
pip install requests
python download.py          # gambar diunduh dari Flickr ke TACO/data/batch_*
cd ..\waste-detection-taco
python src/prepare_taco.py --taco-data ..\TACO\data
python src/eda.py
```
`prepare_taco.py` mengelompokkan kelas (top-9 super-category + Other), membetulkan orientasi EXIF,
mengecilkan gambar (sisi terpanjang 1280), split 70/15/15 (seed 42), dan membuat `data_rfs.yaml`
(oversampling gambar berisi kelas langka / repeat-factor sampling).

## 3. Eksperimen
Lihat `scripts/run_experiments.ps1` (ablasi augmentasi + RFS untuk YOLO, plus Faster R-CNN),
lalu evaluasi semua model di test set dengan `src/evaluate.py` (metrik yang sama untuk semua model).

## 4. Hasil
Isi tabel di bawah dari `results/<nama>/test_metrics.json`.

| Model | Konfigurasi | mAP@0.5 | mAP@0.5:0.95 | FPS |
|---|---|---|---|---|
| YOLOv8s | tanpa augmentasi | | | |
| YOLOv8s | augmentasi default | | | |
| YOLOv8s | augmentasi kuat | | | |
| YOLOv8s | kuat + RFS | | | |
| Faster R-CNN R50-FPN | default | | | |
| Faster R-CNN R50-FPN | + RFS | | | |

## 5. Kontribusi
Ahmad Rafif Har Messina (535250183):
Valentino Almendo Radjawane (535220040):

## Referensi utama
TACO (arXiv:2003.06975), YOLO (arXiv:1506.02640), Faster R-CNN (arXiv:1506.01497),
Focal Loss (arXiv:1708.02002), SAHI (arXiv:2202.06934).
