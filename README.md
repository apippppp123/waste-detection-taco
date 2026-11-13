# Deteksi Sampah pada Lingkungan Nyata dengan YOLOv8 dan Faster R-CNN

Project UAS Deep Learning. Perbandingan detektor satu tahap (YOLOv8s) dan dua tahap (Faster R-CNN ResNet-50 FPN) pada dataset TACO, dengan studi ablasi pada augmentasi dan oversampling kelas langka.

Anggota: Ahmad Rafif Har Messina (535250183), Valentino Almendo Radjawane (535220040)

## Dataset

[TACO](https://github.com/pedropro/TACO) berisi foto sampah di lingkungan nyata dengan anotasi bounding box. Dataset tidak disertakan di repositori ini.

- 1500 gambar dan 4784 bounding box, 60 kategori asli dikelompokkan menjadi 10 kelas: Plastic bag & wrapper, Cigarette, Bottle, Bottle cap, Can, Other plastic, Carton, Cup, Straw, dan Other
- Pembagian data 70/15/15 (1050 latih, 225 validasi, 225 uji), seed 42
- Sekitar 45% bounding box tergolong kecil (kurang dari 0,25% luas gambar)
- Statistik lengkap ada di `results/eda/`

## Struktur Repositori

```
src/
  prepare_taco.py            konversi COCO ke YOLO, pengelompokan kelas, split
  download_taco_parallel.py  unduh gambar TACO secara paralel
  eda.py                     statistik dan grafik dataset
  train_yolo.py              pelatihan YOLOv8
  train_frcnn.py             pelatihan Faster R-CNN
  evaluate.py                evaluasi seragam untuk kedua model
  plot_curves.py             kurva validasi
  make_table.py              tabel hasil gabungan
  common.py                  fungsi bersama
scripts/                     skrip eksperimen
results/                     metrik, log training, kurva, contoh deteksi
data/                        dataset hasil olahan (tidak di-commit)
```

## Instalasi

Lingkungan yang digunakan: Windows, Python 3.14, PyTorch 2.15.0 (CUDA 13.2), Ultralytics 8.4.198, GPU RTX 3050 Laptop 6 GB.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

## Persiapan Data

```powershell
git clone https://github.com/pedropro/TACO.git
python src/download_taco_parallel.py --taco-data <PATH_TACO>\data --workers 8
python src/prepare_taco.py --taco-data <PATH_TACO>\data
python src/eda.py
```

`prepare_taco.py` mengelompokkan kategori menjadi 9 super-category teratas ditambah Other, mengoreksi orientasi EXIF, memperkecil gambar hingga sisi terpanjang 1280 piksel, dan membuat `data_rfs.yaml` untuk oversampling gambar yang memuat kelas langka (repeat-factor sampling).

## Pelatihan

YOLOv8s: bobot awal COCO, 50 epoch, resolusi 640, batch 8, mixed precision, optimizer otomatis (AdamW).
Faster R-CNN R50-FPN: bobot awal COCO, 30 epoch, resolusi 640, batch 4, SGD (lr 0,005) dengan jadwal cosine, mixed precision.

```powershell
python src/train_yolo.py --name y_s_noaug --aug none --epochs 50
python src/train_yolo.py --name y_s_default --aug default --epochs 50
python src/train_yolo.py --name y_s_strong --aug strong --epochs 50
python src/train_yolo.py --name y_s_strong_rfs --aug strong --epochs 50 --data data/yolo/data_rfs.yaml
python src/train_frcnn.py --name f_r50 --epochs 30
python src/train_frcnn.py --name f_r50_rfs --epochs 30 --data data/yolo/data_rfs.yaml
```

## Evaluasi

Seluruh model dievaluasi pada data uji dengan kode yang sama (mAP dari torchmetrics, ambang skor 0,001, NMS IoU 0,5, maksimum 300 deteksi). FPS diukur pada RTX 3050 Laptop dengan batch 1, dari gambar masukan hingga hasil deteksi, termasuk pra dan pasca-proses.

```powershell
python src/evaluate.py --model-type yolo --weights runs/yolo/y_s_default/weights/best.pt --name y_s_default --viz 12
python src/evaluate.py --model-type frcnn --weights runs/frcnn/f_r50/best.pt --name f_r50 --viz 12
python src/make_table.py
```

## Hasil

Hasil pada data uji (225 gambar). YOLOv8s default dan augmentasi kuat dijalankan dengan tiga seed (rata-rata ± simpangan baku); konfigurasi lain satu run (seed 0).

| Model | Konfigurasi | Run | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| YOLOv8s | tanpa augmentasi | 1 | 0.214 | 0.168 |
| YOLOv8s | augmentasi default | 3 | 0.272 ± 0.013 | 0.208 ± 0.007 |
| YOLOv8s | augmentasi kuat | 3 | 0.327 ± 0.011 | 0.206 ± 0.008 |
| YOLOv8s | augmentasi kuat + RFS | 1 | 0.311 | 0.189 |
| Faster R-CNN R50-FPN | default | 1 | 0.326 | 0.234 |
| Faster R-CNN R50-FPN | default + RFS | 1 | 0.310 | 0.213 |

Kecepatan inferensi (RTX 3050 Laptop 6 GB, batch 1, median latensi per gambar termasuk pra dan pasca-proses, half precision diminta, tiga pengukuran pada kondisi bersih):

| Model | FPS |
|---|---|
| YOLOv8s | 46.1 ± 4.6 (kisaran 40.9 sampai 49.6) |
| Faster R-CNN R50-FPN | 21.0 ± 0.1 |

Tabel lengkap dan AP per kelas ada di `results/summary.md`.

Temuan:
- Konfigurasi tanpa augmentasi memberikan hasil terendah.
- Augmentasi kuat menaikkan mAP@0.5 pada ketiga seed, sedangkan mAP@0.5:0.95 setara dengan augmentasi default.
- Pada IoU 0,5, YOLOv8s dengan augmentasi kuat setara dengan Faster R-CNN; pada mAP@0.5:0.95 Faster R-CNN lebih tinggi.
- YOLOv8s sekitar dua kali lebih cepat daripada Faster R-CNN.
- Objek kecil tetap menjadi kelemahan semua konfigurasi (mAP small 0,05 hingga 0,10).
- Oversampling kelas langka (RFS) tidak meningkatkan metrik keseluruhan pada satu run per konfigurasi.
- AP per kelas untuk kelas dengan sedikit sampel uji bervariasi besar antar seed (misalnya Straw 0,031 hingga 0,107 pada konfigurasi yang sama), sehingga tidak dibandingkan antar konfigurasi.

## Kontribusi

- Ahmad Rafif Har Messina (535250183): persiapan data dan EDA, pelatihan dan evaluasi seluruh model, pengelolaan repositori
- Valentino Almendo Radjawane (535220040): studi literatur, analisis kesalahan deteksi, penulisan dan penyuntingan laporan, pengelolaan referensi

## Referensi

1. P. F. Proença dan P. Simões, "TACO: Trash Annotations in Context for Litter Detection," arXiv:2003.06975, 2020.
2. J. Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection," CVPR, 2016.
3. S. Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks," IEEE TPAMI, 2017.