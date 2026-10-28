# Jalankan dari root repo di PowerShell (venv aktif). Urutan = studi ablasi.
# Tiap baris bisa dijalankan terpisah; total beberapa jam, jadi jalankan satu-satu.

# --- YOLO: ablasi augmentasi + pembobotan kelas ---------------------------------
python src/train_yolo.py --name y_s_noaug   --aug none
python src/train_yolo.py --name y_s_default --aug default
python src/train_yolo.py --name y_s_strong  --aug strong
python src/train_yolo.py --name y_s_strong_rfs --aug strong --data data/yolo/data_rfs.yaml

# --- Faster R-CNN ---------------------------------------------------------------
python src/train_frcnn.py --name f_r50     --arch resnet50
python src/train_frcnn.py --name f_r50_rfs --arch resnet50 --data data/yolo/data_rfs.yaml

# --- Evaluasi di TEST set (metrik sama untuk semua model) -----------------------
python src/evaluate.py --model-type yolo  --weights runs/yolo/y_s_noaug/weights/best.pt   --name y_s_noaug
python src/evaluate.py --model-type yolo  --weights runs/yolo/y_s_default/weights/best.pt --name y_s_default
python src/evaluate.py --model-type yolo  --weights runs/yolo/y_s_strong/weights/best.pt  --name y_s_strong --viz 12
python src/evaluate.py --model-type yolo  --weights runs/yolo/y_s_strong_rfs/weights/best.pt --name y_s_strong_rfs
python src/evaluate.py --model-type frcnn --weights runs/frcnn/f_r50/best.pt     --name f_r50 --viz 12
python src/evaluate.py --model-type frcnn --weights runs/frcnn/f_r50_rfs/best.pt --name f_r50_rfs
