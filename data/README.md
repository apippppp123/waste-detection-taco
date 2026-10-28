# Folder data (tidak di-commit)

Dataset TACO TIDAK diupload ke GitHub (aturan tugas). Cara membuatnya ulang:

1. `git clone https://github.com/pedropro/TACO.git` (di sebelah folder repo ini)
2. `cd TACO && python download.py`  (cadangan: Zenodo record 3587843)
3. Dari root repo ini: `python src/prepare_taco.py`  -> menghasilkan `data/yolo/`
