"""Download gambar TACO secara PARALEL (pengganti download.py yang berurutan dan lambat).
Aman dijalankan berulang: file yang sudah ada dilewati."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm


def work(im, root, retries, timeout):
    dst = root / im["file_name"]
    if dst.exists() and dst.stat().st_size > 0:
        return "skip", im["file_name"]
    url = im.get("flickr_url")
    if not url:
        return "fail", f"{im['file_name']} tidak punya flickr_url"
    dst.parent.mkdir(parents=True, exist_ok=True)
    last = ""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            Image.open(BytesIO(r.content)).verify()
            tmp = dst.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(dst)
            return "ok", im["file_name"]
        except Exception as e:
            last = repr(e)
            time.sleep(1.5 * (attempt + 1))
    return "fail", f"{im['file_name']} {last}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taco-data", required=True)
    ap.add_argument("--ann", default="annotations.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=30)
    a = ap.parse_args()

    root = Path(a.taco_data)
    coco = json.loads((root / a.ann).read_text(encoding="utf-8"))
    images = coco["images"]
    print(f"{len(images)} gambar di anotasi, {a.workers} unduhan paralel")

    stat = {"ok": 0, "skip": 0, "fail": 0}
    failed = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, im, root, a.retries, a.timeout) for im in images]
        for f in tqdm(as_completed(futs), total=len(futs), desc="download"):
            s, msg = f.result()
            stat[s] += 1
            if s == "fail":
                failed.append(msg)
    print(f"selesai: baru diunduh {stat['ok']}, sudah ada {stat['skip']}, gagal {stat['fail']}")
    if failed:
        (root / "download_failed.txt").write_text("\n".join(failed), encoding="utf-8")
        print("Daftar gagal: download_failed.txt. Jalankan ulang perintah yang sama untuk mencoba lagi.")


if __name__ == "__main__":
    main()
