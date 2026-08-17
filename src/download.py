"""
Downloads raw MIND-small and EB-NeRD-demo zips into data/raw/.
Idempotent — skips files that already exist.

Usage:
    python src/download.py
    python src/download.py --config configs/pipeline.yaml
"""

import argparse
import os
import zipfile
from pathlib import Path

import requests
import yaml
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
load_dotenv()


def download_file_ebnerd(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {dest}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

def download_file_mind(repo_id: str, filename: str, repo_type: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {repo_id}/{filename} -> {dest}")
    hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, local_dir=dest.parent, token=os.environ.get("HF_TOKEN"))

def unzip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    marker = dest_dir / ".unzipped"
    if marker.exists():
        print(f"[skip] {dest_dir} already unzipped")
        return
    print(f"[unzip] {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    marker.touch()


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["paths"]["raw_dir"])

    # --- MIND ---
    mind_train_zip = raw_dir / "mind" / cfg["mind"]["train_zip"]
    mind_dev_zip = raw_dir / "mind" / cfg["mind"]["dev_zip"]
    download_file_mind(cfg["mind"]["repo_id"], cfg["mind"]["train_zip"], cfg["mind"]["repo_type"], mind_train_zip)
    download_file_mind(cfg["mind"]["repo_id"], cfg["mind"]["dev_zip"], cfg["mind"]["repo_type"], mind_dev_zip)
    unzip(mind_train_zip, raw_dir / "mind" / "train")
    unzip(mind_dev_zip, raw_dir / "mind" / "dev")

    # --- EB-NeRD ---
    ebnerd_zip = raw_dir / "ebnerd" / cfg["ebnerd"]["demo_zip"]
    download_file_ebnerd(cfg["ebnerd"]["urls"]["demo"], ebnerd_zip)
    unzip(ebnerd_zip, raw_dir / "ebnerd" / "demo")

    print("Done. Raw data in:", raw_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    main(args.config)
