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

def download_file_mind(repo_id: str, filename: str, repo_type: str, dest: Path, fallback_url: str = None) -> None:
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {repo_id}/{filename} -> {dest}")
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=dest.parent,
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as e:
        if fallback_url:
            print(f"[HF download failed: {e}. Trying fallback url: {fallback_url}]")
            download_file_ebnerd(fallback_url, dest)
        else:
            raise e

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


def main(config_path: str, dataset: str = "all", download_embeddings: bool = False) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["paths"]["raw_dir"])

    # --- MIND SMALL ---
    if dataset in ["mind", "mind_small", "all"]:
        mind_train_zip = raw_dir / "mind" / cfg["mind"]["train_zip"]
        mind_dev_zip = raw_dir / "mind" / cfg["mind"]["dev_zip"]
        download_file_mind(cfg["mind"]["repo_id"], cfg["mind"]["train_zip"], cfg["mind"]["repo_type"], mind_train_zip)
        download_file_mind(cfg["mind"]["repo_id"], cfg["mind"]["dev_zip"], cfg["mind"]["repo_type"], mind_dev_zip)
        unzip(mind_train_zip, raw_dir / "mind" / "train")
        unzip(mind_dev_zip, raw_dir / "mind" / "dev")

    # --- MIND LARGE ---
    if dataset in ["mind_large", "all_large"]:
        large_cfg = cfg.get("mind_large", {})
        large_raw_dir = raw_dir / "mind_large"
        urls = large_cfg.get("urls", {})
        
        for split_key, zip_key in [("train", "train_zip"), ("dev", "dev_zip"), ("test", "test_zip")]:
            zip_filename = large_cfg.get(zip_key)
            if zip_filename:
                zip_path = large_raw_dir / zip_filename
                fallback_url = urls.get(split_key)
                download_file_mind(
                    large_cfg["repo_id"],
                    zip_filename,
                    large_cfg["repo_type"],
                    zip_path,
                    fallback_url=fallback_url,
                )
                unzip(zip_path, large_raw_dir / split_key)

    # --- EB-NeRD ---
    if dataset in ["ebnerd", "all"]:
        ebnerd_zip = raw_dir / "ebnerd" / cfg["ebnerd"]["demo_zip"]
        download_file_ebnerd(cfg["ebnerd"]["urls"]["demo"], ebnerd_zip)
        unzip(ebnerd_zip, raw_dir / "ebnerd" / "demo")

        # Optional pre-trained embeddings for EB-NeRD
        if download_embeddings:
            if "word2vec" in cfg["ebnerd"]["urls"]:
                w2v_zip = raw_dir / "ebnerd" / "Ekstra_Bladet_word2vec.zip"
                download_file_ebnerd(cfg["ebnerd"]["urls"]["word2vec"], w2v_zip)
                unzip(w2v_zip, raw_dir / "ebnerd" / "word2vec")
            if "bert" in cfg["ebnerd"]["urls"]:
                bert_zip = raw_dir / "ebnerd" / "google_bert_base_multilingual_cased.zip"
                download_file_ebnerd(cfg["ebnerd"]["urls"]["bert"], bert_zip)
                unzip(bert_zip, raw_dir / "ebnerd" / "bert")

    print("Done. Raw data in:", raw_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--dataset", choices=["mind", "mind_small", "mind_large", "ebnerd", "all", "all_large"], default="all")
    parser.add_argument("--download_embeddings", action="store_true", help="Download EB-NeRD pretrained embeddings")
    args = parser.parse_args()
    main(args.config, args.dataset, args.download_embeddings)
