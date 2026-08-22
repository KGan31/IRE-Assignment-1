"""
Downloads raw MIND-small and EB-NeRD-demo zips into data/raw/.
Idempotent — skips files that already exist.

Usage:
    python src/download.py
    python src/download.py --config configs/pipeline.yaml
"""

import argparse
import math
import os
import time
import zipfile
from pathlib import Path


import requests
import yaml
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
load_dotenv()


import concurrent.futures
import threading

def download_file_ebnerd(url: str, dest: Path, max_workers: int = 16) -> None:
    if dest.exists() and is_valid_zip(dest):
        print(f"[skip] {dest} already exists and is valid")
        return
    elif dest.exists():
        print(f"[corrupt/partial] removing {dest}")
        dest.unlink(missing_ok=True)


    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {dest}")

    # Check headers and total content length
    head_resp = requests.head(url, allow_redirects=True)
    total_bytes = int(head_resp.headers.get("content-length", 0))
    accept_ranges = head_resp.headers.get("accept-ranges", "") == "bytes" or head_resp.status_code == 200

    if total_bytes > 20 * 1024 * 1024 and accept_ranges:
        # Multi-threaded parallel chunk download
        chunk_size = 4 * 1024 * 1024  # 4MB chunks
        num_chunks = math.ceil(total_bytes / chunk_size)
        print(f"[parallel download] {num_chunks} chunks ({chunk_size / 1024 / 1024:.1f}MB each) with {max_workers} workers...")

        # Pre-allocate file
        with open(dest, "wb") as f:
            f.seek(total_bytes - 1)
            f.write(b"\0")

        file_lock = threading.Lock()
        pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc=dest.name)

        def download_chunk(chunk_idx: int) -> bool:
            start = chunk_idx * chunk_size
            end = min(start + chunk_size - 1, total_bytes - 1)
            headers = {"Range": f"bytes={start}-{end}"}

            for attempt in range(5):
                try:
                    r = requests.get(url, headers=headers, timeout=60)
                    if r.status_code in [200, 206]:
                        data = r.content
                        with file_lock:
                            with open(dest, "r+b") as f:
                                f.seek(start)
                                f.write(data)
                            pbar.update(len(data))
                        return True
                except Exception:
                    time.sleep(1 + attempt)
            return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(download_chunk, i) for i in range(num_chunks)]
            results = [f.result() for f in futures]

        pbar.close()
        if all(results):
            print(f"[download complete] {dest} ({total_bytes / 1024 / 1024:.1f} MB)")
            return
        print("[parallel download had failures, falling back to stream]")

    # Fallback to standard stream download
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
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

def is_valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return zf.testzip() is None
    except Exception:
        return False


def unzip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    marker = dest_dir / f".unzipped_{zip_path.stem}"
    if marker.exists():
        print(f"[skip] {zip_path.name} already unzipped into {dest_dir}")
        return
    if not zip_path.exists():
        print(f"[error] zip file {zip_path} does not exist")
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

    # --- EB-NeRD LARGE & TESTSET ---
    if dataset in ["ebnerd_large", "all_large"]:
        large_eb_cfg = cfg.get("ebnerd_large", {})
        large_eb_raw_dir = raw_dir / "ebnerd_large"
        urls = large_eb_cfg.get("urls", {})
        zips = large_eb_cfg.get("zips", {})

        # 1. Main large dataset (train + validation + articles)
        if "large" in urls:
            large_zip = large_eb_raw_dir / zips.get("large_zip", "ebnerd_large.zip")
            download_file_ebnerd(urls["large"], large_zip)
            unzip(large_zip, large_eb_raw_dir)

        # 2. Testset (test split with inview candidates)
        if "testset" in urls:
            testset_zip = large_eb_raw_dir / zips.get("testset_zip", "ebnerd_testset.zip")
            download_file_ebnerd(urls["testset"], testset_zip)
            unzip(testset_zip, large_eb_raw_dir)
            # If unzipped as ebnerd_testset subdirectory, ensure files can be found
            test_sub = large_eb_raw_dir / "ebnerd_testset"
            if test_sub.exists() and not (large_eb_raw_dir / "test").exists():
                if (test_sub / "test").exists():
                    import shutil
                    print(f"[link/move] {test_sub / 'test'} -> {large_eb_raw_dir / 'test'}")
                    try:
                        os.symlink(test_sub / "test", large_eb_raw_dir / "test", target_is_directory=True)
                    except Exception:
                        shutil.copytree(test_sub / "test", large_eb_raw_dir / "test", dirs_exist_ok=True)

        # 3. Optional large articles
        if "articles_large_only" in urls:
            art_large_zip = large_eb_raw_dir / zips.get("articles_large_only_zip", "articles_large_only.zip")
            try:
                download_file_ebnerd(urls["articles_large_only"], art_large_zip)
                unzip(art_large_zip, large_eb_raw_dir)
            except Exception as e:
                print(f"[skip optional] articles_large_only: {e}")

        # Optional pre-trained embeddings for EB-NeRD
        if download_embeddings:
            if "word2vec" in urls:
                w2v_zip = large_eb_raw_dir / "Ekstra_Bladet_word2vec.zip"
                download_file_ebnerd(urls["word2vec"], w2v_zip)
                unzip(w2v_zip, large_eb_raw_dir / "word2vec")
            if "bert" in urls:
                bert_zip = large_eb_raw_dir / "google_bert_base_multilingual_cased.zip"
                download_file_ebnerd(urls["bert"], bert_zip)
                unzip(bert_zip, large_eb_raw_dir / "bert")

    print("Done. Raw data in:", raw_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument(
        "--dataset",
        choices=["mind", "mind_small", "mind_large", "ebnerd", "ebnerd_demo", "ebnerd_large", "all", "all_large"],
        default="all",
    )
    parser.add_argument("--download_embeddings", action="store_true", help="Download EB-NeRD pretrained embeddings")
    args = parser.parse_args()
    main(args.config, args.dataset, args.download_embeddings)
