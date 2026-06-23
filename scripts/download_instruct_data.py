import sys
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download, hf_hub_url

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

REPO_ID = "roneneldan/TinyStories-Instruct"
TRAIN_FILE = "TinyStories-Instruct-train.txt"
VAL_FILE = "TinyStories-Instruct-valid.txt"
STORY_SEP = "<|endoftext|>"


def download_train_subset(num_bytes: int) -> Path:
    url = hf_hub_url(REPO_ID, TRAIN_FILE, repo_type="dataset")
    resp = requests.get(url, headers={"Range": f"bytes=0-{num_bytes - 1}"}, timeout=60)
    resp.raise_for_status()
    text = resp.content.decode("utf-8", errors="ignore")

    last_sep = text.rfind(STORY_SEP)
    text = text[: last_sep + len(STORY_SEP)]

    out_path = DATA_DIR / "tinystories_instruct_train.txt"
    out_path.write_text(text)
    print(f"train subset: {len(text):,} chars, {text.count(STORY_SEP):,} examples -> {out_path}")
    return out_path


def download_val() -> Path:
    cached = hf_hub_download(REPO_ID, VAL_FILE, repo_type="dataset")
    out_path = DATA_DIR / "tinystories_instruct_val.txt"
    out_path.write_bytes(Path(cached).read_bytes())
    text = out_path.read_text()
    print(f"val: {len(text):,} chars, {text.count(STORY_SEP):,} examples -> {out_path}")
    return out_path


if __name__ == "__main__":
    num_bytes = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000_000
    DATA_DIR.mkdir(exist_ok=True)
    download_train_subset(num_bytes)
    download_val()
