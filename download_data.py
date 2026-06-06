"""
一键下载 MovieLens 数据集（跨平台：Windows / Linux / macOS）

用法:
    python download_data.py              # 下载 ml-1m + ml-25m
    python download_data.py --dataset ml-1m   # 只下载 ml-1m
    python download_data.py --dataset ml-25m  # 只下载 ml-25m
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

BASE_URL = "https://files.grouplens.org/datasets/movielens"
DATA_DIR = Path(__file__).resolve().parent / "data"

DATASETS = {
    "ml-1m": {
        "url": f"{BASE_URL}/ml-1m.zip",
        "marker": ("ratings.dat",),  # 存在任一即视为已下载
    },
    "ml-25m": {
        "url": f"{BASE_URL}/ml-25m.zip",
        "marker": ("ratings.csv",),
    },
}


def _progress_hook(block_count: int, block_size: int, total_size: int):
    """回调：在终端打印下载进度条"""
    if total_size <= 0:
        return
    downloaded = min(block_count * block_size, total_size)
    pct = downloaded / total_size * 100
    bar_len = 40
    filled = int(bar_len * downloaded / total_size)
    bar = "█" * filled + "░" * (bar_len - filled)
    mb_dl = downloaded / (1024 * 1024)
    mb_total = total_size / (1024 * 1024)
    print(f"\r  [{bar}] {pct:5.1f}%  {mb_dl:.1f}/{mb_total:.1f} MB", end="", flush=True)
    if downloaded >= total_size:
        print()


def download_and_extract(name: str, url: str, markers: tuple[str, ...]) -> None:
    """下载 zip 文件并解压到 data/<name>/ 目录"""
    dest = DATA_DIR / name

    # 检查是否已下载
    if any((dest / m).exists() for m in markers):
        print(f"[skip] {name} 已存在: {dest}")
        return

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / f"{name}.zip"

    print(f"[下载] {name} ← {url}")
    try:
        urlretrieve(url, zip_path, reporthook=_progress_hook)
    except Exception as e:
        print(f"\n[错误] 下载失败: {e}")
        if zip_path.exists():
            zip_path.unlink()
        sys.exit(1)

    print(f"[解压] {name} → {dest}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # MovieLens zip 内部结构: ml-1m/ratings.dat 或 ml-25m/ratings.csv
            # 解压到 data/ 下，自动保留内部目录结构
            zf.extractall(DATA_DIR)
    except zipfile.BadZipFile:
        print(f"[错误] {zip_path} 不是有效的 zip 文件，请手动删除后重试")
        sys.exit(1)

    zip_path.unlink()
    print(f"[完成] {name}")


def main():
    parser = argparse.ArgumentParser(description="MovieLens 数据集下载工具")
    parser.add_argument(
        "--dataset",
        choices=["ml-1m", "ml-25m"],
        help="只下载指定数据集（不指定则下载全部）",
    )
    args = parser.parse_args()

    print("MovieLens 数据集下载")
    print("=" * 40)

    if args.dataset:
        info = DATASETS[args.dataset]
        download_and_extract(args.dataset, info["url"], info["marker"])
    else:
        for ds_name, info in DATASETS.items():
            download_and_extract(ds_name, info["url"], info["marker"])

    print("=" * 40)
    print()

    # 列出最终文件
    if DATA_DIR.exists():
        print("data/ 目录结构:")
        for root, dirs, files in os.walk(DATA_DIR):
            rel = Path(root).relative_to(DATA_DIR)
            for f in sorted(files):
                p = rel / f if str(rel) != "." else Path(f)
                print(f"  {p}")


if __name__ == "__main__":
    main()
