#!/bin/bash
set -euo pipefail

BASE_URL="https://files.grouplens.org/datasets/movielens"
DATA_DIR="$(cd "$(dirname "$0")" && pwd)/data"

download_and_extract() {
    local name="$1"
    local url="$2"
    local dest="$DATA_DIR/$name"

    if [ -f "$dest/ratings.dat" ] || [ -f "$dest/ratings.csv" ]; then
        echo "[skip] $name 已存在: $dest"
        return
    fi

    echo "[下载] $name ← $url"
    mkdir -p "$dest"
    local zip_file="$DATA_DIR/${name}.zip"
    wget -q --show-progress -O "$zip_file" "$url"
    echo "[解压] $name → $dest"
    unzip -qo "$zip_file" -d "$dest"
    rm "$zip_file"
    echo "[完成] $name"
}

echo "MovieLens 数据集下载"
echo "===================="
download_and_extract "ml-1m"  "$BASE_URL/ml-1m.zip"
download_and_extract "ml-25m" "$BASE_URL/ml-25m.zip"
echo "===================="

# 列出最终文件
echo ""
echo "data/ 目录结构:"
find "$DATA_DIR" -type f | sort
