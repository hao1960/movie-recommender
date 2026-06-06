"""一键运行：下载 → 训练 → 启动 API 服务

用法:
    python run_all.py                          # 默认：ml-1m + ALS + 内存模式
    python run_all.py --dataset ml-25m         # 25M 数据集 + 自动 SQLite
    python run_all.py --tune                   # 超参数调优
    python run_all.py --hybrid --alpha 0.7     # 混合推荐
    python run_all.py --skip-download          # 跳过下载
    python run_all.py --skip-train             # 跳过训练（直接启动服务）
    python run_all.py --port 8080              # 指定端口
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: str, description: str) -> int:
    print(f"\n{'=' * 60}")
    print(f">>> {description}")
    print(f">>> {cmd}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, shell=True, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\n[失败] {description} (exit code {result.returncode})")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="一键运行 Movie Recommender 全流程")
    parser.add_argument("--dataset", default="ml-1m", choices=["ml-1m", "ml-25m"],
                        help="数据集（默认 ml-1m）")
    parser.add_argument("--skip-download", action="store_true", help="跳过数据下载")
    parser.add_argument("--skip-train", action="store_true", help="跳过模型训练")
    parser.add_argument("--tune", action="store_true", help="超参数网格搜索")
    parser.add_argument("--hybrid", action="store_true", help="混合推荐")
    parser.add_argument("--alpha", type=float, default=0.7, help="混合推荐 ALS 权重")
    parser.add_argument("--rank", type=int, default=50)
    parser.add_argument("--max_iter", type=int, default=15)
    parser.add_argument("--port", type=int, default=5000, help="API 端口")
    parser.add_argument("--log_file", default="logs/api.log", help="日志文件")
    args = parser.parse_args()

    data_dir = f"data/{args.dataset}"
    db_path = f"output/{args.dataset}.db"
    is_large = args.dataset == "ml-25m"

    # ---- Step 1: 下载 ----
    if not args.skip_download:
        if run(f"python download_data.py --dataset {args.dataset}",
               f"Step 1/3: 下载 {args.dataset} 数据集"):
            sys.exit(1)
    else:
        print(f"\n[跳过] 数据下载")

    # ---- Step 2: 训练 ----
    if not args.skip_train:
        train_cmd = f"python train_als.py --data_dir {data_dir} --output_dir output"
        train_cmd += f" --rank {args.rank} --max_iter {args.max_iter}"
        if is_large:
            train_cmd += " --driver_memory 4g"
        if args.tune:
            train_cmd += " --tune"
        if args.hybrid:
            train_cmd += f" --hybrid --alpha {args.alpha}"

        desc = f"Step 2/3: 训练模型"
        if args.tune:
            desc += " (超参数调优)"
        if args.hybrid:
            desc += f" (混合推荐, α={args.alpha})"

        if run(train_cmd, desc):
            sys.exit(1)
    else:
        print(f"\n[跳过] 模型训练")

    # ---- Step 3: 启动 API ----
    serve_cmd = f"python app.py --port {args.port}"
    serve_cmd += f" --recs_dir output/user_recs --movies_dir output/movies"
    serve_cmd += f" --log_file {args.log_file}"
    if is_large:
        serve_cmd += f" --db {db_path}"

    desc = f"Step 3/3: 启动 API 服务"
    if is_large:
        desc += " (SQLite 模式)"
    run(serve_cmd, desc)


if __name__ == "__main__":
    main()
