"""
一键运行：下载 -> 训练 -> 启动 API 服务

用法:
    # 先激活虚拟环境！
    #   Windows:  venv\\Scripts\\activate
    #   Linux:    source venv/bin/activate
    python run_all.py                          # 默认：ml-1m + ALS + 内存模式
    python run_all.py --dataset ml-25m         # 25M 数据集 + 自动 SQLite
    python run_all.py --tune                   # 超参数调优
    python run_all.py --hybrid --alpha 0.7     # 混合推荐
    python run_all.py --skip-download          # 跳过下载
    python run_all.py --skip-train             # 跳过训练（直接启动服务）
    python run_all.py --port 8080              # 指定端口
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def preflight_check(args: argparse.Namespace) -> None:
    """启动前检查环境是否就绪，发现问题给出明确指引后退出"""
    errors = []

    # 1. pyspark
    try:
        import pyspark  # noqa: F401
    except ImportError:
        errors.append(
            "pyspark 未安装。请先激活虚拟环境并安装依赖:\n"
            "  Windows:  venv\\Scripts\\activate && pip install -r requirements.txt\n"
            "  Linux:    source venv/bin/activate && pip install -r requirements.txt"
        )

    # 2. Java 版本检查（Spark 3.4.1 仅兼容 Java 8/11）
    java_home = os.environ.get("JAVA_HOME", "")
    if java_home:
        java_path = os.path.join(java_home, "bin", "java")
        if sys.platform == "win32":
            java_path += ".exe"
        if not os.path.exists(java_path):
            java_path = java_home  # fallback
    else:
        java_path = shutil.which("java")

    if java_path is None or not os.path.exists(str(java_path)):
        errors.append(
            "未找到 java。Spark 需要 Java 8（推荐）或 Java 11。\n"
            "  Windows: 下载 Adoptium Temurin 8 .zip 解压后设置 JAVA_HOME\n"
            "  Linux:   sudo apt install openjdk-8-jdk -y"
        )
    else:
        try:
            ver = subprocess.run(
                [str(java_path), "-version"], capture_output=True, text=True
            )
            output = ver.stderr or ver.stdout
            import re
            m = re.search(r'version "(\d+)', output)
            if m:
                major = int(m.group(1))
                if major >= 17:
                    errors.append(
                        f"当前 Java {major} 不兼容 Spark 3.4.1（仅支持 Java 8/11）。\n"
                        "请安装 Java 8 并设置 JAVA_HOME:\n"
                        "  1. 下载 Adoptium Temurin 8 .zip: https://adoptium.net/download/\n"
                        "  2. 解压到你的 JDK 目录（如 E:\\java_devlop\\jdk8）\n"
                        "  3. PowerShell: $env:JAVA_HOME = \"E:\\java_devlop\\jdk8\""
                    )
        except Exception:
            pass

    # 3. 训练产出（跳过训练时必须有）
    if args.skip_train:
        recs_dir = ROOT / "output" / "user_recs"
        if not recs_dir.exists() or not list(recs_dir.glob("part-*.csv")):
            errors.append(
                "未找到训练产出 output/user_recs/part-*.csv。\n"
                "请先运行训练: python train_als.py --data_dir data/ml-1m"
            )

    if errors:
        print("\n[环境检查失败]\n")
        for i, e in enumerate(errors, 1):
            print(f"  {i}. {e}\n")
        sys.exit(1)


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

    # 环境检查
    preflight_check(args)
    print("[环境检查] 通过")

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
            desc += f" (混合推荐, alpha={args.alpha})"

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
