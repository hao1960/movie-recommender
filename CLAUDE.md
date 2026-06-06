# Movie Recommender — 项目架构与开发指南

## 项目概述

基于 Spark MLlib ALS 的离线电影推荐系统。训练层用 PySpark 做矩阵分解，在线层用 Flask 提供 REST API 查询推荐结果。

## 技术栈

- **离线训练**: Python 3.8+, PySpark 3.4.1, Spark MLlib ALS + CrossValidator
- **在线服务**: Flask 2.3.0, Pandas 2.0.3, SQLite3（可选大容量模式）
- **测试**: pytest 8.x, Flask test client
- **数据源**: MovieLens 1M / 25M
- **运行环境**: Ubuntu 20.04+ / Windows 10+ / macOS, Java 8, ≥4GB RAM

## 架构设计

```
离线层 (Batch)                               在线层 (Serving)
┌────────────────────────────┐              ┌────────────────────┐
│ train_als.py               │              │ app.py             │
│  1. 加载数据(自动格式/编码) │   写入 CSV   │  内存 dict 或 SQLite │
│  2. 冷启动分析              │ ──────────→ │  结构化请求日志       │
│  3. 训练/测试 80/20        │              │                    │
│  4. ALS / 超参数网格搜索    │              │ GET /recommend/:id │
│  5. 6项指标评估             │              │ GET /movie/:id     │
│  6. 冷启动内容推荐          │              │ GET /health        │
│  7. 混合推荐(ALS+Content)  │              │ GET / (HTML 前端)   │
│  8. 过滤已评分 → 输出       │              │                    │
└────────────────────────────┘              └────────────────────┘
```

**核心设计决策**:
- 全量离线预计算模式：训练后一次性为所有用户生成推荐结果落盘，在线层做 kv 查询
- 冷启动用户（<5 条评分）自动切换为基于内容的电影类型 Jaccard 相似度推荐
- 混合推荐在 ALS 候选池内按内容相似度加权重排序，避免全量 Cross Join
- SQLite 模式适合 25M 数据集，启动无需全量加载到内存，按 userId 索引查询
- 优点：Flask 无状态、响应毫秒级、可水平扩展
- 缺点：新用户/新物品冷查询依赖内容 fallback，全量重训练才能更新模型

## 目录结构

```
movie-recommender/
├── data/
│   ├── ml-1m/           # MovieLens 1M 数据集（下载后）
│   └── ml-25m/          # MovieLens 25M 数据集（下载后）
├── output/              # 训练产出（Spark 自动生成）
│   ├── user_recs/       # 推荐结果 CSV（part-*.csv）
│   ├── movies/          # 电影标题/类型映射 CSV
│   └── als_model/       # ALS 模型 Parquet
├── train_als.py         # 离线训练脚本（实现代码见 design.md §4）
├── app.py               # Flask API 服务（实现代码见 design.md §5）
├── run_all.py           # 一键运行全流程：下载 → 训练 → 启动
├── download_data.py     # 数据集下载（跨平台 Python，推荐使用）
├── download_data.sh      # 一键下载数据集脚本（Linux 备选）
├── requirements.txt     # Python 依赖
├── tests/               # pytest 集成测试
│   ├── __init__.py
│   └── test_app.py      # Flask API 端点测试（10 cases）
├── static/              # 前端静态文件
│   └── index.html       # 复古电影院风格推荐展示页
├── design.md            # 详细设计文档（含完整代码、ALS 原理、报告指引）
├── workflow.md          # 分阶段开发工作流（含验证标准）
├── README.md            # 项目概览与快速开始
└── CLAUDE.md            # 本文件
```

## 开发环境搭建

### Windows 11（已验证可用）

**前提：Spark 3.4.1 仅兼容 Java 8/11，需要 Python 3.11（不要用 3.12+）。**

#### 1. Java 8

下载 [Adoptium Temurin 8](https://adoptium.net/download/)（选 .zip，解压到集中管理的目录）：

```powershell
# 解压到 E:\java_devlop\jdk8\
E:\java_devlop\jdk8\bin\java -version   # 确认 1.8.0_xxx
setx JAVA_HOME "E:\java_devlop\jdk8"    # 永久设置
```

> 多版本 Java 可并存，设 `JAVA_HOME` 指向 8 即可。

#### 2. Hadoop winutils（Spark 写文件必需）

从 [cdarlint/winutils](https://github.com/cdarlint/winutils) 下载对应 Hadoop 版本的 `winutils.exe` 和 `hadoop.dll`（Spark 3.4.1 → hadoop-3.3.x 目录）：

```powershell
# 放到集中目录
mkdir E:\java_devlop\hadoop\bin
# 将 winutils.exe 和 hadoop.dll 放入 E:\java_devlop\hadoop\bin\
setx HADOOP_HOME "E:\java_devlop\hadoop"   # 永久设置
```

#### 3. Python 环境（推荐 conda）

不要用 Python 3.12+（缺少 distutils，与 PySpark 3.4.1 不兼容）：

```bash
conda create -n movie python=3.11 -y
conda activate movie
pip install -r requirements.txt
```

#### 4. 数据集

```bash
python download_data.py                    # ml-1m + ml-25m
python download_data.py --dataset ml-1m    # 只要 1M
```

#### 5. 新终端启动前确认

每次新开 PowerShell：

```powershell
# JAVA_HOME / HADOOP_HOME setx 后永久生效，但需确认
$env:JAVA_HOME
$env:HADOOP_HOME
conda activate movie
```

### Ubuntu 20.04+

```bash
# 1. 安装 Java 8
sudo apt update && sudo apt install openjdk-8-jdk -y

# 2. 安装 Spark 3.4.1 到 /opt/spark，配置 SPARK_HOME 环境变量

# 3. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. 下载数据集
python download_data.py
```

## 开发约定

### 文档驱动开发
- 本项目的可执行代码在 `design.md` 的代码块中定义
- `train_als.py` 和 `app.py` 目前为桩代码（stub），团队从 `design.md` 提取实现
- 修改算法逻辑时，先在 `design.md` 中更新设计说明，再同步到 .py 文件
- `workflow.md` 定义了 5 个 Phase，每个 Phase 有明确验证标准

### 数据格式注意事项
- **MovieLens 1M**: `ratings.dat` 使用 `::` 分隔符，不能用标准 CSV reader
- **MovieLens 25M**: `ratings.csv` 使用 `,` 分隔符，可以用标准 CSV reader
- 加载数据时必须适配两种格式，不能硬编码分隔符

### 开发流程
1. 先用 ml-1m 数据集跑通全流程（训练 ~2-5 分钟）
   ```bash
   python run_all.py                          # 一键: 下载 → 训练 → 启动
   python run_all.py --tune                   # 超参数调优
   python run_all.py --hybrid --alpha 0.7     # 混合推荐
   ```
2. 验证通过后再换 ml-25m（训练 ~15-30 分钟）
   ```bash
   python run_all.py --dataset ml-25m         # 自动 SQLite + 4g 内存
   ```
3. 每次改动后立即用 curl 或 pytest 验证 API 端点
4. 遇到问题先查 design.md §8 排错表

### Git 约定
- 不要提交数据集文件（.zip, ratings.dat 等），已在 .gitignore 中排除
- output/ 下的生成文件不提交，只保留 .gitkeep 占位目录结构
- `logs/` 目录不提交（运行时生成）
- 提交前确保 `python -m pytest tests/ -q` 全部通过
- 提交前确保 `python train_als.py --help` 和 `python app.py --help` 能正常运行

### 代码风格
- Python 3.8+ 语法，类型提示可用但非必须
- 命令行参数统一用 argparse，不做硬编码路径
- 日志用 logging 模块（非 print），带时间戳格式
- Spark RDD/DataFrame 操作避免 collect() 到 driver（OOM 风险）
