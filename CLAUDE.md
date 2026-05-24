# Movie Recommender — 项目架构与开发指南

## 项目概述

基于 Spark MLlib ALS 的离线电影推荐系统。训练层用 PySpark 做矩阵分解，在线层用 Flask 提供 REST API 查询推荐结果。

## 技术栈

- **离线训练**: Python 3.8+, PySpark 3.4.1, Spark MLlib ALS
- **在线服务**: Flask 2.3.0, Pandas 2.0.3
- **数据源**: MovieLens 1M / 25M
- **运行环境**: Ubuntu 20.04+, Java 8, ≥4GB RAM

## 架构设计

```
离线层 (Batch)                          在线层 (Serving)
┌──────────────────────┐               ┌──────────────────┐
│ train_als.py         │               │ app.py           │
│  1. 加载 ratings.dat │   写入 CSV    │  启动时加载 CSV   │
│  2. 训练/测试 80/20  │ ────────────→ │  到内存 dict      │
│  3. ALS 模型训练      │               │                  │
│  4. RMSE+P/R/NDCG评估 │               │ GET /recommend/:id│
│  5. 全量 Top-N 推荐   │               │ GET /movie/:id    │
│  6. CSV + 模型持久化  │               │ GET /health      │
└──────────────────────┘               └──────────────────┘
```

**核心设计决策**:
- 全量离线预计算模式：训练完后一次性为所有用户生成推荐结果落盘，在线层只做 kv 查询
- 优点：Flask 无状态、响应毫秒级、可水平扩展
- 缺点：无法处理新用户/新物品冷查询，需重新训练才能更新

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
├── download_data.sh      # 一键下载数据集脚本
├── requirements.txt     # Python 依赖
├── design.md            # 详细设计文档（含完整代码、ALS 原理、报告指引）
├── workflow.md          # 分阶段开发工作流（含验证标准）
├── README.md            # 项目概览与快速开始
└── CLAUDE.md            # 本文件
```

## 开发环境搭建

```bash
# 1. 安装 Java 8
sudo apt update && sudo apt install openjdk-8-jdk -y

# 2. 安装 Spark 3.4.1 到 /opt/spark，配置 SPARK_HOME 环境变量

# 3. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

	# 4. 下载数据集（一次下载 ml-1m + ml-25m）
	bash download_data.sh
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
2. 验证通过后再换 ml-25m（训练 ~15-30 分钟）
3. 每次改动后立即用 curl 验证 API 端点
4. 遇到问题先查 design.md §8 排错表

### Git 约定
- 不要提交数据集文件（.zip, ratings.dat 等），已在 .gitignore 中排除
- output/ 下的生成文件不提交，只保留 .gitkeep 占位目录结构
- 提交前确保 `python3 train_als.py --help` 和 `python3 app.py --help` 能正常运行

### 代码风格
- Python 3.8+ 语法，类型提示可用但非必须
- 命令行参数统一用 argparse，不做硬编码路径
- 日志用 logging 模块（非 print），带时间戳格式
- Spark RDD/DataFrame 操作避免 collect() 到 driver（OOM 风险）
