# Movie Recommender — 基于 Spark ALS 的离线电影推荐系统

基于 Apache Spark MLlib 的交替最小二乘（ALS）协同过滤算法，对 MovieLens 百万级评分数据训练隐语义模型，为每个用户生成个性化 Top-N 电影推荐，并通过 Flask REST API 对外提供查询服务。

---

## 架构一览

```text
MovieLens 数据集            Spark ALS 离线训练              Flask API 在线服务
┌──────────────┐        ┌─────────────────────┐        ┌──────────────────┐
│ ratings.dat  │ ────→  │ 1. 数据加载与清洗    │        │ GET /recommend/1  │
│ movies.dat   │        │ 2. 训练/测试划分     │        │ GET /movie/1193   │
│ users.dat    │        │ 3. ALS 矩阵分解      │ ────→  │ GET /health       │
└──────────────┘        │ 4. RMSE + P/R/NDCG  │        └──────────────────┘
                        │ 5. 全量 Top-N 推荐   │
                        │ 6. CSV + 模型持久化  │
                        └─────────────────────┘
```

- **离线层**：Spark 批量训练，结果落盘为 CSV
- **在线层**：Flask 启动时加载 CSV 到内存，纯 key-value 查询，毫秒级响应

---

## 功能特性

- [x] 支持 MovieLens 1M 和 25M 数据集，命令行切换
- [x] ALS 模型训练，超参数可通过命令行配置
- [x] 多维度评估：RMSE（评分预测）+ Precision@K / Recall@K / NDCG@K（排序质量）
- [x] 冷启动分析：自动统计低评分用户占比
- [x] 模型持久化为 Parquet 格式，支持后续增量推理
- [x] Flask REST API：推荐查询、电影详情、健康检查
- [x] 命令行参数化，无硬编码路径
- [x] 完整日志输出，训练失败自动打印堆栈

---

## 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| Ubuntu | 20.04+ | 虚拟机或物理机均可 |
| Java | 8 | `openjdk-8-jdk`，Spark 3.x 最稳定搭配 |
| Spark | 3.4.1 | 预编译 Hadoop 3 版本 |
| Python | 3.8+ | 系统自带或 `apt install python3` |
| 内存 | ≥ 4 GB | 1M 数据集 2 GB 即可，25M 建议 8 GB+ |
| 磁盘 | ≥ 5 GB | 数据集 + 模型输出 |

---

## 快速开始（15 分钟跑通全流程）

以下命令从头搭建环境到启动 API 服务。每一步都有验证手段，出问题立即能定位。

### Step 1：安装 Java 8

```bash
sudo apt update && sudo apt install openjdk-8-jdk -y
java -version
# 预期: openjdk version "1.8.0_xxx"
```

### Step 2：安装 Spark

```bash
wget https://archive.apache.org/dist/spark/spark-3.4.1/spark-3.4.1-bin-hadoop3.tgz
sudo tar -xzf spark-3.4.1-bin-hadoop3.tgz -C /opt
sudo mv /opt/spark-3.4.1-bin-hadoop3 /opt/spark

cat >> ~/.bashrc << 'EOF'
export SPARK_HOME=/opt/spark
export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
export PYSPARK_DRIVER_PYTHON=python3
export PYSPARK_PYTHON=python3
EOF

source ~/.bashrc
spark-shell --version
# 预期: Spark 3.4.1 版本信息
```

### Step 3：创建项目并安装 Python 依赖

```bash
mkdir -p ~/movie-recommender && cd ~/movie-recommender
python3 -m venv venv
source venv/bin/activate
pip install pyspark==3.4.1 flask==2.3.0 pandas==2.0.3
```

> 验证：进入 `pyspark`，执行 `spark.range(10).show()` 能打印 10 行数据即正常。输入 `exit()` 退出。

### Step 4：下载数据集

```bash
# 一键下载 ml-1m + ml-25m 到 data/ 下
bash download_data.sh
# 确认: ls data/ml-1m/ 应看到 ratings.dat, movies.dat, users.dat
```

> 默认同时下载两个数据集。如果只想下某一个，可以直接用 `wget` 手动下载，见 [design.md](design.md) §3.1。

### Step 5：下载或创建训练脚本

将 `train_als.py` 放入 `~/movie-recommender/` 目录（代码见 [design.md](design.md) 第四节）。

### Step 6：运行训练

```bash
python3 train_als.py \
    --data_dir data/ml-1m \
    --output_dir output \
    --rank 50 \
    --max_iter 15 \
    --reg_param 0.1 \
    --top_n 10
```

训练完成后检查：

```bash
ls output/user_recs/part-*.csv   # 推荐结果
ls output/movies/part-*.csv      # 电影映射
ls output/als_model/             # 持久化模型
```

### Step 7：启动 API 服务

将 `app.py` 放入 `~/movie-recommender/` 目录（代码见 [design.md](design.md) 第五节）。

```bash
python3 app.py --port 5000 --recs_dir output/user_recs --movies_dir output/movies
```

### Step 8：验证 API

```bash
# 用户 1 的 Top-10 推荐
curl http://localhost:5000/recommend/1

# 用户 1 的 Top-5 推荐
curl "http://localhost:5000/recommend/1?limit=5"

# 电影详情
curl http://localhost:5000/movie/1193

# 服务状态
curl http://localhost:5000/health
```

全部返回有效 JSON 即为成功。

---

## 项目结构

```text
~/movie-recommender/
├── data/
│   ├── ml-1m/                    # MovieLens 1M 原始数据（下载后）
│   │   ├── ratings.dat           # 用户-电影-评分（:: 分隔）
│   │   ├── movies.dat            # 电影元信息
│   │   └── users.dat             # 用户画像
│   └── ml-25m/                   # MovieLens 25M 原始数据（下载后）
│       ├── ratings.csv           # 用户-电影-评分（逗号分隔）
│       ├── movies.csv            # 电影元信息
│       └── tags.csv              # 用户标签
├── output/                       # 训练产出（由 train_als.py 生成）
│   ├── user_recs/                # 推荐结果 CSV（多 part 文件）
│   ├── movies/                   # 电影标题/类型映射 CSV
│   └── als_model/                # ALS 模型（Parquet）
├── venv/                         # Python 虚拟环境（不提交）
├── train_als.py                  # 离线训练脚本（实现见 design.md §4）
├── app.py                        # Flask API 服务（实现见 design.md §5）
├── requirements.txt              # Python 依赖列表
├── CLAUDE.md                     # 项目架构与开发指南
├── .gitignore
├── design.md                     # 详细设计文档
├── workflow.md                   # 分阶段开发工作流
└── README.md                     # 本文件
```

---

## API 文档

### `GET /recommend/<user_id>`

为指定用户返回个性化电影推荐。

**参数**：

| 参数 | 位置 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `user_id` | path | int | 必填 | 用户 ID（1 ~ 6040 for ml-1m） |
| `limit` | query | int | 10 | 返回条数，范围 [1, 50] |

**响应示例**（200）：

```json
{
  "user_id": 1,
  "recommendations": [
    {
      "movieId": 1193,
      "title": "One Flew Over the Cuckoo's Nest (1975)",
      "genres": "Drama"
    },
    ...
  ]
}
```

**响应示例**（404）：

```json
{
  "error": "User 99999 not found"
}
```

### `GET /movie/<movie_id>`

查询单部电影元信息。

**响应示例**（200）：

```json
{
  "movieId": 1193,
  "title": "One Flew Over the Cuckoo's Nest (1975)",
  "genres": "Drama"
}
```

### `GET /health`

返回服务状态和加载数据量。

**响应示例**（200）：

```json
{
  "status": "ok",
  "users": 6040,
  "movies": 3706
}
```

---

## 配置参考

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `data/ml-1m` | 数据集路径，切换到 25M 时改为 `data/ml-25m` |
| `--output_dir` | `output` | 推荐结果和模型输出目录 |
| `--rank` | 50 | ALS 隐因子维度，越大表达力越强但训练越慢 |
| `--max_iter` | 15 | 最大迭代次数，通常 10~20 之间收敛 |
| `--reg_param` | 0.1 | L2 正则化系数，防止过拟合 |
| `--top_n` | 10 | 为每个用户推荐的电影数 |
| `--driver_memory` | 2g | Spark Driver JVM 堆内存 |

### 数据集规模对照

| 数据集 | 用户数 | 电影数 | 评分数 | 建议 `shuffle.partitions` | 训练时间（参考） |
|--------|--------|--------|--------|---------------------------|-----------------|
| ml-1m | 6,040 | 3,706 | 100 万 | 16 | 2~5 分钟 |
| ml-25m | 162,541 | 62,423 | 2,500 万 | 200 | 15~30 分钟 |

### 切换数据集

```bash
# 25M 数据集
wget https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip -d data/

# 训练时修改参数
python3 train_als.py --data_dir data/ml-25m --driver_memory 4g
# 同时需要修改 train_als.py 中 init_spark() 的 shuffle.partitions 为 200
```

---

## 开发指南

### 推荐工作流

详见 [workflow.md](workflow.md)。核心原则：

1. **先跑通再优化**：用 1M 数据 + 默认参数验证全流程，再考虑调参或切 25M
2. **每步有验证**：环境 `spark.range()` → 数据 `count()` 一致 → 训练 RMSE < 1.0 → API curl 返回 JSON
3. **出问题先看日志**：训练脚本默认输出 INFO 级别日志，时间戳 + 消息体，足以定位 90% 的问题

### Git 开发规范

#### 仓库克隆与分支管理

```bash
# 1. 克隆仓库
git clone https://github.com/<org>/movie-recommender.git
cd movie-recommender

# 2. 基于 main 创建功能分支（禁止直接在 main 上开发）
git checkout main
git pull origin main
git checkout -b feature/<你的名字>/<功能简述>

# 示例
git checkout -b feature/zhangsan/als-training
git checkout -b feature/lisi/flask-api
git checkout -b fix/wangwu/data-format-detection
```

#### 分支命名规范

| 前缀 | 用途 | 示例 |
|------|------|------|
| `feature/<name>/<desc>` | 新功能开发 | `feature/zhangsan/add-coverage-metric` |
| `fix/<name>/<desc>` | Bug 修复 | `fix/lisi/coldstart-crash` |
| `docs/<name>/<desc>` | 文档更新 | `docs/wangwu/api-examples` |
| `exp/<name>/<desc>` | 实验性分支（不合并） | `exp/zhangsan/try-ncf-model` |

#### 日常开发流程

```bash
# 每天开始工作前，同步 main 最新代码
git checkout main
git pull origin main

# 切回自己的分支，rebase main（保持提交历史线性）
git checkout feature/zhangsan/als-training
git rebase main

# 有冲突时解决冲突，然后继续
# git add <冲突文件>
# git rebase --continue

# 提交代码
git add <文件>
git commit -m "feat: 实现 ALS 模型训练脚本"

# 推送到远程（首次推送需要 -u）
git push -u origin feature/zhangsan/als-training
```

#### Commit Message 规范

采用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
<type>: <简短描述>

<详细说明（可选）>
```

| type | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `docs` | 文档修改 |
| `refactor`| 代码重构（不改变功能） |
| `test` | 测试相关 |
| `chore` | 构建/工具/依赖变更 |

示例：
```
feat: 实现 1M/25M 数据集分隔符自动检测

ratings.dat 使用 :: 分隔，ratings.csv 使用逗号分隔。
通过检测文件扩展名自动选择解析方式，避免硬编码。
```

#### 代码合并（PR）流程

1. 本地开发完成，通过自测后 push 到远程分支
2. 在 GitHub 上创建 Pull Request（feature/xxx → main）
3. 在 PR 描述中写清楚：**改了啥、怎么验证**
4. 至少 1 人 Code Review 通过后才能合并
5. 合并使用 **Squash and Merge**（将分支上的多个 commit 压缩为 1 个）
6. 合并后删除远程分支

#### 不要提交的内容

- `venv/`、`__pycache__/`、`.pyc`（已在 .gitignore 中排除）
- 数据集文件（`*.zip`、`ratings.dat` 等）
- `output/` 下的训练产物（`part-*.csv`、模型文件）
- IDE 配置文件（`.vscode/`、`.idea/`）
- 任何包含密码/密钥的 `.env` 文件

### 重要约定

- **始终激活虚拟环境**：每次新终端先 `source venv/bin/activate`
- **Spark 输出会覆盖**：`output/` 目录使用 `mode="overwrite"`，多次运行仅保留最后一次结果
- **不要手动创建 `output/`**：Spark 会自动创建，手动创建可能导致权限冲突
- **Flask 启动前必须先训练**：`app.py` 依赖 `output/user_recs/` 和 `output/movies/`

---

## 常见问题

| 现象 | 可能原因 | 解决 |
|------|---------|------|
| `java: command not found` | Java 未安装或 PATH 未配置 | `sudo apt install openjdk-8-jdk -y` |
| `NoClassDefFoundError` | Java 版本不兼容 | `sudo update-alternatives --config java` 选 Java 8 |
| `OutOfMemoryError` | Driver 内存不足 | 加 `--driver_memory 4g` 或改用 ml-1m |
| `FileNotFoundError: part-*.csv` | Flask 启动前未训练 | 先执行 `python3 train_als.py` |
| Flask 返回 404 | 用户 ID 超过数据范围 | ml-1m 用户 ID 范围 1~6040 |
| Flask 宿主机无法访问 | 监听地址或防火墙 | `host='0.0.0.0'` + `sudo ufw allow 5000` |
| 训练脚本报 `split("::")` 索引越界 | `ratings.dat` 文件损坏或有空行 | 重新下载并解压数据集 |

更多排错见 [design.md](design.md) 第八节。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [README.md](README.md) | 项目概述、快速开始、API 文档（本文件） |
| [CLAUDE.md](CLAUDE.md) | 项目架构与开发指南（新人/团队必读） |
| [design.md](design.md) | 详细设计：完整代码、ALS 原理分析、报告撰写指引 |
| [workflow.md](workflow.md) | 开发工作流：分阶段实现计划与验证标准 |

---

## 参考资料

- [MovieLens 数据集](https://grouplens.org/datasets/movielens/)
- [Spark MLlib ALS 官方文档](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)
- [Spark 性能调优指南](https://spark.apache.org/docs/latest/tuning.html)
