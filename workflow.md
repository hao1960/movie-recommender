# 开发工作流

## 阶段总览

```
Phase 1         Phase 2         Phase 3         Phase 4         Phase 5
环境搭建 ──→ 数据管道 ──→ 模型训练与评估 ──→ API服务 ──→ 优化与扩展
(0.5天)       (0.5天)        (1天)            (0.5天)        (按需)
```

每个 Phase 结束都有明确的**验证标准**，不通过不进入下一阶段。

---

## Phase 1：环境搭建

**目标**：Spark + Python 环境可正常启动，能执行基本的 Spark DataFrame 操作。

### 1.1 安装 Java 8

```bash
sudo apt update && sudo apt install openjdk-8-jdk -y
java -version  # 必须输出 1.8.0_xxx
```

### 1.2 安装 Spark

```bash
wget https://archive.apache.org/dist/spark/spark-3.4.1/spark-3.4.1-bin-hadoop3.tgz
sudo tar -xzf spark-3.4.1-bin-hadoop3.tgz -C /opt
sudo mv /opt/spark-3.4.1-bin-hadoop3 /opt/spark
```

追加环境变量到 `~/.bashrc`：

```bash
export SPARK_HOME=/opt/spark
export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
export PYSPARK_DRIVER_PYTHON=python3
export PYSPARK_PYTHON=python3
```

### 1.3 创建虚拟环境

```bash
mkdir -p ~/movie-recommender && cd ~/movie-recommender
python3 -m venv venv
source venv/bin/activate
pip install pyspark==3.4.1 flask==2.3.0 pandas==2.0.3
```

### 1.4 验证标准

在 `pyspark` 交互终端中执行以下代码，无报错即为通过：

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("smoke_test").getOrCreate()
df = spark.range(100)
df.show(5)
print(f"Partitions: {df.rdd.getNumPartitions()}")
spark.stop()
```

**退出条件**：`spark.range(100).show(5)` 能打印出 5 行数据。

---

## Phase 2：数据管道

**目标**：MovieLens 数据成功加载为 Spark DataFrame，通过基本质量检查。

### 2.1 下载数据

```bash
cd ~/movie-recommender
mkdir -p data
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip -d data/
```

### 2.2 实现数据加载函数

> **代码参考**：完整实现见 [design.md](design.md) 第四节，包含 `parse_args()`、`init_spark()`、`load_ratings()`、`load_movies()` 等全部函数。

按以下顺序在 `train_als.py` 中实现（先写脚本骨架，再逐步填充）：

1. **骨架**：`parse_args()` + `init_spark()` + `main()` 空壳
2. **`load_ratings()`**：读取 `ratings.dat`，解析 `::` 分隔符，返回 `(userId, movieId, rating)` 三列
3. **`load_movies()`**：读取 `movies.dat`，返回 `(movieId, title, genres)` 三列

### 2.3 数据质量检查

在 `main()` 中加载数据后打印统计信息：

```python
ratings = load_ratings(spark, args.data_dir)
movies = load_movies(spark, args.data_dir)
# 必须输出: N 用户, M 电影, K 条评分
```

### 2.4 验证标准

| 数据集 | 预期用户数 | 预期电影数 | 预期评分数 |
|--------|-----------|-----------|-----------|
| ml-1m | 6,040 | 3,706 | 1,000,209 |
| ml-25m | 162,541 | 62,423 | 25,000,095 |

**退出条件**：脚本输出统计数字与上表一致。

---

## Phase 3：模型训练与评估

**目标**：ALS 模型训练完成，RMSE < 1.0，排序指标正常输出。

### 3.1 训练/测试划分

```python
train, test = ratings.randomSplit([0.8, 0.2], seed=42)
```

先跑通，不要在此阶段做交叉验证。

### 3.2 ALS 训练

用默认超参数跑第一轮（先验证流程，后调参）：

```bash
python3 train_als.py --rank 50 --max_iter 15 --reg_param 0.1
```

### 3.3 实现评估

按顺序实现三项评估：

| 顺序 | 指标 | 含义 |
|------|------|------|
| 1 | RMSE | 评分预测误差（回归指标，必做） |
| 2 | 冷启动丢弃比例 | 检查 `coldStartStrategy="drop"` 影响（诊断用） |
| 3 | Precision@10 / Recall@10 / NDCG@10 | 推荐排序质量（报告核心指标） |

### 3.4 生成推荐结果

```python
user_recs_flat = generate_recommendations(model, args)
save_outputs(user_recs_flat, movies, model, args.output_dir)
```

### 3.5 验证标准

- RMSE < 1.0（1M 数据集，rank=50 时通常 0.83~0.90）
- `output/user_recs/` 和 `output/movies/` 目录存在且包含 part-*.csv
- `output/als_model/` 目录存在（模型持久化成功）
- 冷启动丢弃比例 < 5%（若 >5% 说明 train/test split 有问题）

**退出条件**：三项全部满足。

---

## Phase 4：API 服务

**目标**：Flask 服务启动，`/recommend/1` 返回 JSON 推荐列表，电影标题可读。

### 4.1 实现数据加载

```python
# app.py 中实现 load_csv_dir() → load_recs() → load_movies()
# 启动时一次性加载到全局 dict
init_data(args.recs_dir, args.movies_dir)
```

### 4.2 实现路由

| 路由 | 用途 | 优先级 |
|------|------|--------|
| `GET /` | 导航页 | 低 |
| `GET /health` | 健康检查 | 中 |
| `GET /recommend/<user_id>?limit=N` | 核心推荐接口 | **高** |
| `GET /movie/<movie_id>` | 电影详情（辅助调试） | 低 |

先实现 `/recommend/<user_id>` 能返回 `[{"movieId": ..., "title": ...}]`，其余路由后续补齐。

### 4.3 验证标准

```bash
# 终端 1
python3 app.py --port 5000

# 终端 2
curl http://localhost:5000/recommend/1
# 预期返回: {"user_id":1, "recommendations":[{"movieId":...,"title":"...","genres":"..."}, ...]}

curl http://localhost:5000/recommend/99999
# 预期返回: 404 + {"error": "User 99999 not found"}

curl http://localhost:5000/health
# 预期返回: {"movies":3706, "status":"ok", "users":6040}
```

**退出条件**：三个 curl 测试全部返回预期结果。

---

## Phase 5：优化与扩展（按需选择）

以下任务无顺序依赖，根据时间和需求选择性实现。

### 5.1 超参数调优

用 `CrossValidator` + `ParamGridBuilder` 搜索最优 `(rank, regParam, maxIter)` 组合。仅当 Phase 3 的基线 RMSE > 0.95 时优先级提高。

### 5.2 HTML 前端页面

用纯 HTML + 原生 JS 做一个推荐展示页：
- 输入 userId → 调用 `/recommend/<id>` → 渲染电影卡片
- 可嵌入 TMDb 海报图片（通过电影标题搜索）

### 5.3 切换 25M 数据集

```bash
# 重新训练即可，注意调大 shuffle.partitions
python3 train_als.py --data_dir data/ml-25m --rank 50 --max_iter 15
```

### 5.4 基于内容的冷启动方案

对评分 < 5 条的用户，改为推荐其历史评分电影的同类电影（利用 genres 字段做 Jaccard 相似度）。

### 5.5 添加实时日志

在 Flask 中记录每次请求的 `user_id`、响应时间、返回数量，写入日志文件供后期分析。

---

## 开发原则

- **每个 Phase 结束时验证，不跳过**。Phase N 的 bug 留到 Phase N+2 排查代价是 10 倍。
- **先用 ml-1m 跑通全流程，再考虑换 25M**。1M 数据每轮训练约 2 分钟，25M 约 20 分钟，迭代效率差一个数量级。
- **代码改动后立即用 curl 验证 API**，不要等到全部写完了才测试。
- **遇到报错先查第八节排错表**，未覆盖的错误补充进去。
