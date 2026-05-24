# 基于 Spark ALS 的大规模离线电影推荐系统

## 一、项目概述

- **目标**：使用 Spark MLlib 的 ALS（Alternating Least Squares）算法，基于 MovieLens 数据集训练协同过滤模型，为每个用户生成 Top-N 电影推荐，并通过 Flask 提供 RESTful API 展示推荐结果。
- **核心技术栈**：Spark DataFrame / MLlib / ALS 矩阵分解 / Flask
- **数据源**：MovieLens 1M（约 100 万条评分，6040 用户 × 3706 电影）或 25M（约 2500 万条评分，162000 用户 × 62000 电影）
- **运行环境**：Ubuntu 20.04+、Java 8、Spark 3.4.1、Python 3.8+

### 系统架构图

```text
┌───────────────────────────────────────────────────┐
│                   离线训练层                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐  │
│  │ ratings  │ → │ ALS 训练  │ → │ 推荐结果 CSV  │  │
│  │ movies   │   │ (Spark)  │   │ + 模型持久化   │  │
│  └──────────┘   └──────────┘   └──────────────┘  │
└───────────────────────────────────────────────────┘
                          ↓
┌───────────────────────────────────────────────────┐
│                   在线服务层                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐  │
│  │ Flask    │ ← │ 推荐缓存  │ ← │ 预计算推荐    │  │
│  │ REST API │   │ (内存)   │   │ 结果          │  │
│  └──────────┘   └──────────┘   └──────────────┘  │
└───────────────────────────────────────────────────┘
```

> **架构说明**：本方案采用全量离线预计算模式——训练完成后一次性为所有用户生成推荐结果并落盘，在线层仅做 key-value 查询。优点是 Flask 服务无状态、响应极快；缺点是无法处理新用户/新物品的冷查询（需重新训练）。生产环境下应将全局 dict 缓存替换为 Redis 或 SQLite 等外部存储，以便多进程/多实例共享。

---

## 二、环境准备（Ubuntu 虚拟机）

### 2.1 安装 Java 8

Spark 3.x 运行时要求 Java 8/11/17，推荐 Java 8（最稳定）：

```bash
sudo apt update
sudo apt install openjdk-8-jdk -y
java -version   # 确认输出为 1.8.0_xxx

# 如果系统同时安装了多个 Java 版本，手动选择：
sudo update-alternatives --config java
```

### 2.2 安装 Spark 3.4.1

```bash
wget https://archive.apache.org/dist/spark/spark-3.4.1/spark-3.4.1-bin-hadoop3.tgz
sudo tar -xzf spark-3.4.1-bin-hadoop3.tgz -C /opt
sudo mv /opt/spark-3.4.1-bin-hadoop3 /opt/spark

# 配置环境变量
cat >> ~/.bashrc << 'EOF'
export SPARK_HOME=/opt/spark
export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
export PYSPARK_DRIVER_PYTHON=python3
export PYSPARK_PYTHON=python3
EOF

source ~/.bashrc
spark-shell --version   # 验证安装
```

### 2.3 创建虚拟环境并安装依赖

```bash
sudo apt install python3-pip python3-venv -y

# 创建隔离的虚拟环境
cd ~/movie-recommender
python3 -m venv venv
source venv/bin/activate

# 安装依赖（固定版本号保证可复现）
pip install pyspark==3.4.1 flask==2.3.0 pandas==2.0.3
```

> **版本锁定说明**：固定主版本号避免后续兼容性问题。若使用 Spark 3.5+，PySpark 版本需对应调整。每次重新打开终端需要执行 `source venv/bin/activate` 激活环境。如需记录依赖快照：`pip freeze > requirements.txt`。

---

## 三、数据集下载与预处理

### 3.1 下载 MovieLens

```bash
mkdir -p ~/movie-recommender/data && cd ~/movie-recommender

# MovieLens 1M（推荐先用此数据集验证流程，训练约 2-5 分钟）
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip -d data/

# MovieLens 25M（后期替换，训练约 15-30 分钟）
# wget https://files.grouplens.org/datasets/movielens/ml-25m.zip
# unzip ml-25m.zip -d data/
```

### 3.2 数据文件结构

**MovieLens 1M**（`::` 分隔符，不能用标准 CSV reader）：

| 文件 | 格式 | 示例 |
|------|------|------|
| `ratings.dat` | `UserId::MovieId::Rating::Timestamp` | `1::1193::5::978300760` |
| `movies.dat` | `MovieId::Title::Genres` | `1193::One Flew Over the Cuckoo's Nest (1975)::Drama` |
| `users.dat` | `UserId::Gender::Age::Occupation::Zip-code` | `1::F::1::10::48067` |

**MovieLens 25M**（`,` 分隔符，可用标准 CSV reader）：

| 文件 | 格式 |
|------|------|
| `ratings.csv` | `userId,movieId,rating,timestamp` |
| `movies.csv` | `movieId,title,genres` |
| `tags.csv` | `userId,movieId,tag,timestamp` |

> **加载时注意**：`load_ratings()` 必须根据数据集自动检测分隔符，或通过 `--data_dir` 路径判断（`ml-1m` → `::`，`ml-25m` → `,`），不能硬编码单一格式。

### 3.3 数据质量检查（Spark SQL 交互式分析）

在编写训练脚本前，建议先用 `pyspark` 交互式终端快速探查数据（注意：`ratings.dat` 使用双冒号 `::` 分隔，Spark CSV reader 仅支持单字符分隔符，因此需要 RDD 解析）：

```python
# 启动 pyspark 后执行
ratings = spark.read.text("data/ml-1m/ratings.dat") \
    .rdd.map(lambda row: row[0].split("::")) \
    .filter(lambda parts: len(parts) >= 3) \
    .map(lambda parts: (int(parts[0]), int(parts[1]), float(parts[2]))) \
    .toDF(["userId", "movieId", "rating"])

# 基本统计
ratings.describe("rating").show()
# 用户评分数量分布
ratings.groupBy("userId").count().describe("count").show()
# 电影评分数量分布
ratings.groupBy("movieId").count().describe("count").show()
```

---

## 四、训练脚本 `train_als.py`

### 4.1 完整代码

```python
"""
离线 ALS 训练脚本
用法: python3 train_als.py [--data_dir data/ml-1m] [--output_dir output]
"""
import argparse
import logging
import sys
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, explode, count, avg, stddev, collect_list
from pyspark.mllib.evaluation import RankingMetrics

# ---------- 日志配置 ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Spark ALS 电影推荐训练")
    parser.add_argument("--data_dir", default="data/ml-1m", help="MovieLens 数据集目录")
    parser.add_argument("--output_dir", default="output", help="输出目录")
    parser.add_argument("--rank", type=int, default=50, help="ALS 隐因子维度")
    parser.add_argument("--max_iter", type=int, default=15, help="ALS 最大迭代次数")
    parser.add_argument("--reg_param", type=float, default=0.1, help="ALS 正则化参数")
    parser.add_argument("--top_n", type=int, default=10, help="为每个用户推荐 Top-N")
    parser.add_argument("--driver_memory", default="2g", help="Spark driver 内存")
    return parser.parse_args()


def init_spark(driver_memory: str) -> SparkSession:
    return SparkSession.builder \
        .appName("MovieRec_ALS") \
        .config("spark.driver.memory", driver_memory) \
        .config("spark.sql.shuffle.partitions", "16") \
        .config("spark.default.parallelism", "16") \
        .getOrCreate()


def load_ratings(spark: SparkSession, data_dir: str):
    """读取 ratings.dat，返回 (userId, movieId, rating) DataFrame"""
    ratings_path = f"{data_dir}/ratings.dat"
    logger.info(f"读取评分数据: {ratings_path}")

    # 注意：ratings.dat 使用 :: 分隔（非标准 CSV），因此用 RDD map 解析
    ratings = spark.read.text(ratings_path) \
        .rdd.map(lambda row: row[0].split("::")) \
        .filter(lambda parts: len(parts) >= 3) \
        .map(lambda parts: (int(parts[0]), int(parts[1]), float(parts[2]))) \
        .toDF(["userId", "movieId", "rating"])

    ratings.cache()
    n_users = ratings.select("userId").distinct().count()
    n_items = ratings.select("movieId").distinct().count()
    n_ratings = ratings.count()
    logger.info(f"数据统计: {n_users} 用户, {n_items} 电影, {n_ratings} 条评分")
    return ratings


def load_movies(spark: SparkSession, data_dir: str):
    """读取 movies.dat，返回 (movieId, title, genres) DataFrame"""
    movies_path = f"{data_dir}/movies.dat"
    logger.info(f"读取电影数据: {movies_path}")
    movies = spark.read.text(movies_path) \
        .rdd.map(lambda row: row[0].split("::")) \
        .filter(lambda parts: len(parts) >= 3) \
        .map(lambda parts: (int(parts[0]), parts[1], parts[2])) \
        .toDF(["movieId", "title", "genres"])
    return movies


def analyze_cold_start(ratings):
    """冷启动分析：统计评分数量极少（<5条）的用户占比"""
    user_rating_counts = ratings.groupBy("userId").agg(count("rating").alias("cnt"))
    user_rating_counts.cache()   # 缓存避免两次 count() 触发重复 groupBy
    cold_users = user_rating_counts.filter(col("cnt") < 5)
    total_users = user_rating_counts.count()
    n_cold = cold_users.count()
    logger.info(f"冷启动用户（<5条评分）: {n_cold}/{total_users} = "
                f"{n_cold / max(total_users, 1) * 100:.1f}%")
    user_rating_counts.unpersist()
    return n_cold, total_users


def train_als(train, args):
    """训练 ALS 模型"""
    logger.info(f"ALS 参数: rank={args.rank}, maxIter={args.max_iter}, "
                f"regParam={args.reg_param}")
    als = ALS(
        maxIter=args.max_iter,
        regParam=args.reg_param,
        rank=args.rank,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
        seed=42,
        nonnegative=True,       # 因子矩阵非负约束，提升可解释性
        implicitPrefs=False,     # 显式评分反馈
    )
    t0 = time.time()
    model = als.fit(train)
    elapsed = time.time() - t0
    logger.info(f"ALS 训练完成，耗时 {elapsed:.1f}s")
    return model


def evaluate_model(model, test):
    """计算测试集 RMSE，并报告冷启动丢弃情况"""
    predictions = model.transform(test)
    total_test = test.count()
    predicted = predictions.count()
    dropped = total_test - predicted
    if dropped > 0:
        logger.warning(
            f"coldStartStrategy='drop' 丢弃了 {dropped}/{total_test} 条记录 "
            f"({dropped / total_test * 100:.1f}%)，请检查训练/测试划分是否合理"
        )
    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    logger.info(f"测试集 RMSE = {rmse:.4f}（基于 {predicted} 条有效预测）")
    return rmse


def compute_ranking_metrics(model, test, k=10):
    """计算 Precision@K、Recall@K、NDCG@K（推荐质量指标）"""
    test_users = test.select("userId").distinct()
    # 为测试集用户生成 Top-K 推荐
    user_recs = model.recommendForUserSubset(test_users, k)

    # 正样本：评分 >= 3.5 的电影
    actual = test.filter(col("rating") >= 3.5) \
        .groupBy("userId").agg(collect_list("movieId").alias("actual_items"))

    pred = user_recs.select(
        "userId",
        col("recommendations.movieId").alias("pred_items")
    )

    joined = pred.join(actual, "userId", "inner")
    pred_actual_rdd = joined.select("pred_items", "actual_items").rdd \
        .map(lambda r: (list(map(int, r.pred_items)), list(map(int, r.actual_items))))

    metrics = RankingMetrics(pred_actual_rdd)
    p_at_k = metrics.precisionAt(k)
    r_at_k = metrics.recallAt(k)
    ndcg = metrics.ndcgAt(k)
    logger.info(
        f"Precision@{k}={p_at_k:.4f}  Recall@{k}={r_at_k:.4f}  NDCG@{k}={ndcg:.4f}"
    )
    return p_at_k, r_at_k, ndcg


def generate_recommendations(model, args):
    """为所有用户生成 Top-N 推荐"""
    logger.info(f"为所有用户生成 Top-{args.top_n} 推荐...")
    t0 = time.time()
    user_recs = model.recommendForAllUsers(args.top_n)

    # 展开嵌套结构: (userId, [{movieId, rating}, ...]) -> (userId, movieId, predRating)
    user_recs_flat = user_recs.select(
        "userId",
        explode("recommendations").alias("rec")
    ).select(
        "userId",
        col("rec.movieId").alias("movieId"),
        col("rec.rating").alias("predRating")
    )
    elapsed = time.time() - t0
    logger.info(f"推荐生成完成，耗时 {elapsed:.1f}s")
    return user_recs_flat


def save_outputs(user_recs_flat, movies, model, output_dir):
    """保存推荐结果、电影映射和模型"""
    # 推荐结果保留多个 part 文件（分布式写入），Flask 侧 load_csv_dir 会合并读取
    recs_path = f"{output_dir}/user_recs"
    logger.info(f"保存推荐结果到 {recs_path}")
    user_recs_flat.write.csv(recs_path, header=True, mode="overwrite")

    # 电影映射数据量小，合并为单文件方便查看
    movies_path = f"{output_dir}/movies"
    logger.info(f"保存电影映射到 {movies_path}")
    movies.coalesce(1).write.csv(movies_path, header=True, mode="overwrite")

    # 持久化 ALS 模型（供后续新用户推理或增量更新）
    model_path = f"{output_dir}/als_model"
    logger.info(f"持久化 ALS 模型到 {model_path}")
    model.write().overwrite().save(model_path)


def main():
    args = parse_args()
    spark = init_spark(args.driver_memory)

    try:
        # 1. 加载数据
        ratings = load_ratings(spark, args.data_dir)
        movies = load_movies(spark, args.data_dir)

        # 2. 冷启动分析
        analyze_cold_start(ratings)

        # 3. 划分训练/测试集
        train, test = ratings.randomSplit([0.8, 0.2], seed=42)
        logger.info("训练集/测试集划分: 80%/20%")

        # 4. 训练 ALS
        model = train_als(train, args)

        # 5. 评估
        evaluate_model(model, test)
        compute_ranking_metrics(model, test, k=args.top_n)

        # 6. 生成推荐
        user_recs_flat = generate_recommendations(model, args)

        # 7. 保存
        save_outputs(user_recs_flat, movies, model, args.output_dir)

        logger.info("全部流程完成！")
    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
```

### 4.2 运行训练

```bash
cd ~/movie-recommender
python3 train_als.py --data_dir data/ml-1m --output_dir output --rank 50 --max_iter 15
```

> **预期输出**：RMSE 约 0.83~0.90，Precision@10 / Recall@10 / NDCG@10 等排序指标也会一并打印。rank=50、maxIter=15 是经过权衡的推荐配置——rank 太低欠拟合，太高过拟合且训练变慢。使用 25M 数据集时，建议在 `init_spark` 中增大 `shuffle.partitions` 至 200。

### 4.3 超参数调优（可选）

使用 Spark MLlib 的 `ParamGridBuilder` + `CrossValidator` 进行网格搜索：

```python
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# 定义 ALS estimator（注意不要在此处调用 .fit()）
als = ALS(
    maxIter=15, userCol="userId", itemCol="movieId", ratingCol="rating",
    coldStartStrategy="drop", seed=42,
)
evaluator = RegressionEvaluator(
    metricName="rmse", labelCol="rating", predictionCol="prediction"
)

param_grid = ParamGridBuilder() \
    .addGrid(als.rank, [10, 30, 50]) \
    .addGrid(als.regParam, [0.01, 0.1, 0.5]) \
    .addGrid(als.maxIter, [10, 15]) \
    .build()

cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid,
                    evaluator=evaluator, numFolds=3, seed=42)
cv_model = cv.fit(train)

# 查看最优参数
print(f"最佳 rank={cv_model.bestModel.rank}, "
      f"regParam={cv_model.bestModel._java_obj.parent().getRegParam()}")
```

---

## 五、Flask 推荐接口 `app.py`

### 5.1 完整代码

```python
"""
推荐系统 REST API 服务
用法: python3 app.py [--port 5000] [--recs_dir output/user_recs] [--movies_dir output/movies]
"""
import argparse
import logging
import os

import pandas as pd
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局缓存
_rec_dict = {}
_movie_titles = {}
_movie_genres = {}


def load_csv_dir(dir_path: str) -> pd.DataFrame:
    """加载 Spark 输出的 CSV 目录（合并所有 part-*.csv 文件）"""
    frames = []
    for fname in sorted(os.listdir(dir_path)):
        if fname.startswith("part-") and fname.endswith(".csv"):
            filepath = os.path.join(dir_path, fname)
            frames.append(pd.read_csv(filepath))
    if not frames:
        raise FileNotFoundError(f"在 {dir_path} 中未找到 part-*.csv 文件")
    return pd.concat(frames, ignore_index=True)


def load_recs(recs_dir: str):
    """加载推荐结果 -> {userId: [movieId, ...]}"""
    df = load_csv_dir(recs_dir)
    # 按 predRating 降序排列后取每组 movieId 列表
    df_sorted = df.sort_values(["userId", "predRating"], ascending=[True, False])
    return df_sorted.groupby("userId")["movieId"].apply(list).to_dict()


def load_movies(movies_dir: str):
    """加载电影标题和类型映射 -> {movieId: title}, {movieId: genres}"""
    df = load_csv_dir(movies_dir)
    titles = dict(zip(df["movieId"], df["title"]))
    genres = dict(zip(df["movieId"], df["genres"]))
    return titles, genres


def init_data(recs_dir: str, movies_dir: str):
    global _rec_dict, _movie_titles, _movie_genres
    logger.info(f"加载推荐数据: {recs_dir}")
    _rec_dict = load_recs(recs_dir)
    logger.info(f"加载电影数据: {movies_dir}")
    _movie_titles, _movie_genres = load_movies(movies_dir)
    logger.info(f"加载完成: {len(_rec_dict)} 用户, {len(_movie_titles)} 电影")


@app.route("/recommend/<int:user_id>")
def recommend(user_id):
    """为指定用户返回 Top-N 推荐"""
    limit = request.args.get("limit", 10, type=int)
    limit = max(1, min(limit, 50))   # 限制范围 [1, 50]，防止负数或过大请求
    if user_id not in _rec_dict:
        return jsonify({"error": f"User {user_id} not found"}), 404

    movie_ids = _rec_dict[user_id][:limit]
    rec_list = []
    for mid in movie_ids:
        mid_int = int(mid)
        rec_list.append({
            "movieId": mid_int,
            "title": _movie_titles.get(mid_int, "Unknown"),
            "genres": _movie_genres.get(mid_int, "Unknown"),
        })
    return jsonify({"user_id": user_id, "recommendations": rec_list})


@app.route("/movie/<int:movie_id>")
def movie_info(movie_id):
    """查询单部电影信息"""
    return jsonify({
        "movieId": movie_id,
        "title": _movie_titles.get(movie_id, "Unknown"),
        "genres": _movie_genres.get(movie_id, "Unknown"),
    })


@app.route("/health")
def health():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "users": len(_rec_dict),
        "movies": len(_movie_titles),
    })


@app.route("/")
def home():
    return """
    <h2>Movie Recommender API</h2>
    <ul>
        <li><a href="/recommend/1">/recommend/1</a> — 用户 1 的推荐</li>
        <li><a href="/recommend/1?limit=5">/recommend/1?limit=5</a> — 用户 1 的 Top-5 推荐</li>
        <li><a href="/movie/1193">/movie/1193</a> — 电影详情</li>
        <li><a href="/health">/health</a> — 服务状态</li>
    </ul>
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--recs_dir", default="output/user_recs")
    parser.add_argument("--movies_dir", default="output/movies")
    args = parser.parse_args()

    init_data(args.recs_dir, args.movies_dir)
    logger.info(f"启动服务: http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
```

### 5.2 启动服务与测试

```bash
# 终端 1：启动 Flask
python3 app.py --port 5000

# 终端 2：测试接口
curl http://localhost:5000/recommend/1
curl "http://localhost:5000/recommend/1?limit=5"
curl http://localhost:5000/movie/1193
curl http://localhost:5000/health
```

从宿主机浏览器访问 `http://<虚拟机IP>:5000/recommend/1`。如果无法访问，执行：

```bash
sudo ufw allow 5000
```

---

## 六、项目目录结构

```text
~/movie-recommender/
├── data/
│   ├── ml-1m/                # MovieLens 1M 数据集（:: 分隔符）
│   │   ├── ratings.dat
│   │   ├── movies.dat
│   │   └── users.dat
│   └── ml-25m/               # MovieLens 25M 数据集（逗号分隔符）
│       ├── ratings.csv
│       ├── movies.csv
│       └── tags.csv
├── output/                   # 训练产出（Spark 写入）
│   ├── user_recs/            # 用户推荐结果 CSV（part-*.csv）
│   ├── movies/               # 电影标题/类型映射 CSV
│   └── als_model/            # ALS 模型持久化（Parquet 格式）
├── train_als.py              # 离线训练脚本
├── app.py                    # Flask 推荐服务
├── requirements.txt          # Python 依赖
├── CLAUDE.md                 # 项目架构与开发指南
└── design.md                 # 本文档
```

---

## 七、报告核心内容建议（大数据原理分析）

### 7.1 ALS 并行化原理与 Spark 执行模型

**算法本质**：ALS 将用户-物品评分矩阵 $R_{m \times n}$ 分解为两个低秩矩阵 $U_{m \times k}$（用户因子）和 $V_{n \times k}$（物品因子），使得 $R \approx U \times V^T$。优化目标：

$$\min_{U, V} \sum_{(i,j) \in \Omega} (R_{ij} - U_i V_j^T)^2 + \lambda(\|U_i\|^2 + \|V_j\|^2)$$

其中 $\Omega$ 是已观测评分的集合。

**并行策略（Spark MLlib ALS 实际实现）**：

- 将用户和物品分别按 ID 范围划分为多个 Block（用户块 $U_1, U_2, \dots$，物品块 $V_1, V_2, \dots$）
- 评分数据根据用户 ID 和物品 ID 路由到对应的 $(UserBlock, ItemBlock)$ 交叉分区，形成分块评分矩阵
- **求解 $U$ 的半轮迭代**：固定所有 $V$ 块，每个用户块从 BlockManager 拉取其所评分物品对应的 $V$ 块到本地，独立求解该块内所有用户的因子向量（无全局 Shuffle，仅拉取需要的 $V$ 块）
- **求解 $V$ 的半轮迭代**：固定所有 $U$ 块，每个物品块拉取评分过该块内物品的用户对应的 $U$ 块，独立求解该块内所有物品的因子向量
- BlockManager 作为内存缓存层：因子块在 Executor 间通过点对点传输，首次拉取后缓存在本地，下次迭代直接命中，大幅减少网络 I/O
- 与 "broadcast $V$" 的简化说法不同：$V$ 不是全量广播，而是**按需拉取**——每个用户块只需要它实际评分过的物品对应的 $V$ 块


**DAG 分析要点**（报告截图素材）：
1. `randomSplit` 产生随机分区，依赖类型为 NarrowDependency
2. `ALS.fit` 内部有迭代的交替最小二乘，Job 边界在每轮迭代处
3. `recommendForAllUsers` 触发 `flatMap` + `groupByKey`（Shuffle 密集型操作）
4. 从 Spark UI → Stages 页面观察各 Stage 的 Input Size、Shuffle Read/Write 量

### 7.2 冷启动问题量化分析

在报告中加入以下分析：

```sql
-- 在 pyspark 中执行，统计不同评分数量段的用户分布
SELECT
    CASE
        WHEN cnt < 5   THEN 'cold (<5)'
        WHEN cnt < 20  THEN 'warm (5-19)'
        WHEN cnt < 50  THEN 'hot (20-49)'
        ELSE                'very hot (50+)'
    END AS user_type,
    COUNT(*) AS user_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct
FROM (SELECT userId, COUNT(*) AS cnt FROM ratings GROUP BY userId)
GROUP BY user_type
ORDER BY user_type;
```


**讨论内容**：
- 冷启动用户在新系统上线时无法获得个性化推荐
- 短期方案：热门推荐（全局评分 Top-N）、基于用户画像（年龄/性别）的统计推荐
- 长期方案：基于内容的推荐（利用电影类型/标题文本）、混合推荐（Hybrid）、新用户冷启动引导问卷

### 7.3 模型评估与推荐质量分析

**定量评估**：

| 指标 | 含义 | 本实验参考值 |
|------|------|-------------|
| RMSE | 预测评分与真实评分的均方根误差（回归指标） | 0.83~0.90 |
| Precision@10 | Top-10 推荐中命中用户实际高评分电影的比例 | 训练脚本会输出 |
| Recall@10 | 用户高评分电影中被 Top-10 推荐覆盖的比例 | 训练脚本会输出 |
| NDCG@10 | 归一化折损累计增益——考虑排序位置质量的综合指标 | 训练脚本会输出 |
| Coverage | 被推荐到的电影占全部电影的比例 | 越高说明推荐多样性越好 |

**定性评估**：挑选 3~5 个用户，展示其历史高评分电影和 Top-10 推荐电影，人工判断推荐是否合理。

### 7.4 Spark UI 执行计划解读（报告截图指引）

1. **Jobs 页面**：展示每个 Action（如 `count()`、`write.csv`）触发的 Job 列表
2. **Stages 页面**：重点标注 Shuffle Read/Write 量较大的 Stage，说明数据倾斜风险
3. **Storage 页面**：确认 `ratings.cache()` 是否成功缓存到内存（Storage Level: MEMORY_AND_DISK）
4. **Executors 页面**：观察各 Executor 的任务分配是否均衡

### 7.5 数据倾斜分析与优化思路

- **现象**：某几个 Executor 的 Shuffle Read 远超其他（在 Stage 详情页可见）
- **原因**：热门电影（如《肖申克的救赎》）被大量用户评分，导致按 movieId 分组时数据倾斜
- **优化方案**：加盐（Salt）打散热点 key、增大 `spark.sql.shuffle.partitions`、使用 broadcast join 代替 shuffle join

---

## 八、常见问题与排错

| 错误现象 | 原因 | 解决办法 |
|---------|------|----------|
| `java.lang.NoClassDefFoundError` | Java 版本不兼容 | `sudo update-alternatives --config java` 选择 Java 8 |
| `OutOfMemoryError: Java heap space` | Driver 内存不足 | 增加 `--driver_memory 4g` 或使用 ml-1m 数据集 |
| `recommendForAllUsers` 执行过慢 / OOM | 用户或物品数量过多 | 减小 `top_n`；或对用户随机采样先验证流程 |
| 读取 ratings.dat 列数不对 | 分隔符 `::` 处理不正确 | 确认使用 RDD `split("::")` 而非 CSV reader |
| Flask 无法从宿主机访问 | 防火墙拦截或监听地址不对 | `host='0.0.0.0'` + `sudo ufw allow 5000` |
| `part-*.csv` 文件加载报错 | 目录中有 `_SUCCESS`、`.crc` 等杂文件 | 过滤只读取 `part-*.csv`（已在 app.py 中处理） |

---

## 九、扩展建议（加分项）

| 难度 | 扩展内容 | 说明 |
|------|---------|------|
| ★☆☆ | HTML 前端页面 | 用纯 HTML+JS 做一个简单的电影推荐展示页 |
| ★★☆ | 超参数网格搜索 | 见 4.3 节代码，搜索 rank/regParam/maxIter 最优组合 |
| ★★☆ | 基于内容的推荐 | 利用电影类型和标题 TF-IDF 计算物品相似度，解决冷启动 |
| ★★★ | 混合推荐（Hybrid） | 加权融合 ALS 协同过滤 + Content-Based 的结果 |
| ★★★ | Lambda 架构实时层 | 接入 Kafka 消费用户实时行为，用 Redis 存储实时特征 |
| ★★★ | 模型 A/B 测试框架 | 多版本模型同时在线，按用户 ID hash 分流，对比点击率 |

---

## 十、参考资料

1. [MovieLens 数据集官方页面](https://grouplens.org/datasets/movielens/)
2. [Spark MLlib ALS 官方文档](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)
3. Zhou et al. "Large-scale Parallel Collaborative Filtering for the Netflix Prize" (ALS 经典论文)
4. [Spark 调优指南](https://spark.apache.org/docs/latest/tuning.html)
