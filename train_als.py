"""
离线 ALS 训练脚本
支持 MovieLens 1M（:: 分隔）和 25M（逗号分隔）数据集，自动检测格式。

用法:
    source venv/bin/activate
    python3 train_als.py --data_dir data/ml-1m --output_dir output --rank 50 --max_iter 15
    python3 train_als.py --data_dir data/ml-25m --output_dir output --rank 50 --max_iter 15 --driver_memory 4g
"""
import argparse
import logging
import os
import sys
import time

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, explode, count, collect_list
from pyspark.mllib.evaluation import RankingMetrics

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# 命令行参数
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spark ALS 电影推荐训练")
    parser.add_argument("--data_dir", default="data/ml-1m", help="MovieLens 数据集目录")
    parser.add_argument("--output_dir", default="output", help="输出目录")
    parser.add_argument("--rank", type=int, default=50, help="ALS 隐因子维度")
    parser.add_argument("--max_iter", type=int, default=15, help="ALS 最大迭代次数")
    parser.add_argument("--reg_param", type=float, default=0.1, help="ALS 正则化参数")
    parser.add_argument("--top_n", type=int, default=10, help="为每个用户推荐 Top-N")
    parser.add_argument("--driver_memory", default="2g", help="Spark driver 内存")
    return parser.parse_args()


# ============================================================
# Spark 初始化
# ============================================================

def init_spark(driver_memory: str) -> SparkSession:
    return (
        SparkSession.builder.appName("MovieRec_ALS")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.default.parallelism", "16")
        .getOrCreate()
    )


# ============================================================
# 数据加载
# ============================================================

def detect_format(data_dir: str) -> tuple[str, str]:
    """根据数据集路径自动检测分隔符和编码。

    MovieLens 1M: :: 分隔, ISO-8859-1 编码（法语/德语重音字符）
    MovieLens 25M: , 分隔, UTF-8 编码
    """
    if "25m" in data_dir.lower():
        return ",", "UTF-8"
    return "::", "ISO-8859-1"


def load_dat_file(
    spark: SparkSession,
    filepath: str,
    columns: list[str],
    delimiter: str,
    encoding: str = "UTF-8",
) -> DataFrame:
    """通用数据加载：读取编码文本文件，按分隔符解析 → Spark DataFrame"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在: {filepath}")

    return (
        spark.read.option("charset", encoding).text(filepath)
        .rdd.map(lambda row: row[0].split(delimiter))
        .filter(lambda parts: len(parts) >= len(columns))
        .map(lambda parts: tuple(parts[: len(columns)]))
        .toDF(columns)
    )


def load_ratings(spark: SparkSession, data_dir: str) -> DataFrame:
    """读取评分数据 → (userId, movieId, rating) DataFrame"""
    delimiter, encoding = detect_format(data_dir)
    for fname in ("ratings.dat", "ratings.csv"):
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            logger.info(f"读取评分数据: {path} (分隔符='{delimiter}', 编码={encoding})")
            df = load_dat_file(spark, path, ["userId", "movieId", "rating"], delimiter, encoding)
            break
    else:
        raise FileNotFoundError(f"在 {data_dir} 中未找到 ratings.dat 或 ratings.csv")

    df = df.select(
        col("userId").cast("int"),
        col("movieId").cast("int"),
        col("rating").cast("float"),
    )
    df.cache()
    n_users = df.select("userId").distinct().count()
    n_items = df.select("movieId").distinct().count()
    n_ratings = df.count()
    logger.info(f"数据统计: {n_users} 用户, {n_items} 电影, {n_ratings} 条评分")
    return df


def load_movies(spark: SparkSession, data_dir: str) -> DataFrame:
    """读取电影元数据 → (movieId, title, genres) DataFrame"""
    delimiter, encoding = detect_format(data_dir)
    for fname in ("movies.dat", "movies.csv"):
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            logger.info(f"读取电影数据: {path} (分隔符='{delimiter}', 编码={encoding})")
            df = load_dat_file(spark, path, ["movieId", "title", "genres"], delimiter, encoding)
            break
    else:
        raise FileNotFoundError(f"在 {data_dir} 中未找到 movies.dat 或 movies.csv")

    return df.select(col("movieId").cast("int"), "title", "genres")


# ============================================================
# 冷启动分析
# ============================================================

def analyze_cold_start(ratings: DataFrame) -> tuple[int, int]:
    """统计评分 <5 条的用户占比"""
    user_cnt = ratings.groupBy("userId").agg(count("rating").alias("cnt"))
    user_cnt.cache()
    total_users = user_cnt.count()
    n_cold = user_cnt.filter(col("cnt") < 5).count()
    pct = n_cold / max(total_users, 1) * 100
    logger.info(f"冷启动用户（<5条评分）: {n_cold}/{total_users} = {pct:.1f}%")
    user_cnt.unpersist()
    return n_cold, total_users


# ============================================================
# ALS 训练
# ============================================================

def train_als(train: DataFrame, args: argparse.Namespace) -> ALSModel:
    """训练 ALS 矩阵分解模型"""
    logger.info(
        f"ALS 参数: rank={args.rank}, maxIter={args.max_iter}, "
        f"regParam={args.reg_param}"
    )
    als = ALS(
        maxIter=args.max_iter,
        regParam=args.reg_param,
        rank=args.rank,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
        seed=42,
        nonnegative=True,
        implicitPrefs=False,
    )
    t0 = time.time()
    model = als.fit(train)
    elapsed = time.time() - t0
    logger.info(f"ALS 训练完成，耗时 {elapsed:.1f}s")
    return model


# ============================================================
# 模型评估
# ============================================================

def evaluate_model(model: ALSModel, test: DataFrame) -> float:
    """计算测试集 RMSE"""
    predictions = model.transform(test)
    total = test.count()
    predicted = predictions.count()
    dropped = total - predicted
    if dropped > 0:
        logger.warning(
            f"coldStartStrategy='drop' 丢弃了 {dropped}/{total} 条记录 "
            f"({dropped / total * 100:.1f}%)"
        )
    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    logger.info(f"测试集 RMSE = {rmse:.4f}（基于 {predicted} 条有效预测）")
    return rmse


def compute_ranking_metrics(
    model: ALSModel, test: DataFrame, k: int = 10
) -> tuple[float, float, float]:
    """计算 Precision@K、Recall@K、NDCG@K"""
    test_users = test.select("userId").distinct()
    user_recs = model.recommendForUserSubset(test_users, k)

    # 正样本：评分 ≥ 3.5 的电影
    actual = (
        test.filter(col("rating") >= 3.5)
        .groupBy("userId")
        .agg(collect_list("movieId").alias("actual_items"))
    )

    pred = user_recs.select(
        "userId", col("recommendations.movieId").alias("pred_items")
    )

    joined = pred.join(actual, "userId", "inner")
    pred_actual_rdd = joined.select("pred_items", "actual_items").rdd.map(
        lambda r: (list(map(int, r.pred_items)), list(map(int, r.actual_items)))
    )

    metrics = RankingMetrics(pred_actual_rdd)
    p = metrics.precisionAt(k)
    r = metrics.recallAt(k)
    ndcg = metrics.ndcgAt(k)
    logger.info(f"Precision@{k}={p:.4f}  Recall@{k}={r:.4f}  NDCG@{k}={ndcg:.4f}")
    return p, r, ndcg


# ============================================================
# 推荐生成 & 输出
# ============================================================

def generate_recommendations(
    model: ALSModel, top_n: int
) -> DataFrame:
    """为所有用户生成 Top-N 推荐，展开嵌套为扁平表"""
    logger.info(f"为所有用户生成 Top-{top_n} 推荐...")
    t0 = time.time()
    user_recs = model.recommendForAllUsers(top_n)

    flat = user_recs.select(
        "userId", explode("recommendations").alias("rec")
    ).select(
        "userId",
        col("rec.movieId").alias("movieId"),
        col("rec.rating").alias("predRating"),
    )
    elapsed = time.time() - t0
    logger.info(f"推荐生成完成，耗时 {elapsed:.1f}s")
    return flat


def save_outputs(
    user_recs_flat: DataFrame,
    movies: DataFrame,
    model: ALSModel,
    output_dir: str,
) -> None:
    """持久化推荐结果、电影映射、ALS 模型"""
    # 推荐结果（分布式写入，多 part 文件）
    recs_path = os.path.join(output_dir, "user_recs")
    logger.info(f"保存推荐结果到 {recs_path}")
    user_recs_flat.write.csv(recs_path, header=True, mode="overwrite")

    # 电影映射（数据量小，合并为单文件）
    movies_path = os.path.join(output_dir, "movies")
    logger.info(f"保存电影映射到 {movies_path}")
    movies.coalesce(1).write.csv(movies_path, header=True, mode="overwrite")

    # ALS 模型（Parquet 格式）
    model_path = os.path.join(output_dir, "als_model")
    logger.info(f"持久化 ALS 模型到 {model_path}")
    model.write().overwrite().save(model_path)


# ============================================================
# 主流程
# ============================================================

def main() -> None:
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

        # 6. 生成全量推荐
        user_recs_flat = generate_recommendations(model, args.top_n)

        # 7. 保存输出
        save_outputs(user_recs_flat, movies, model, args.output_dir)

        logger.info("✅ 全部流程完成！")
    except Exception:
        logger.error("训练失败", exc_info=True)
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
