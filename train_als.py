"""
离线 ALS 训练脚本（跨平台：Windows / Linux / macOS）
支持 MovieLens 1M（:: 分隔）和 25M（逗号分隔）数据集，自动检测格式。

用法:
    # Linux / macOS
    #   source venv/bin/activate
    #   python3 train_als.py --data_dir data/ml-1m --output_dir output --rank 50 --max_iter 15
    #
    # Windows
    #   venv\\Scripts\\activate
    #   python train_als.py --data_dir data/ml-1m --output_dir output --rank 50 --max_iter 15
    #   python train_als.py --data_dir data/ml-25m --output_dir output --rank 50 --max_iter 15 --driver_memory 4g
"""
import argparse
import logging
import os
import sys
import time

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, explode, count, collect_list, collect_set, row_number, desc, udf, concat_ws, split
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window
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
    """初始化 Spark Session，自动适配 Windows / Linux 平台"""
    builder = (
        SparkSession.builder.appName("MovieRec_ALS")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.default.parallelism", "16")
        .config("spark.driver.bindAddress", "127.0.0.1")
    )

    if sys.platform == "win32":
        # Windows: Hadoop 默认使用 /tmp 和 POSIX 权限，需要重定向到合法路径
        import tempfile
        win_tmp = "file:///" + tempfile.gettempdir().replace("\\", "/").lstrip("/")
        builder = (
            builder
            .config("spark.sql.warehouse.dir", win_tmp + "/spark-warehouse")
            .config("spark.local.dir", win_tmp + "/spark-tmp")
            .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        )
        logger.info("检测到 Windows 平台，已配置 Hadoop 兼容路径")

    return builder.getOrCreate()


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
# 基于内容的冷启动推荐
# ============================================================

_jaccard = udf(
    lambda a, b: len(set(a.split("|")) & set(b.split("|"))) / max(len(set(a.split("|")) | set(b.split("|"))), 1)
    if a and b else 0.0,
    DoubleType(),
)


def recommend_content_based(
    ratings: DataFrame, movies: DataFrame, top_n: int, cold_threshold: int = 5
) -> DataFrame:
    """为冷启动用户（评分 < cold_threshold 条）生成基于电影类型的推荐。

    提取用户所有已评分电影的类型合集作为"口味画像"，
    计算与每部电影类型的 Jaccard 相似度，排除已评分后取 Top-N。
    """
    cold_users = (
        ratings.groupBy("userId").agg(count("rating").alias("cnt"))
        .filter(col("cnt") < cold_threshold)
        .select("userId")
    )
    cold_count = cold_users.count()
    if cold_count == 0:
        logger.info("无冷启动用户，跳过内容推荐")
        return None

    logger.info(f"为 {cold_count} 个冷启动用户生成基于内容的推荐...")
    t0 = time.time()

    # 构建用户口味画像：收集用户所有已评分电影的类型去重集合
    user_profile = (
        ratings.join(cold_users, "userId")
        .join(movies.select("movieId", "genres"), "movieId")
        .withColumn("genre", explode(split(col("genres"), "\\|")))
        .groupBy("userId")
        .agg(concat_ws("|", collect_set("genre")).alias("user_genres"))
    )

    # Cross join → 计算 Jaccard
    cross = user_profile.crossJoin(
        movies.select("movieId", col("genres").alias("movie_genres"))
    )
    scored = cross.withColumn("jaccard", _jaccard(col("user_genres"), col("movie_genres")))

    # 排除已评分电影
    filtered = scored.join(
        ratings.select("userId", "movieId").distinct(),
        on=["userId", "movieId"], how="left_anti",
    )

    # 按 Jaccard 降序取 Top-N
    window = Window.partitionBy("userId").orderBy(desc("jaccard"))
    result = (
        filtered.withColumn("rn", row_number().over(window))
        .filter(col("rn") <= top_n)
        .select("userId", "movieId", col("jaccard").alias("predRating"))
    )

    elapsed = time.time() - t0
    logger.info(f"内容推荐完成，耗时 {elapsed:.1f}s，共 {result.count()} 条")
    return result


# ============================================================
# 推荐生成 & 输出
# ============================================================

def generate_recommendations(
    model: ALSModel, ratings: DataFrame, top_n: int
) -> DataFrame:
    """为所有用户生成 Top-N 推荐，排除已评分电影。

    先多取一些候选（top_n * 3），过滤掉已评分电影后截断到 top_n，
    确保即使用户看过较多电影也不会推荐不足。
    """
    logger.info(f"为所有用户生成 Top-{top_n} 推荐（排除已评分电影）...")
    t0 = time.time()

    # 用户已评分的电影（去重）
    rated = ratings.select("userId", "movieId").distinct()

    # 多取候选，为过滤留余量
    fetch_n = top_n * 3
    user_recs = model.recommendForAllUsers(fetch_n)

    flat = user_recs.select(
        "userId", explode("recommendations").alias("rec")
    ).select(
        "userId",
        col("rec.movieId").alias("movieId"),
        col("rec.rating").alias("predRating"),
    )

    # 排除已评分电影（left anti join）
    filtered = flat.join(rated, on=["userId", "movieId"], how="left_anti")

    # 按预测评分降序，每用户截断到 top_n
    window = Window.partitionBy("userId").orderBy(desc("predRating"))
    result = (
        filtered
        .withColumn("rn", row_number().over(window))
        .filter(col("rn") <= top_n)
        .drop("rn")
    )

    elapsed = time.time() - t0
    logger.info(f"推荐生成完成，耗时 {elapsed:.1f}s")
    return result


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
        user_recs_flat = generate_recommendations(model, ratings, args.top_n)

        # 6b. 冷启动用户替换为内容推荐
        cold_recs = recommend_content_based(ratings, movies, args.top_n)
        if cold_recs is not None:
            cold_ids = cold_recs.select("userId").distinct()
            user_recs_flat = user_recs_flat.join(cold_ids, "userId", "left_anti").unionByName(cold_recs)

        # 7. 保存输出
        save_outputs(user_recs_flat, movies, model, args.output_dir)

        logger.info("全部流程完成!")
    except Exception:
        logger.error("训练失败", exc_info=True)
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
