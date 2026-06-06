"""
离线 ALS 训练脚本（跨平台：Windows / Linux / macOS）
支持 MovieLens 1M（:: 分隔）和 25M（逗号分隔）数据集，自动检测格式。
冷启动用户自动使用基于内容的推荐，评估包含 RMSE/P@K/R@K/NDCG@K/Coverage/Diversity。

用法:
    # 默认参数训练
    python train_als.py --data_dir data/ml-1m --output_dir output --rank 50 --max_iter 15

    # 超参数网格搜索
    python train_als.py --data_dir data/ml-1m --tune

    # 25M 数据集
    python train_als.py --data_dir data/ml-25m --driver_memory 4g
"""
import argparse
import logging
import os
import sys
import time

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, explode, count, collect_list, collect_set, row_number, desc, udf, concat_ws, split, avg, min as spark_min, max as spark_max, when
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
    parser.add_argument("--tune", action="store_true", help="启用超参数网格搜索（CrossValidator）")
    parser.add_argument("--hybrid", action="store_true", help="启用混合推荐（ALS + 内容相似度加权融合）")
    parser.add_argument("--alpha", type=float, default=0.7, help="混合推荐中 ALS 权重（0-1），默认 0.7")
    return parser.parse_args()


# ============================================================
# Spark 初始化
# ============================================================

def init_spark(driver_memory: str) -> SparkSession:
    """初始化 Spark Session，自动适配 Windows / Linux 平台 + Java 17+"""
    # 确保 PySpark 使用当前虚拟环境的 Python（避免 Python worker 连接超时）
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    # Java 17+ 模块系统兼容：Spark 3.4.1 需要开放内部模块访问
    jvm_opens = (
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.util=ALL-UNNAMED "
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
        "--add-opens=java.base/javax.security.auth=ALL-UNNAMED"
    )

    # Java 17+ 绕过 Hadoop UserGroupInformation.getSubject() 调用
    if "HADOOP_USER_NAME" not in os.environ:
        os.environ["HADOOP_USER_NAME"] = os.environ.get("USERNAME", os.environ.get("USER", "hadoop"))

    builder = (
        SparkSession.builder.appName("MovieRec_ALS")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.default.parallelism", "16")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.extraJavaOptions", jvm_opens)
        .config("spark.executor.extraJavaOptions", jvm_opens)
    )

    if sys.platform == "win32":
        # Windows: Hadoop 默认使用 /tmp 和 POSIX 权限，需要重定向到合法路径
        import tempfile
        win_tmp = tempfile.gettempdir()
        builder = (
            builder
            .config("spark.sql.warehouse.dir", "file:///" + win_tmp.replace("\\", "/") + "/spark-warehouse")
            .config("spark.local.dir", os.path.join(win_tmp, "spark-tmp"))
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


def tune_hyperparams(train: DataFrame, folds: int = 3) -> ALSModel:
    """使用交叉验证搜索最优 ALS 超参数。

    搜索空间: rank ∈ {10, 30, 50}, regParam ∈ {0.01, 0.1, 0.5}, maxIter ∈ {10, 15}
    评估指标: RMSE
    """
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

    als = ALS(
        userCol="userId", itemCol="movieId", ratingCol="rating",
        coldStartStrategy="drop", seed=42, nonnegative=True, implicitPrefs=False,
    )

    param_grid = (
        ParamGridBuilder()
        .addGrid(als.rank, [10, 30, 50])
        .addGrid(als.regParam, [0.01, 0.1, 0.5])
        .addGrid(als.maxIter, [10, 15])
        .build()
    )

    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction"
    )

    cv = CrossValidator(
        estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator,
        numFolds=folds, seed=42, parallelism=4,
    )

    logger.info(f"开始超参数网格搜索（{len(param_grid)} 组 × {folds} 折）...")
    t0 = time.time()
    cv_model = cv.fit(train)
    elapsed = time.time() - t0

    best = cv_model.bestModel
    best_reg = best._java_obj.parent().getRegParam()
    best_iter = best._java_obj.parent().getMaxIter()
    logger.info(f"网格搜索完成，耗时 {elapsed:.1f}s")
    logger.info(f"最佳参数: rank={best.rank}, regParam={best_reg:.4f}, maxIter={best_iter}")

    # 打印所有参数组合的平均 RMSE
    logger.info("参数组合评估结果:")
    for i, params in enumerate(cv_model.getEstimatorParamMaps()):
        logger.info(
            f"  [{i+1}/{len(param_grid)}] rank={params[als.rank]}, "
            f"regParam={params[als.regParam]}, maxIter={params[als.maxIter]} → "
            f"avg RMSE={cv_model.avgMetrics[i]:.4f}"
        )

    return best


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


def compute_coverage_diversity(
    recs: DataFrame, movies: DataFrame, top_n: int
) -> tuple[float, float]:
    """计算推荐覆盖率与列表内多样性。

    Coverage: 被推荐到的不同电影数 / 总电影数（越高越好，范围 0-100%）
    Intra-list Diversity: 每个用户推荐列表中所有电影对的平均 (1 - Jaccard)，
                          越高表示推荐列表内电影类型差异越大（范围 0-1）
    """
    total_movies = movies.count()
    recommended_movies = recs.select("movieId").distinct().count()
    coverage = recommended_movies / max(total_movies, 1) * 100

    # 推荐列表去重（取每用户 top_n 条，已由上游保证）
    recs_genre = recs.join(
        movies.select("movieId", col("genres").alias("g1")), "movieId"
    )

    # Self-join：每用户推荐列表内所有 movieId_a < movieId_b 的电影对
    pairs = (
        recs_genre.alias("a")
        .join(recs_genre.alias("b"), on="userId")
        .filter(col("a.movieId") < col("b.movieId"))
    )

    # 每对电影的 Jaccard 距离 = 1 - Jaccard 相似度
    pairs_with_dist = pairs.withColumn(
        "jaccard_dist", 1.0 - _jaccard(col("a.g1"), col("b.g1"))
    )

    row = pairs_with_dist.select(avg("jaccard_dist")).first()
    diversity = row[0] if row and row[0] is not None else 0.0

    logger.info(f"Coverage = {coverage:.2f}%  Intra-list Diversity = {diversity:.4f}")
    return coverage, diversity


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


def recommend_hybrid(
    model: ALSModel, ratings: DataFrame, movies: DataFrame, top_n: int, alpha: float = 0.7
) -> DataFrame:
    """混合推荐：在 ALS 候选池内按内容相似度加权重排序。

    1. 取 ALS top_n*5 候选池（排除已评分）
    2. 为每部候选电影计算与用户口味画像的 Jaccard 相似度
    3. 两个分数各自 min-max 归一化到 [0,1]
    4. hybrid_score = α×norm_als + (1-α)×norm_content
    5. 重新排序取 top_n

    冷启动用户（评分 <5 条）跳过混合，仍用纯内容推荐。
    """
    rated = ratings.select("userId", "movieId").distinct()
    fetch_n = top_n * 5

    # 区分暖/冷用户
    user_cnt = ratings.groupBy("userId").agg(count("rating").alias("cnt")).cache()
    cold_ids = user_cnt.filter(col("cnt") < 5).select("userId")
    warm_ids = user_cnt.filter(col("cnt") >= 5).select("userId")
    cold_count = cold_ids.count()
    warm_count = warm_ids.count()
    user_cnt.unpersist()
    logger.info(f"混合推荐: {warm_count} 暖用户 (ALS+Content), {cold_count} 冷用户 (纯Content)")

    # 1. 暖用户的 ALS 候选池
    als_pool = (
        model.recommendForAllUsers(fetch_n)
        .select("userId", explode("recommendations").alias("rec"))
        .select(
            "userId",
            col("rec.movieId").alias("movieId"),
            col("rec.rating").alias("als_score"),
        )
        .join(warm_ids, "userId")
        .join(rated, ["userId", "movieId"], "left_anti")
    )

    # 2. 用户口味画像（所有用户）
    t0 = time.time()
    user_profile = (
        ratings.join(movies.select("movieId", "genres"), "movieId")
        .withColumn("genre", explode(split(col("genres"), "\\|")))
        .groupBy("userId")
        .agg(concat_ws("|", collect_set("genre")).alias("user_genres"))
    )

    # 3. 为 ALS 候选池附加内容分数
    pool = (
        als_pool
        .join(user_profile, "userId")
        .join(movies.select("movieId", col("genres").alias("m_genres")), "movieId")
        .withColumn("content_score", _jaccard(col("user_genres"), col("m_genres")))
        .drop("user_genres", "m_genres")
    )

    # 4. 每用户内 min-max 归一化 + 加权
    stats = pool.groupBy("userId").agg(
        spark_min("als_score").alias("als_min"), spark_max("als_score").alias("als_max"),
        spark_min("content_score").alias("c_min"), spark_max("content_score").alias("c_max"),
    )
    normed = pool.join(stats, "userId").withColumn(
        "hybrid_score",
        alpha * (col("als_score") - col("als_min")) / (col("als_max") - col("als_min") + 1e-9)
        + (1 - alpha) * (col("content_score") - col("c_min")) / (col("c_max") - col("c_min") + 1e-9),
    )

    # 5. 重新排序 → 暖用户混合推荐结果
    window = Window.partitionBy("userId").orderBy(desc("hybrid_score"))
    warm_recs = (
        normed.withColumn("rn", row_number().over(window))
        .filter(col("rn") <= top_n)
        .select("userId", "movieId", col("hybrid_score").alias("predRating"))
    )

    # 6. 冷启动用户使用纯内容推荐
    if cold_count > 0:
        cold_recs = recommend_content_based(ratings, movies, top_n)
        result = warm_recs.unionByName(cold_recs) if cold_recs is not None else warm_recs
    else:
        result = warm_recs

    elapsed = time.time() - t0
    logger.info(f"混合推荐完成，耗时 {elapsed:.1f}s（α={alpha}）")
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

        # 4. 训练 ALS（可选超参数调优）
        if args.tune:
            model = tune_hyperparams(train)
        else:
            model = train_als(train, args)

        # 5. 评估
        evaluate_model(model, test)
        compute_ranking_metrics(model, test, k=args.top_n)

        # 6. 生成全量推荐
        if args.hybrid:
            user_recs_flat = recommend_hybrid(model, ratings, movies, args.top_n, args.alpha)
        else:
            user_recs_flat = generate_recommendations(model, ratings, args.top_n)
            # 冷启动用户替换为内容推荐
            cold_recs = recommend_content_based(ratings, movies, args.top_n)
            if cold_recs is not None:
                cold_ids = cold_recs.select("userId").distinct()
                user_recs_flat = user_recs_flat.join(cold_ids, "userId", "left_anti").unionByName(cold_recs)

        # 6b. Coverage / Diversity 评估
        compute_coverage_diversity(user_recs_flat, movies, args.top_n)

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
