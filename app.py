"""
推荐系统 REST API 服务（跨平台：Windows / Linux / macOS）
启动时加载离线训练产出的 CSV 到内存，提供 kv 查询接口。

推荐结果已自动排除用户历史评分的电影，确保推荐内容的发现性。

用法:
    # 默认启动（内存模式，仅控制台日志）
    python app.py --port 5000

    # SQLite 模式（适合 25M 大数据集，无需全量加载到内存）
    python app.py --port 5000 --db output/recommender.db

    # 启用文件日志（控制台 + 文件，10MB 自动轮转）
    python app.py --port 5000 --log_file logs/api.log
"""
import argparse
import json
import logging
import os
import sqlite3
import time
from logging.handlers import RotatingFileHandler

import pandas as pd
from flask import Flask, jsonify, request, g, send_from_directory

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------- 全局缓存（启动时加载）----------
_rec_dict: dict[int, list[int]] = {}
_movie_titles: dict[int, str] = {}
_movie_genres: dict[int, str] = {}
_db_path: str | None = None  # SQLite 数据库路径，None 表示使用内存模式


# ============================================================
# 数据加载
# ============================================================

def load_csv_dir(dir_path: str) -> pd.DataFrame:
    """加载 Spark 输出的 CSV 目录（合并所有 part-*.csv）"""
    frames = []
    for fname in sorted(os.listdir(dir_path)):
        if fname.startswith("part-") and fname.endswith(".csv"):
            filepath = os.path.join(dir_path, fname)
            # Spark CSV 默认用反斜杠转义引号（\"），而 pandas 默认按 RFC 双引号（""）解析。
            # MovieLens 25M 部分标题含逗号/引号，必须用 escapechar='\\' 才能正确解析。
            frames.append(pd.read_csv(filepath, escapechar="\\"))
    if not frames:
        raise FileNotFoundError(f"在 {dir_path} 中未找到 part-*.csv 文件，请先运行 train_als.py")
    return pd.concat(frames, ignore_index=True)


def load_recs(recs_dir: str) -> dict[int, list[int]]:
    """加载推荐结果 → {userId: [movieId, ...]}，按评分降序"""
    df = load_csv_dir(recs_dir)
    df_sorted = df.sort_values(
        ["userId", "predRating"], ascending=[True, False]
    )
    return df_sorted.groupby("userId")["movieId"].apply(list).to_dict()


def load_movies(movies_dir: str) -> tuple[dict[int, str], dict[int, str]]:
    """加载电影元数据 → ({movieId: title}, {movieId: genres})"""
    df = load_csv_dir(movies_dir)
    titles = dict(zip(df["movieId"], df["title"]))
    genres = dict(zip(df["movieId"], df["genres"]))
    return titles, genres


def init_data(recs_dir: str, movies_dir: str, db_path: str | None = None) -> None:
    """启动时加载数据：内存模式（db_path=None）或 SQLite 模式"""
    global _rec_dict, _movie_titles, _movie_genres, _db_path
    _db_path = db_path

    if db_path is None:
        # 内存模式
        logger.info(f"加载推荐数据: {recs_dir}")
        _rec_dict = load_recs(recs_dir)
        logger.info(f"加载电影数据: {movies_dir}")
        _movie_titles, _movie_genres = load_movies(movies_dir)
        logger.info(f"加载完成: {len(_rec_dict)} 用户, {len(_movie_titles)} 电影 (内存模式)")
    else:
        # SQLite 模式
        logger.info(f"加载数据到 SQLite: {db_path}")
        _init_sqlite(db_path, recs_dir, movies_dir)
        counts = _query_counts_sqlite()
        logger.info(f"加载完成: {counts[0]} 用户, {counts[1]} 电影 (SQLite 模式)")


def _init_sqlite(db_path: str, recs_dir: str, movies_dir: str) -> None:
    """创建 SQLite 表并从 CSV 导入数据（仅首次）"""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            userId INTEGER, movieId INTEGER, predRating REAL,
            PRIMARY KEY (userId, movieId)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS movies (
            movieId INTEGER PRIMARY KEY, title TEXT, genres TEXT
        )
    """)

    # 仅在表为空时导入
    if conn.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0] == 0:
        logger.info("导入推荐数据到 SQLite...")
        recs_df = load_csv_dir(recs_dir)
        recs_df[["userId", "movieId", "predRating"]].to_sql(
            "recommendations", conn, if_exists="replace", index=False
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_recs_user ON recommendations(userId)")

    if conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0] == 0:
        logger.info("导入电影数据到 SQLite...")
        movies_df = load_csv_dir(movies_dir)
        movies_df.to_sql("movies", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()


def _query_recs(user_id: int, limit: int) -> list[dict]:
    """查询用户推荐结果（内存或 SQLite）"""
    if _db_path is None:
        if user_id not in _rec_dict:
            return []
        movie_ids = _rec_dict[user_id][:limit]
        return [
            {"movieId": int(mid), "title": _movie_titles.get(int(mid), "Unknown"),
             "genres": _movie_genres.get(int(mid), "Unknown")}
            for mid in movie_ids
        ]
    else:
        conn = sqlite3.connect(_db_path)
        rows = conn.execute(
            "SELECT r.movieId, m.title, m.genres FROM recommendations r "
            "JOIN movies m ON r.movieId = m.movieId "
            "WHERE r.userId = ? ORDER BY r.predRating DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        conn.close()
        return [{"movieId": r[0], "title": r[1], "genres": r[2]} for r in rows]


def _query_movie(movie_id: int) -> dict | None:
    """查询单部电影信息（内存或 SQLite）"""
    if _db_path is None:
        title = _movie_titles.get(movie_id)
        if title is None:
            return None
        return {
            "movieId": movie_id, "title": title,
            "genres": _movie_genres.get(movie_id, "Unknown"),
        }
    else:
        conn = sqlite3.connect(_db_path)
        row = conn.execute(
            "SELECT movieId, title, genres FROM movies WHERE movieId = ?",
            (movie_id,),
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return {"movieId": row[0], "title": row[1], "genres": row[2]}


def _query_counts_sqlite() -> tuple[int, int]:
    """查询 SQLite 中的用户数和电影数"""
    conn = sqlite3.connect(_db_path)
    n_users = conn.execute("SELECT COUNT(DISTINCT userId) FROM recommendations").fetchone()[0]
    n_movies = conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
    conn.close()
    return n_users, n_movies


# ============================================================
# 中间件
# ============================================================

def _extract_log_context(response) -> str:
    """从响应中提取业务上下文（userId / movieId / 推荐条数）"""
    if response.status_code >= 400:
        return ""
    if not response.content_type or "json" not in response.content_type:
        return ""
    try:
        data = json.loads(response.get_data(as_text=True))
    except (json.JSONDecodeError, TypeError, RuntimeError):
        return ""

    parts = []
    if "user_id" in data:
        parts.append(f"userId={data['user_id']}")
    if "recommendations" in data:
        parts.append(f"n={len(data['recommendations'])}")
    if "movieId" in data and "user_id" not in data:
        parts.append(f"movieId={data['movieId']}")
    if "users" in data and "movies" in data:
        parts.append(f"users={data['users']} movies={data['movies']}")
    return "  ".join(parts)


@app.before_request
def before_request() -> None:
    g.start_time = time.time()


@app.after_request
def after_request(response):
    """结构化请求日志：method、path、status、耗时 + 业务上下文"""
    elapsed = (time.time() - g.start_time) * 1000
    ctx = _extract_log_context(response)
    entry = f"{request.method} {request.path} → {response.status_code} ({elapsed:.1f}ms)"
    if ctx:
        entry += f"  [{ctx}]"
    logger.info(entry)
    return response


# ============================================================
# 路由
# ============================================================

@app.route("/")
def home():
    """返回前端推荐展示页面"""
    return send_from_directory("static", "index.html")


@app.route("/recommend/<int:user_id>")
def recommend(user_id: int):
    """为指定用户返回个性化推荐列表"""
    limit = request.args.get("limit", 10, type=int)
    limit = max(1, min(limit, 50))

    rec_list = _query_recs(user_id, limit)
    if not rec_list:
        return jsonify({"error": f"User {user_id} not found"}), 404

    return jsonify({"user_id": user_id, "recommendations": rec_list})


@app.route("/movie/<int:movie_id>")
def movie_info(movie_id: int):
    """查询单部电影元信息"""
    data = _query_movie(movie_id)
    if data is None:
        return jsonify({"error": f"Movie {movie_id} not found"}), 404
    return jsonify(data)


@app.route("/health")
def health():
    """服务健康检查"""
    if _db_path is None:
        n_users, n_movies = len(_rec_dict), len(_movie_titles)
    else:
        n_users, n_movies = _query_counts_sqlite()
    return jsonify({"status": "ok", "users": n_users, "movies": n_movies})


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie Recommender API")
    parser.add_argument("--port", type=int, default=5000, help="服务端口")
    parser.add_argument("--recs_dir", default="output/user_recs", help="推荐结果目录")
    parser.add_argument("--movies_dir", default="output/movies", help="电影映射目录")
    parser.add_argument("--log_file", default=None, help="日志文件路径（10MB 自动轮转，保留 3 个备份）")
    parser.add_argument("--db", default=None, help="SQLite 数据库路径（启用后无需全量加载到内存，适合 25M 数据集）")
    args = parser.parse_args()

    # 文件日志
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
        fh = RotatingFileHandler(
            args.log_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(fh)
        logger.info(f"日志文件: {os.path.abspath(args.log_file)}")

    init_data(args.recs_dir, args.movies_dir, db_path=args.db)
    logger.info(f"启动服务: http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
