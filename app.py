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
from functools import lru_cache
from logging.handlers import RotatingFileHandler

import pandas as pd
import requests
from flask import Flask, jsonify, request, g, send_from_directory

from config import TMDB_API_KEY, TMDB_BASE_URL, TMDB_IMAGE_BASE_URL, POSTER_SIZE, REQUEST_TIMEOUT

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
    # 自动检测分隔符：电影文件用 |，推荐文件用 ,
    frames = []
    sample = None
    for fname in sorted(os.listdir(dir_path)):
        if fname.startswith("part-") and fname.endswith(".csv"):
            if sample is None:
                with open(os.path.join(dir_path, fname)) as f:
                    first = f.readline()
                    sample = "|" if "|" in first else ","
            filepath = os.path.join(dir_path, fname)
            frames.append(pd.read_csv(filepath, sep=sample))
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
# TMDB 电影海报获取
# ============================================================

# 简单缓存字典（不缓存 None 结果）
_tmdb_cache: dict[str, dict | None] = {}


def _search_movie_from_tmdb(title: str) -> dict | None:
    """从 TMDB 搜索电影（带缓存）

    如果网络不可用，返回 None 并使用默认值。
    不缓存 None 结果，以便网络恢复后可以重试。
    """
    # 检查缓存（只缓存成功结果）
    if title in _tmdb_cache:
        return _tmdb_cache[title]

    # 清理标题：去掉年份括号，如 "Star Wars (1977)" -> "Star Wars"
    clean_title = title.split("(")[0].strip()

    url = f"{TMDB_BASE_URL}/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": clean_title,
        "language": "zh-CN",  # 中文结果
    }

    logger.info(f"TMDB 搜索: {clean_title}")

    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        if data.get("results"):
            movie = data["results"][0]  # 取第一个匹配结果
            poster_path = movie.get("poster_path")
            logger.info(f"TMDB 找到: {movie.get('title')}, poster: {poster_path}")
            result = {
                "tmdb_id": movie["id"],
                "poster_path": poster_path,
                "overview": movie.get("overview", ""),
                "release_date": movie.get("release_date", ""),
                "vote_average": movie.get("vote_average", 0),
            }
            _tmdb_cache[title] = result  # 缓存成功结果
            return result
        else:
            logger.warning(f"TMDB 未找到: {clean_title}")
            # 不缓存 None，以便重试
    except requests.exceptions.Timeout:
        logger.warning(f"TMDB API 超时（网络问题）: {clean_title}")
    except requests.exceptions.ConnectionError:
        logger.warning(f"TMDB API 连接失败（网络不可用）: {clean_title}")
    except Exception as e:
        logger.warning(f"TMDB 搜索失败 [{title}]: {e}")

    return None


def get_movie_poster(title: str) -> str:
    """根据电影标题获取海报 URL"""
    result = _search_movie_from_tmdb(title)
    if result and result.get("poster_path"):
        return f"{TMDB_IMAGE_BASE_URL}/{POSTER_SIZE}{result['poster_path']}"
    return "/static/default_poster.jpg"


def get_movie_details(title: str, genres: str = "") -> dict:
    """获取电影详情（海报、简介、评分等）

    海报获取优先级：
    1. TMDB 海报（真实海报）
    2. AI 生成海报（Pollinations.ai）
    """
    result = _search_movie_from_tmdb(title)

    if result:
        # TMDB 有结果
        if result.get("poster_path"):
            # 有真实海报
            poster_url = f"{TMDB_IMAGE_BASE_URL}/{POSTER_SIZE}{result['poster_path']}"
        else:
            # TMDB 有结果但没有海报，使用 AI 生成
            poster_url = generate_ai_poster_url(title, genres)
        return {
            "poster_url": poster_url,
            "overview": result.get("overview", ""),
            "release_date": result.get("release_date", ""),
            "vote_average": result.get("vote_average", 0),
        }

    # TMDB 没有结果，使用 AI 生成海报
    return {
        "poster_url": generate_ai_poster_url(title, genres),
        "overview": "",
        "release_date": "",
        "vote_average": 0,
    }


def get_movie_genres(title: str) -> str:
    """获取电影类型（用于生成个性化海报）"""
    result = _search_movie_from_tmdb(title)
    if result:
        return result.get("overview", "")
    return ""


def generate_ai_poster_url(title: str, genres: str) -> str:
    """使用 Pollinations.ai 生成 AI 海报 URL

    这是一个免费的 AI 图像生成服务，可以根据提示词生成电影海报风格的图片。
    """
    # 清理标题
    clean_title = title.split("(")[0].strip()

    # 根据类型生成风格提示词
    style_map = {
        "Action": "action movie poster, dramatic lighting, explosions",
        "Adventure": "adventure movie poster, epic landscape, treasure",
        "Comedy": "comedy movie poster, funny scene, bright colors",
        "Drama": "drama movie poster, emotional portrait, cinematic",
        "Horror": "horror movie poster, dark atmosphere, scary",
        "Romance": "romance movie poster, romantic scene, soft lighting",
        "Sci-Fi": "sci-fi movie poster, futuristic, space, technology",
        "Thriller": "thriller movie poster, suspenseful, dark mood",
        "Animation": "animated movie poster, colorful, cartoon style",
        "Documentary": "documentary poster, real photo style, informative",
        "Fantasy": "fantasy movie poster, magical, mythical creatures",
        "Mystery": "mystery movie poster, detective, dark shadows",
        "War": "war movie poster, battlefield, soldiers",
        "Western": "western movie poster, cowboy, desert",
        "Music": "music movie poster, concert, instruments",
        "Crime": "crime movie poster, detective, noir style",
    }

    # 获取主类型
    first_genre = genres.split("|")[0] if genres else "Drama"
    style = style_map.get(first_genre, "cinematic movie poster")

    # 构建提示词
    prompt = f"{clean_title}, {style}, movie poster art, detailed, professional"

    # URL 编码提示词
    import urllib.parse
    encoded_prompt = urllib.parse.quote(prompt)

    # Pollinations.ai 免费 API
    # 使用固定种子确保同一电影生成相同的海报
    seed = hash(title) % 1000000

    return f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=500&height=750&seed={seed}&nologo=true"


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
    """为指定用户返回个性化推荐列表（含电影海报）"""
    limit = request.args.get("limit", 10, type=int)
    limit = max(1, min(limit, 50))

    rec_list = _query_recs(user_id, limit)
    if not rec_list:
        return jsonify({"error": f"User {user_id} not found"}), 404

    # 为每部电影添加海报和详情
    logger.info(f"获取用户 {user_id} 的推荐海报...")
    for rec in rec_list:
        logger.info(f"  处理电影: {rec['title']}")
        details = get_movie_details(rec["title"], rec.get("genres", ""))
        rec["poster_url"] = details["poster_url"]
        rec["overview"] = details["overview"]
        rec["release_date"] = details["release_date"]
        rec["vote_average"] = details["vote_average"]
        logger.info(f"    海报: {rec['poster_url'][:60]}...")

    return jsonify({"user_id": user_id, "recommendations": rec_list})


@app.route("/movie/<int:movie_id>")
def movie_info(movie_id: int):
    """查询单部电影元信息（含海报）"""
    data = _query_movie(movie_id)
    if data is None:
        return jsonify({"error": f"Movie {movie_id} not found"}), 404

    # 添加海报和详情
    details = get_movie_details(data["title"])
    data.update(details)

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
