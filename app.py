"""
推荐系统 REST API 服务（跨平台：Windows / Linux / macOS）
启动时加载离线训练产出的 CSV 到内存，提供 kv 查询接口。

推荐结果已自动排除用户历史评分的电影，确保推荐内容的发现性。

用法:
    # Linux / macOS
    #   source venv/bin/activate
    #   python3 app.py --port 5000 --recs_dir output/user_recs --movies_dir output/movies
    #
    # Windows
    #   venv\\Scripts\\activate
    #   python app.py --port 5000 --recs_dir output/user_recs --movies_dir output/movies
"""
import argparse
import logging
import os
import time

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


# ============================================================
# 数据加载
# ============================================================

def load_csv_dir(dir_path: str) -> pd.DataFrame:
    """加载 Spark 输出的 CSV 目录（合并所有 part-*.csv）"""
    frames = []
    for fname in sorted(os.listdir(dir_path)):
        if fname.startswith("part-") and fname.endswith(".csv"):
            filepath = os.path.join(dir_path, fname)
            frames.append(pd.read_csv(filepath))
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


def init_data(recs_dir: str, movies_dir: str) -> None:
    """启动时一次性加载所有数据到全局字典"""
    global _rec_dict, _movie_titles, _movie_genres
    logger.info(f"加载推荐数据: {recs_dir}")
    _rec_dict = load_recs(recs_dir)
    logger.info(f"加载电影数据: {movies_dir}")
    _movie_titles, _movie_genres = load_movies(movies_dir)
    logger.info(f"加载完成: {len(_rec_dict)} 用户, {len(_movie_titles)} 电影")


# ============================================================
# 中间件
# ============================================================

@app.before_request
def before_request() -> None:
    g.start_time = time.time()


@app.after_request
def after_request(response):
    """请求日志：记录 method、path、status、耗时"""
    elapsed = (time.time() - g.start_time) * 1000
    logger.info(
        f"{request.method} {request.path} → {response.status_code} "
        f"({elapsed:.1f}ms)"
    )
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
def movie_info(movie_id: int):
    """查询单部电影元信息"""
    title = _movie_titles.get(movie_id)
    if title is None:
        return jsonify({"error": f"Movie {movie_id} not found"}), 404

    return jsonify({
        "movieId": movie_id,
        "title": title,
        "genres": _movie_genres.get(movie_id, "Unknown"),
    })


@app.route("/health")
def health():
    """服务健康检查"""
    return jsonify({
        "status": "ok",
        "users": len(_rec_dict),
        "movies": len(_movie_titles),
    })


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie Recommender API")
    parser.add_argument("--port", type=int, default=5000, help="服务端口")
    parser.add_argument("--recs_dir", default="output/user_recs", help="推荐结果目录")
    parser.add_argument("--movies_dir", default="output/movies", help="电影映射目录")
    args = parser.parse_args()

    init_data(args.recs_dir, args.movies_dir)
    logger.info(f"启动服务: http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
