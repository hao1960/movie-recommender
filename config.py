"""
TMDB API 配置
用于获取电影海报、详情等信息
"""

# TMDB API v3 密钥
TMDB_API_KEY = "7e3cc65f947743110d52d0103fdf6845"

# TMDB API 基础 URL
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# 图片基础 URL
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p"

# 海报尺寸: w92, w154, w185, w342, w500, w780, original
POSTER_SIZE = "w500"

# 请求超时时间（秒）
REQUEST_TIMEOUT = 3
