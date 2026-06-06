"""Flask API 集成测试 — 使用 Flask test client，无需启动真实服务"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app as api_module


@pytest.fixture
def client(monkeypatch):
    """注入模拟数据到全局缓存后返回 Flask test client"""
    monkeypatch.setattr(api_module, "_rec_dict", {
        1: [1193, 260, 1210, 2028, 589],
        2: [296, 1270, 593, 2396, 2571],
    })
    monkeypatch.setattr(api_module, "_movie_titles", {
        1193: "One Flew Over the Cuckoo's Nest (1975)",
        260: "Star Wars: Episode IV - A New Hope (1977)",
        1210: "Star Wars: Episode VI - Return of the Jedi (1983)",
        2028: "Saving Private Ryan (1998)",
        589: "Terminator 2: Judgment Day (1991)",
        296: "Pulp Fiction (1994)",
    })
    monkeypatch.setattr(api_module, "_movie_genres", {
        1193: "Drama",
        260: "Action|Adventure|Sci-Fi",
        1210: "Action|Adventure|Sci-Fi",
        2028: "Action|Drama|War",
        589: "Action|Sci-Fi|Thriller",
        296: "Comedy|Crime|Drama",
    })
    return api_module.app.test_client()


class TestHealth:
    def test_returns_ok_and_counts(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["users"] == 2
        assert data["movies"] == 6


class TestRecommend:
    def test_returns_top_10_by_default(self, client):
        resp = client.get("/recommend/1")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["user_id"] == 1
        assert len(data["recommendations"]) == 5  # only 5 in mock

    def test_respects_limit(self, client):
        resp = client.get("/recommend/1?limit=3")
        assert resp.status_code == 200
        assert len(resp.get_json()["recommendations"]) == 3

    def test_user_not_found(self, client):
        resp = client.get("/recommend/99999")
        assert resp.status_code == 404
        assert "error" in resp.get_json()

    def test_limit_below_1_is_clamped(self, client):
        resp = client.get("/recommend/1?limit=0")
        assert resp.status_code == 200
        assert len(resp.get_json()["recommendations"]) >= 1

    def test_limit_above_50_is_clamped(self, client):
        resp = client.get("/recommend/1?limit=100")
        assert resp.status_code == 200
        assert len(resp.get_json()["recommendations"]) <= 50

    def test_response_has_correct_fields(self, client):
        resp = client.get("/recommend/1?limit=1")
        rec = resp.get_json()["recommendations"][0]
        assert "movieId" in rec
        assert "title" in rec
        assert "genres" in rec


class TestMovieInfo:
    def test_returns_movie_details(self, client):
        resp = client.get("/movie/1193")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["movieId"] == 1193
        assert "Cuckoo" in data["title"]

    def test_movie_not_found(self, client):
        resp = client.get("/movie/99999")
        assert resp.status_code == 404
        assert "error" in resp.get_json()


class TestHome:
    def test_home_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "html" in resp.content_type or "text/html" in resp.content_type
