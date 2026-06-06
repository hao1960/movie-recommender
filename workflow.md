# 开发工作流

## 阶段总览

```
Phase 1         Phase 2         Phase 3         Phase 4         Phase 5         Phase 6
环境搭建 ──→ 数据管道 ──→ 模型训练与评估 ──→ API服务 ──→ 优化与扩展 ──→ 测试验证
(0.5天)       (0.5天)        (1天)            (0.5天)        (按需)          (0.5天)
```

每个 Phase 结束都有明确的**验证标准**，不通过不进入下一阶段。

---

## Phase 1：环境搭建

**目标**：Spark + Python 环境可正常启动，能执行基本的 Spark DataFrame 操作。

### 1.1 安装 Java 8

**Windows**：下载 [Adoptium Temurin 8](https://adoptium.net/download/) 或 `winget install EclipseAdoptium.Temurin.8.JDK`。

**Ubuntu**：
```bash
sudo apt update && sudo apt install openjdk-8-jdk -y
java -version  # 必须输出 1.8.0_xxx
```

### 1.2 安装 Spark

**Windows**：下载 spark-3.4.1-bin-hadoop3.tgz 解压到 `C:\spark`，设置系统环境变量：
- `SPARK_HOME=C:\spark`，PATH 追加 `%SPARK_HOME%\bin`
- 下载 [winutils.exe](https://github.com/cdarlint/winutils)（Hadoop 3.2 版本）放到 `C:\hadoop\bin\`，设置 `HADOOP_HOME=C:\hadoop`

**Ubuntu**：
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
# Windows                          # Linux / macOS
python -m venv venv                python3 -m venv venv
venv\Scripts\activate              source venv/bin/activate
pip install -r requirements.txt
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
# 跨平台 Python 脚本
python download_data.py
# 或只下载 ml-1m
python download_data.py --dataset ml-1m
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
python train_als.py --rank 50 --max_iter 15 --reg_param 0.1
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
python app.py --port 5000

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

## Phase 5：优化与扩展

> **推荐方式**：使用 `run_all.py` 一键串联，以下每条对应一个参数组合。
> ```bash
> python run_all.py --tune                     # 5.1 超参数调优
> python run_all.py --hybrid --alpha 0.7       # 5.3 混合推荐
> python run_all.py --dataset ml-25m           # 5.4 25M + SQLite
> ```

### 5.1 超参数调优（已实现）

```bash
python train_als.py --tune
# 搜索 18 组参数 × 3 折 = 54 次训练，输出最佳 rank/regParam/maxIter
```

### 5.2 冷启动内容推荐（已实现）

评分 <5 条的用户自动用电影类型 Jaccard 相似度推荐，无需手动切换。

### 5.3 混合推荐（已实现）

```bash
python train_als.py --hybrid --alpha 0.7
# ALS 候选池内按内容相似度加权融合重排序
```

### 5.4 切换 25M 数据集 + SQLite 模式

```bash
python train_als.py --data_dir data/ml-25m --driver_memory 4g
python app.py --db output/recommender.db
# SQLite 模式按需查询，无需全量加载到内存
```

### 5.5 结构化请求日志

```bash
python app.py --log_file logs/api.log
# 自动记录 userId / 推荐条数 / 耗时，10MB 轮转
```

---

## Phase 6：测试验证

### 6.1 运行测试

```bash
python -m pytest tests/ -v
# 预期: 10 passed，覆盖 /health /recommend /movie / 四个端点
```

### 6.2 验证标准

- 所有 10 个测试通过
- 覆盖正常返回、参数限制、404 错误等场景

---

## 开发原则

- **每个 Phase 结束时验证，不跳过**。Phase N 的 bug 留到 Phase N+2 排查代价是 10 倍。
- **先用 ml-1m 跑通全流程，再考虑换 25M**。1M 数据每轮训练约 2 分钟，25M 约 20 分钟，迭代效率差一个数量级。
- **代码改动后立即用 curl 或 pytest 验证**，不要等到全部写完了才测试。
- **遇到报错先查第八节排错表**，未覆盖的错误补充进去。
