# 架构与设计决策

本文记录"为什么这么做"，实现细节看代码。适合面试前过一遍，也适合接手的人快速定位。

---

## 1. 分层

```
接入层   FastAPI（SSE / REST）            app/api, app/main.py
编排层   LangGraph 6 节点状态机            app/agent
能力层   检索 / 工具 / 模型                app/retrieval, app/tools, app/llm
数据层   SQLite 商品库 + 内存索引 + 文件索引 app/store, app/data
```

编排层不 import FastAPI，能力层不 import LangGraph。事件通过 `ContextVar` 里的 `EventSink`
往外冒（`app/core/events.py`），HTTP 层把它接到 SSE，测试接到 `CollectSink`，CLI 接 `NullSink`。
这就是为什么整条链可以在 pytest 里跑，而不用起服务。

## 2. 对话状态机

```
START → understand ─┬─(chitchat/aftersales)→ respond → END
                    └→ retrieve ─┬─(conf < 0.42 且未改写)→ rewrite → retrieve
                                 └→ plan → tool → respond → END
```

| 节点 | 职责 | 关键点 |
|---|---|---|
| understand | 规则 + LLM 抽槽，多轮合并 | 品类切换清空旧需求点；"再便宜点"降预算 28% |
| retrieve | 双路召回 + RRF + 过滤 + 重排 | 先过滤后截断，见 §4 |
| rewrite | 低置信度改写 | LLM 失败自动降级为规则扩写（同义词表） |
| plan | 决定调哪个工具 | Function Calling（LLM）失败自动降级为"总是查前 3 个" |
| tool | 执行工具、合并库存 | 超时/异常 → degraded=True，用商品库快照 |
| respond | 流式生成 / 模板兜底 / 澄清 | 无品类且置信度低时先问清楚，不硬推 |

状态是 `TypedDict`，节点只返回增量。会话状态（history + slots）落在 SQLite 的 `sessions` 表，
每轮开始时读、结束时写。这样做的好处是**进程重启后多轮上下文不丢**，也方便回放排查；
代价是少用 LangGraph 自带的 checkpointer——两者二选一，这里选了可观测性。

## 3. 槽位模型

```python
class Slots:
    category: str | None          # 10 个品类枚举
    brand: str | None             # 品牌白名单（归一化中文名）
    price_min / price_max: float  # 元
    attributes: list[str]         # 23 个标准需求点
    scene: str | None             # 送礼/学生/商务/家庭/户外
    exclude_brands: list[str]
```

- **品牌归一**：`apple / 苹果 / Apple` 统一成 `Apple`，靠 `taxonomy.py` 里的别名表。
- **品类别名**：`笔记本 / 轻薄本 / 游戏本` → `笔记本电脑`。
- **否定约束**：`不要小米` 会进 `exclude_brands`，同时在检索前置过滤里强删（不参与打分）。
- **冲突策略**：同一次查询里正则和模型给出不同价格时，正则胜。价格是硬约束，错一天都白干。

## 4. 检索流水线

```
query → 关键词构建 →┬ BM25 top200 ─┐
                    └ 向量 top200 ─┴→ RRF(k=60) → 硬约束过滤 → 截断 30 → 重排 → top6
```

### 4.1 BM25F

自研倒排索引，字段加权：标题 3.0、品牌+品类 3.0、标签 2.0、规格 1.5、描述 1.0。
查询词的加权词频代入 BM25 公式，`k1=1.2, b=0.75`。5 万文档、7400 词表，
构建 23s、单次查询 1ms 级。不用 SQLite FTS5 的原因：FTS5 的 unicode61 不切中文，
要自己塞预切分词串，那不如直接把权重控制权拿回来。

### 4.2 向量

默认 `HashingEmbedder`（词 + 字符 2/3-gram 的带符号哈希，512 维，L2 归一化）。
它是**兜底**，不是最优解：能捕捉词面相似，捕捉不到"拍照好"和"影像旗舰"的语义关系。
装 `fastembed` 后 `EMBEDDING_MODE=auto` 会走 `BAAI/bge-small-zh-v1.5`，维度同为 512，
重建索引即可切换，pipeline 无感。

5 万 × 512 float32 ≈ 100MB，全量点积 2.6ms，比维护 Milvus 更适合单机 demo；
换成 Milvus 只需替换 `VectorIndex.search`。

### 4.3 RRF

`score(d) = Σ 1/(60 + rank_i(d))`。两路分数量纲不同（BM25 十几分、余弦 0~1），
加权求和需要调参且不稳定，RRF 只看名次，零调参且对头部友好。

### 4.4 重排

`FeatureReranker` 用绝对特征：词面（饱和映射）、语义（区间拉伸）、约束满足度
（按品类/品牌/价格/需求点的权重归一）、商品质量（评分 + 销量对数）。
CrossEncoder 版本用 fastembed 的 `bge-reranker-base`，默认关闭（模型 1GB 级）。

**为什么先截断再重排**：CrossEncoder 是 O(N) 次前向，200 条直接重排要几百毫秒；
30 条是延迟与效果的折中。但**约束过滤必须在截断之前**——否则会出现"池子里没有符合预算的，
只能整体放宽"的 bug（见 README §2）。

## 5. 置信度与改写

```
confidence = 0.55·top1 + 0.15·margin + 0.30·coverage
margin     = (top1 - top2)/top1
coverage   = 前三条满足硬约束的比例（无约束时取 0.5）
```

阈值 `0.42`，低于则改写一次。改写优先用 LLM（补同义词、规格词），失败退回规则扩写。
改写后置信度仍低 → 不再循环，交给 respond 决定"降级推荐"还是"澄清追问"。

无品类的查询（"送女朋友的礼物"）置信度天然偏低，系统会先问品类，而不是推一堆随机商品。

## 6. MCP 工具

协议跑在 stdio 上，一行一个 JSON：

```
client → {"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}
server → {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05",...}}
client → {"method":"notifications/initialized"}
client → {"id":2,"method":"tools/list"}
client → {"id":3,"method":"tools/call","params":{"name":"query_inventory","arguments":{"sku_ids":[...]}}}
```

两个工具：`query_inventory`（库存/仓库/时效）、`get_promotion`（券）。
`sku_id` 的 md5 决定仓库与时效，保证同一商品每次查询结果稳定（可复现）。

客户端是一个常驻子进程 + 读循环 + 按 id 的未来对象。超时 100ms（`TOOL_TIMEOUT_MS`），
失败即 `degraded=True` 并回退 `_fallback()` 读 SQLite 快照。每次调用写 `tool_logs`，
`/api/meta` 能看到平均耗时与成功率。

## 7. 可观测性

- 日志：JSON Lines，带 `trace_id` / `session_id`，见 `app/core/logging.py`
- 事件：每个阶段都有 SSE 事件，前端链路面板直接可视化
- 工具日志：`tool_logs` 表，按工具聚合成功率与平均耗时
- 评测：`data/eval_report.json`、`data/eval_ablation.json`

## 8. 数据管道

```
generator（合成 5.1 万，含脏数据）→ etl（清洗/去重/归一）→ 4.97 万 → SQLite + jsonl → 索引
```

脏数据是故意的：价格 `￥4,999.00` / `4999元` / `4999` 三种写法，品牌 `apple` / `苹果`，
尺寸 `6.1寸` / `6.1英寸`，标题带 `【官方正品】` 前缀和 `全新未拆封` 后缀，还有重复铺货。
ETL 的规则都在 `app/data/etl.py` 顶部注释里，清洗统计会打印出来（去重 1740、非法 107）。

## 9. 取舍清单

| 选择 | 替代方案 | 为什么这么选 |
|---|---|---|
| 自研 BM25F | SQLite FTS5 / ES | 中文分词与字段权重可控，零外部依赖 |
| numpy 向量矩阵 | Milvus / FAISS | 5 万条内存检索 2.6ms，省掉一套中间件 |
| SQLite 会话 | LangGraph checkpointer | 跨进程可恢复、可审计，实现更直白 |
| 手写 MCP | 官方 SDK | 协议透明，少一个依赖；生产会换 SDK |
| 模板兜底生成 | 只用 LLM | 没有 key 也能完整跑通，CI 里可断言 |
