# RAGInsight 项目状态报告

> 生成时间：2026-05-07 | 当前完成：Phase 1 + Phase 2

---

## 1. 项目概述

RAGInsight 是一个用于可视化和诊断 RAG（检索增强生成）系统检索过程的 Web 应用。

**核心目标**：
- 将 RAG 检索流程以流程图形式可视化
- 实时监测检索质量，自动预警
- 通过扰动分析找出导致不佳回答的根因
- 支持多种检索架构的统一可视化
- 提供检索策略推荐，并可视化解释推荐理由

**成本约束**：仅答案生成调用 DeepSeek API（deepseek-v4-flash），其余全部本地计算。

---

## 2. 已完成的功能

### Phase 1：基础设施与统一检索抽象层 ✅
- 后端 FastAPI + SQLAlchemy(async) + SQLite 数据库
- ChromaDB 向量库 + ONNX embedding（DefaultEmbeddingFunction）
- DeepSeek API 集成（仅用于最终答案生成，带引用标记 `[ref:chunk_X]`）
- 完整的向量 RAG 流程：查询解析 → 向量检索 → 上下文构建 → 答案生成
- SSE 流式推送每步完成事件到前端
- 前端 React + Vite + ReactFlow + Tailwind CSS 三栏布局
- 20 条合成 Wikipedia 文档作为测试知识库

### Phase 2：实时质量监控与故障预警 ✅
- **质量评估器**（`backend/app/services/quality.py`）：
  - 相关性（平均相似度）
  - 多样性（Jaccard 距离）
  - 覆盖率（查询实体在结果中的出现率）
  - 综合分数 = relevance×0.5 + diversity×0.2 + coverage×0.3
- **故障检测**（集成在 `rag_pipeline.py` 中）：
  - `empty_results` — 检索结果为空（ERROR）
  - `low_relevance` — 相关性 < 0.3（WARNING）
  - `low_coverage` — 覆盖率 < 0.3（WARNING）
  - `low_diversity` — 多样性 < 0.2（WARNING）
  - `context_too_long` — 上下文 > 3000 字符（WARNING/ERROR）
  - `hallucination_risk` — 答案引用不存在的 chunk（WARNING）
- **根因分析规则引擎**（`backend/app/services/root_cause.py`）：
  - 检索失败 / 知识不完整 / 幻觉风险 / 策略不匹配
  - 流程结束时推送 `root_cause` SSE 事件
- **前端可视化**：
  - 根因分析面板（流程图上方，显示根因类型+解释+修复建议）
  - 质量指标进度条（右侧面板显示相关性/多样性/覆盖率）
  - 故障节点视觉编码（warning 黄色边框+图标，error 红色边框）
  - 故障路径高亮（根因涉及步骤间边变红色）

---

## 3. 技术栈

| 层级 | 技术 |
|-----|------|
| 后端 | Python 3.13 + FastAPI 0.115 + SQLAlchemy 2.0(async) + SQLite |
| 向量库 | ChromaDB 0.5.20 + DefaultEmbeddingFunction(ONNX) |
| LLM | DeepSeek API (deepseek-v4-flash) |
| 前端 | React 18 + TypeScript + Vite + ReactFlow + Tailwind CSS |
| 数据 | 20 条合成 Wikipedia 文档（Paris/France/ML/NLP等） |

---

## 4. 项目结构

```
raginsight/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── config.py          # 配置（DeepSeek API Key、模型名）
│   │   │   ├── database.py        # SQLAlchemy async engine + session
│   │   │   └── models.py          # ORM: Session, Step, Chunk, Alert
│   │   ├── routers/
│   │   │   ├── query.py           # POST /api/query (SSE 流式)
│   │   │   └── sessions.py        # GET /api/sessions, /api/sessions/{id}
│   │   ├── services/
│   │   │   ├── deepseek.py        # DeepSeek API 客户端（v4-flash）
│   │   │   ├── quality.py         # 质量评估器（相关性/多样性/覆盖率）
│   │   │   ├── root_cause.py      # 根因分析规则引擎
│   │   │   ├── retriever.py       # ChromaDB 向量检索适配器
│   │   │   └── rag_pipeline.py    # 完整 RAG 流程 + SSE 推送
│   │   ├── main.py                # FastAPI 入口 + CORS
│   │   └── schemas.py             # Pydantic 数据模型
│   ├── scripts/
│   │   └── init_data.py           # 初始化 20 条测试文档到 ChromaDB
│   ├── requirements.txt
│   ├── .env                       # 环境变量（API Key 等配置）
│   └── venv/                      # Python 虚拟环境
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Layout.tsx         # 三栏布局
│   │   │   ├── QueryPanel.tsx     # 左侧：查询输入+历史列表
│   │   │   ├── FlowChart.tsx      # 中间：ReactFlow 流程图
│   │   │   ├── DetailPanel.tsx    # 右侧：节点详情+质量指标+警告
│   │   │   └── RootCausePanel.tsx # 顶部：根因分析面板
│   │   ├── hooks/
│   │   │   └── useSSE.ts          # SSE EventSource 封装
│   │   └── types/
│   │       └── index.ts           # TypeScript 类型定义
│   └── package.json
└── PROJECT_STATUS.md              # 本文件
```

---

## 5. 关键文件说明

### 后端核心

**`rag_pipeline.py`** — 最重要的文件
- `run_rag_pipeline(db, query, strategy)`：异步生成器，yield SSE 事件
- 事件类型：`session_created` → `step` → `alert` → `root_cause` → `done`
- 每步检索后调用 `quality_evaluator.evaluate_all()` 计算质量指标
- 答案生成后解析 `[ref:chunk_X]` 引用标记，检测幻觉风险
- 流程结束时调用 `root_cause_analyzer.analyze()` 生成根因分析

**`quality.py`** — 质量评估器
- `evaluate_relevance(chunks)`：平均 relevance_score
- `evaluate_diversity(chunks)`： pairwise Jaccard distance
- `evaluate_coverage(query, chunks)`：查询实体在 chunk 中的出现率
- `evaluate_all(query, chunks)`：返回 {relevance, diversity, coverage, combined}

**`root_cause.py`** — 根因分析
- `analyze(steps, alerts)`：基于规则矩阵返回根因诊断对象
- 规则：empty_results → 知识不完整；low_relevance+coverage → 检索失败；hallucination → 幻觉风险

**`deepseek.py`** — LLM 客户端
- 固定使用 `deepseek-v4-flash` 模型
- System Prompt 要求输出引用标记 `[ref:chunk_INDEX]`
- 仅 `generate_answer(query, context_chunks)` 一个方法

### 前端核心

**`App.tsx`** — 主组件
- `handleQuery()`：提交查询后建立 SSE 连接
- 处理 `step` / `alert` / `root_cause` / `done` 事件更新 state
- 结构：RootCausePanel（顶部）+ Layout（三栏）

**`FlowChart.tsx`** — 流程图
- `buildFlowData(steps, rootCause)`：将 steps 转为 ReactFlow nodes/edges
- 节点 key：`step-${step.id}`；chunk 节点 key：`chunk-${step.id}-${chunk.chunk_index}`
- 根因涉及步骤的边显示为红色粗线
- 节点样式根据 alert severity 变化（warning 黄色 / error 红色）

---

## 6. 环境配置

### 后端 `.env`
```
RAGINSIGHT_ENVIRONMENT=development
RAGINSIGHT_DEEPSEEK_API_KEY=your_api_key_here
RAGINSIGHT_DEEPSEEK_BASE_URL=https://api.deepseek.com
RAGINSIGHT_DEEPSEEK_MODEL=deepseek-v4-flash
RAGINSIGHT_DATABASE_URL=sqlite+aiosqlite:///./raginsight.db
RAGINSIGHT_CHROMA_DB_PATH=./chroma_db
RAGINSIGHT_CHROMA_DB_TEST_PATH=./chroma_db_test
RAGINSIGHT_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

### 启动命令

**后端**（PowerShell）：
```powershell
cd backend
.\venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**前端**（PowerShell，另开窗口）：
```powershell
cd frontend
npm run dev
```

访问 `http://localhost:5173`

---

## 7. 数据模型

| 表 | 关键字段 |
|---|---------|
| sessions | id, query, final_answer, status, execution_trace(JSON) |
| steps | id, session_id, step_type, input_data, output_data, quality_score, duration_ms |
| chunks | id, step_id, content, source, relevance_score, chunk_index |
| alerts | id, session_id, step_id, alert_type, severity, message, suggestion |

**StepType 枚举**：query_parse, vector_retrieve, rerank, context_build, answer_generate

---

## 8. 已知注意事项

1. **Chunk key 问题**：SSE 流式推送时 chunk 尚无数据库 id，FlowChart 使用 `chunk-${step.id}-${chunk.chunk_index}` 作为唯一 key，已修复。
2. **中文编码**：PowerShell 中调用 REST API 时中文可能显示乱码，不影响实际功能。
3. **ChromaDB ONNX 模型**：首次运行时会自动下载 `all-MiniLM-L6-v2` ONNX 模型（约 80MB）到用户缓存目录。
4. **上下文长度阈值**：`CONTEXT_LENGTH_WARNING = 3000` 字符（近似值），`CONTEXT_LENGTH_ERROR = 6000`。

---

## 9. 后续开发计划（Phase 3/4/5）

### Phase 3：扰动分析与视觉编码增强
**目标**：量化每个检索片段对最终答案的重要性，并将结果直观编码到流程图中。

**后端任务**：
1. 实现扰动分析器（`services/perturbation.py`）
   - 移除指定 chunk → 重建上下文 → 调用 DeepSeek 生成新答案
   - 计算原答案与新答案的语义相似度 → 重要性 = 1 - 相似度
   - **成本优化**：仅对 top-3 重要片段做完整 LLM 扰动，其余用 embedding 余弦相似度近似
2. 添加 API 端点：
   - `POST /api/sessions/{id}/perturbation` — 触发全量扰动分析（异步任务）
   - `GET /api/tasks/{task_id}/status` — 查询进度
3. 请求队列：控制并发 DeepSeek API 调用数（max 2 并发），避免 rate limit

**前端任务**：
1. 节点视觉编码：重要性高的节点不透明+粗边框+稍大；重要性低的半透明+细边框
2. 边视觉编码：从 chunk 到答案的边，粗细编码贡献度，颜色编码贡献等级
3. 悬停提示：鼠标悬停 chunk 节点显示重要性分数
4. what-if 交互：用户勾选要移除的 chunk → 执行扰动 → 对比视图（原答案 vs 新答案，高亮差异）
5. 重要性排序面板：右侧栏按重要性分数降序排列所有片段

**验收标准**：
- 扰动分析正确计算每个 chunk 的重要性分数
- 流程图节点透明度/边框正确映射重要性
- what-if 分析功能正常工作

---

### Phase 4：查询复杂度分析与策略推荐
**目标**：根据查询复杂度推荐合适的检索策略，并可视化解释推荐理由。

**后端任务**：
1. 实现查询复杂度分析器（`services/complexity.py`）
   - 复用 Phase 1 查询解析结果（实体数、关键词数）
   - 计算特征：查询长度、句子数、实体数、关系词数、问题类型
   - 综合复杂度评分 = 加权规则计算（完全本地，不调用 LLM）
2. 实现策略推荐引擎（`services/strategy_recommender.py`）
   - 规则矩阵：complexity < 0.3 → 向量检索；0.3~0.7 → 混合检索；> 0.7 → 图检索
   - 生成推荐理由（模板填充）
3. 添加 API 端点：
   - `POST /api/analyze-complexity` — 分析查询复杂度
   - `POST /api/query?strategy=xxx` — 支持传入策略参数执行查询

**前端任务**：
1. 查询复杂度面板：左侧查询区域输入后显示复杂度分数和特征详情
2. 策略推荐卡片：流程图上方显示推荐策略+理由+备选策略按钮
3. 策略对比视图：左右并排显示两个策略的流程图和答案
4. 复杂度雷达图：5 个维度（长度、实体数、关系深度、跳数需求、语义复杂度）

**验收标准**：
- 复杂度评分合理，策略推荐与复杂度匹配
- 支持切换策略并对比结果
- 雷达图正确显示

---

### Phase 5：系统集成与实验验证
**目标**：完成所有功能集成，构建实验数据集，进行对照实验。

**任务**：
1. 端到端集成测试：完整流程走通（查询 → 复杂度分析 → 策略推荐 → 检索 → 质量监控 → 扰动分析 → 答案生成）
2. 构建实验数据集：
   - 已有 20 条合成文档
   - 新增 30 条测试查询（覆盖简单事实/复杂多跳/比较总结/无答案查询）
   - 共 50 条查询，每条标注期望答案类型、所需检索深度、推荐策略
3. 对照实验：
   - 实验1：故障检测准确率（人工标注 vs 系统自动检测，计算 precision/recall/F1）
   - 实验2：跨检索架构对比（同一组查询分别用向量/混合策略运行，对比检索深度、召回率、答案质量）
   - 实验3：数据导出（JSON + 流程图截图）
4. Docker 部署配置（可选）

**验收标准**：
- 所有功能完整集成，端到端流程通畅
- 实验数据集构建完成
- 完成对照实验并记录结果

---

## 11. Phase 6：一个月迭代（检索内核 + 中文支持 + 工程化）✅

### 11.1 工程地基（Week 1）
- **pytest 测试框架**：新增 48 个自动化测试用例，覆盖 quality / complexity / root_cause / strategy / retriever / e2e
- **混合检索真正实现**：`HybridRetrieverAdapter` 基于 BM25 + 向量 RRF 融合，支持中英文双语分词
- **前端修复**：策略对比视图标签显示逻辑修复

### 11.2 中文支持（Week 2）
- **双语复杂度分析器**：自动检测查询语言，中文路径使用 jieba POS 标注提取实体（nr/ns/nt/nz），扩展中文关系词库与问题类型检测
- **中文测试数据集**：新增 30 条唐诗宋词查询（simple_fact / multi_hop / comparison / unanswerable），支持实验面板切换
- **双语策略推荐**：根据查询语言自动选择中文/英文推荐理由模板

### 11.3 图检索与缓存（Week 3）
- **轻量 GraphRAG 原型**：`GraphRetrieverAdapter` 基于 networkx 实体共现图，支持 2-hop BFS 多跳扩展，返回带推理路径的 Chunk
- **建图脚本**：`scripts/build_knowledge_graph.py` 从 ChromaDB 自动抽取实体建图
- **三级缓存层**：`QueryCache`（检索结果 TTL=300s）、`AnswerCache`（简单事实型答案 TTL=600s）、`EmbeddingCache`（向量缓存）

### 11.4 可视化与集成（Week 4）
- **答案可信度热力图**：最终答案按引用标记分段渲染，有引用支撑绿色 `#dcfce7`，无引用红色 `#fee2e2`
- **实验报告导出**：一键导出 Markdown 格式实验报告（含故障检测准确率表格与跨策略对比统计）
- **端到端集成测试**：4 个 e2e 测试验证完整 RAG 流程、缓存命中、答案分段解析

### 11.5 关键依赖更新
```
rank-bm25==0.2.2
jieba==0.42.1
networkx==3.4.0
cachetools==5.5.0
pytest==8.3.0
pytest-asyncio==0.24.0
```

### 11.6 测试统计
- `pytest backend/tests`：48 passed, 0 failed
- 测试覆盖模块：quality / complexity / root_cause / strategy_recommender / retriever_utils / e2e_pipeline

---

## 10. 给 AI 的开发建议

1. **按 Phase 顺序提问**：完成一个 Phase 并验收后再给下一个 Phase
2. **技术选型**：后端已确定 FastAPI + ChromaDB，前端 React + ReactFlow，不要改动技术栈
3. **成本控制**：embedding 和复杂度分析全部本地计算；扰动分析优先用 embedding 近似，仅 top-3 调 LLM
4. **兼容性**：新代码必须与已有数据模型兼容（不删除已有表/字段）
5. **测试查询**："What is the capital of France?" / "What is machine learning?" / "Tell me about Paris and France"
