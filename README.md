# RAGInsight — RAG 检索过程可视化与诊断平台

<p align="center">
  <img src="raginsight-homepage.png" width="800" alt="RAGInsight 首页" />
</p>

**RAGInsight** 是一个用于可视化和诊断 RAG（检索增强生成）系统检索过程的交互式 Web 应用。它将原本黑盒的检索流程以实时流程图的形式展开，帮助开发者直观理解"答案从何而来"，并自动诊断检索过程中的质量缺陷与根因。

---

## ✨ 核心特性

### 已完成的功能阶段

| 阶段 | 功能 | 状态 |
|------|------|------|
| **Phase 1** | 基础设施与统一检索抽象层 | ✅ 已完成 |
| **Phase 2** | 实时质量监控与故障预警 | ✅ 已完成 |
| **Phase 3** | 扰动分析与视觉编码增强 | ✅ 已完成 |
| **Phase 4** | 查询复杂度分析与策略推荐 | ✅ 已完成 |
| **Phase 5** | 系统集成与实验验证 | ✅ 已完成 |
| **Phase 6** | 检索内核增强 + 中文支持 + 工程化 | ✅ 已完成 |

### 1. 实时检索流程可视化
- 基于 **ReactFlow** 的交互式流程图，展示查询解析 → 向量检索 → 上下文构建 → 答案生成的完整链路
- 检索片段（Chunk）节点与答案生成节点的贡献关系可视化
- SSE 流式推送，每步完成后实时更新流程图
- **答案逐 token 流式渲染**，生成完成后自动切换为引用可信度热力图

### 2. 多维质量监控与故障预警
- **三维质量评估**：相关性（Relevance）、多样性（Diversity）、覆盖率（Coverage）
- **六种故障检测**：空结果、低相关性、低覆盖率、低多样性、上下文超长、幻觉风险
- 故障节点视觉编码：⚠️ 警告（黄色边框）/ ❌ 错误（红色边框+发光效果）
- 根因分析面板自动定位问题环节

### 3. 扰动分析（Perturbation Analysis）
- **Leave-one-out** 扰动：逐个移除检索片段，量化每个 Chunk 对最终答案的重要性
- **成本优化策略**：全部片段先通过 Embedding 余弦相似度快速近似；仅对 Top-3 重要片段调用 LLM 做完整扰动
- **What-if 交互**：用户可手动勾选移除任意片段，实时对比原答案与新答案的差异
- 流程图节点透明度、边框粗细、边线粗细全部映射重要性分数

### 4. 因果归因分析（Causal Attribution）
- **组件级反事实干预**：切换策略 / 调整 top_k / 移除片段 / 压缩上下文，量化各组件对答案质量的影响
- 同样采用"Embedding 近似全量 + LLM 精确验证 Top-3"的两阶段成本控制
- 因果 DAG 可视化，节点大小映射归因分数；结果持久化，历史会话可完整回放

### 5. 查询复杂度分析与策略推荐
- **六维度复杂度评分**：查询长度、句子数、实体密度、关系词密度、语义复杂度、跳数需求
- **自动问题类型识别**：事实型 / 定义型 / 列表型 / 比较型 / 因果型 / 多跳型
- **双模式策略推荐**：启发式规则 + 逻辑回归学习型路由（`RAGINSIGHT_ROUTER_MODE` 可切换）
- **策略对比视图**：左右并排对比不同检索策略的流程图与答案
- 复杂度雷达图可视化

### 6. 三种真实检索策略
- **向量检索**：基于 ChromaDB + bge-small-zh-v1.5 的语义检索
- **混合检索**：BM25 稀疏检索 + 向量稠密检索 RRF 融合，兼顾关键词精确匹配与语义理解
- **图检索**：基于实体共现图的多跳 BFS 扩展（5000+ 节点），适合复杂关系查询，返回带推理路径的 Chunk

### 7. 批量实验与评估
- 预置多维度测试数据集：英文 SQuAD 子集 225 条 + 中文唐诗宋词 50 条
- **实验一**：故障检测准确率评估（Precision / Recall / F1，按类型与宏平均）
- **实验二**：跨检索架构对比（向量 / 混合 / 图，对比相关性、覆盖率、多样性、答案质量、耗时）
- **实验三**：单条会话 JSON 导出 + Markdown / PDF 实验报告一键生成，LaTeX 表格直接进论文
- 实验面板支持中英文数据集切换

### 8. 工程化保障
- **114 个自动化测试**：pytest 覆盖核心服务与端到端流程，全部通过
- **三级缓存层**：检索结果缓存 / 答案缓存 / Embedding 缓存，降低 API 调用成本
- **异步性能**：阻塞调用统一卸载线程池，50 并发实测 0.57s；LLM 调用信号量限流 + 指数退避重试
- **答案可信度热力图**：有引用支撑的句子绿色高亮，无引用的红色警示
- **部署**：Docker Compose 一键启动，Alembic 数据库迁移

---

## 🏗 技术栈

| 层级 | 技术选型 |
|------|----------|
| **后端** | Python 3.13 + FastAPI 0.115 + SQLAlchemy 2.0 (async) + SQLite |
| **向量库** | ChromaDB 0.5.20 + ONNX Embedding (`all-MiniLM-L6-v2`) |
| **LLM** | DeepSeek API (`deepseek-v4-flash`，仅用于最终答案生成） |
| **前端** | React 18 + TypeScript + Vite + ReactFlow + Tailwind CSS |
| **稀疏检索** | BM25 (`rank-bm25`) + jieba 中文分词 |
| **图检索** | networkx 实体共现图 + BFS 多跳扩展 |
| **数据流** | SSE (Server-Sent Events) 流式推送 |
| **缓存** | cachetools TTLCache（检索/答案/Embedding） |
| **科学计算** | NumPy + scikit-learn |

---

## 📁 项目结构

```
raginsight/
├── backend/
│   ├── app/
│   │   ├── core/                  # 数据库、配置、ORM 模型
│   │   │   ├── config.py
│   │   │   ├── database.py
│   │   │   └── models.py          # Session / Step / Chunk / Alert
│   │   ├── routers/               # API 路由
│   │   │   ├── query.py           # SSE 流式查询 POST /api/query
│   │   │   ├── sessions.py        # 会话管理
│   │   │   ├── perturbation.py    # 扰动分析 + What-if
│   │   │   ├── complexity.py      # 复杂度分析
│   │   │   └── experiments.py     # 批量实验与指标计算
│   │   ├── services/              # 核心业务逻辑
│   │   │   ├── rag_pipeline.py    # 完整 RAG 流程 + SSE 事件生成
│   │   │   ├── retriever.py       # 向量/混合/图检索适配器 + RRF 融合
│   │   │   ├── deepseek.py        # DeepSeek API 客户端
│   │   │   ├── quality.py         # 质量评估器（三维评分）
│   │   │   ├── root_cause.py      # 根因分析规则引擎
│   │   │   ├── perturbation.py    # 扰动分析器（两阶段近似+精确）
│   │   │   ├── complexity.py      # 查询复杂度分析器（中英文双语）
│   │   │   ├── strategy_recommender.py  # 策略推荐引擎
│   │   │   ├── experiment_runner.py     # 批量实验执行器
│   │   │   ├── export_service.py        # 数据导出服务
│   │   │   └── cache.py                 # 三级缓存层
│   │   ├── utils/
│   │   │   └── text_utils.py      # 中英文分词与语言检测
│   │   ├── main.py                # FastAPI 入口
│   │   └── schemas.py             # Pydantic 数据模型
│   ├── scripts/
│   │   ├── init_data.py           # 初始化唐诗宋词到 ChromaDB
│   │   └── build_knowledge_graph.py # 构建实体共现图
│   ├── tests/                     # pytest 自动化测试（114 用例）
│   ├── experiments/               # 实验结果自动保存目录
│   ├── requirements.txt
│   └── .env                       # DeepSeek API Key 等配置
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Layout.tsx         # 三栏布局框架
│   │   │   ├── QueryPanel.tsx     # 左侧：查询输入 + 历史记录 + 复杂度雷达图
│   │   │   ├── FlowChart.tsx      # 中间：ReactFlow 检索流程图
│   │   │   ├── DetailPanel.tsx    # 右侧：节点详情 + 质量指标 + 扰动/What-if
│   │   │   ├── RootCausePanel.tsx # 顶部：根因分析横幅
│   │   │   ├── StrategyPanel.tsx  # 策略推荐卡片
│   │   │   ├── CompareView.tsx    # 策略对比并排视图
│   │   │   ├── ExperimentPanel.tsx# 实验验证面板（支持中英文数据集切换）
│   │   │   └── RadarChart.tsx     # 复杂度雷达图
│   │   ├── hooks/
│   │   │   └── useSSE.ts          # SSE EventSource 封装
│   │   ├── types/
│   │   │   └── index.ts           # TypeScript 类型定义
│   │   └── App.tsx                # 主组件：状态管理与事件分发
│   └── package.json
├── data/                          # 测试数据集（SQuAD / NQ-Open / 中文诗词）
└── README.md
```

---

## 🚀 快速开始

### 环境要求
- Python 3.13+
- Node.js 18+
- DeepSeek API Key（仅用于答案生成，其余全部本地计算）

### 1. 克隆与初始化

```bash
# 后端依赖
cd backend
python -m venv venv
# Windows:
.\venv\Scripts\pip install -r requirements.txt
# 首次运行会自动下载 ONNX Embedding 模型（约 80MB）

# 初始化测试数据（中文唐诗宋词知识库）
.\venv\Scripts\python scripts\init_data.py

# （可选）构建知识图谱以启用图检索策略
.\venv\Scripts\python scripts\build_knowledge_graph.py

# 运行自动化测试
.\venv\Scripts\pytest tests/ -v

# 前端依赖
cd ../frontend
npm install
```

### 2. 配置环境变量

复制模板文件并填入你的 API Key：

```bash
cp backend/.env.example backend/.env
# 编辑 backend/.env，填入真实的 DEEPSEEK_API_KEY
```

`.env` 文件格式（所有变量使用 `RAGINSIGHT_` 前缀）：

```env
RAGINSIGHT_ENVIRONMENT=development
RAGINSIGHT_DEEPSEEK_API_KEY=your_deepseek_api_key
RAGINSIGHT_DEEPSEEK_BASE_URL=https://api.deepseek.com
RAGINSIGHT_DEEPSEEK_MODEL=deepseek-v4-flash
RAGINSIGHT_DATABASE_URL=sqlite+aiosqlite:///./raginsight.db
RAGINSIGHT_CHROMA_DB_PATH=./chroma_db
RAGINSIGHT_CHROMA_DB_TEST_PATH=./chroma_db_test
RAGINSIGHT_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

### 3. 启动服务

```bash
# 终端 1：启动后端（PowerShell）
cd backend
.\venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 终端 2：启动前端（PowerShell）
cd frontend
npm run dev
```

访问 `http://localhost:5173`

### 4. 验证

在查询框输入 `"What is machine learning?"`，观察：
1. 中间流程图实时构建检索链路
2. 右侧面板显示质量指标进度条
3. 顶部根因分析面板给出诊断
4. 查询完成后点击"扰动分析"，查看片段重要性排序与 What-if 功能

---

## 📸 功能截图

<p align="center">
  <img src="raginsight-flow-chart.png" width="700" alt="流程图可视化" />
  <br/>
  <em>检索流程实时可视化，节点颜色编码故障级别</em>
</p>

<p align="center">
  <img src="raginsight-node-detail.png" width="700" alt="节点详情" />
  <br/>
  <em>点击节点查看详细输入输出、质量指标与检索片段</em>
</p>

<p align="center">
  <img src="raginsight-query-result.png" width="700" alt="查询结果" />
  <br/>
  <em>根因分析、策略推荐、质量指标与扰动分析面板</em>
</p>

---

## 🔌 关键 API 端点

| 方法 | 端点 | 说明 |
|------|------|------|
| `POST` | `/api/query` | SSE 流式执行 RAG 查询（事件：step / alert / answer_token / root_cause / done） |
| `GET` | `/api/sessions` | 获取历史会话列表 |
| `GET` | `/api/sessions/{id}` | 获取单条会话详情 |
| `POST` | `/api/sessions/{id}/perturbation` | 触发全量扰动分析（异步任务） |
| `POST` | `/api/sessions/{id}/perturbation/what-if` | 移除指定片段，返回新答案与差异 |
| `GET` | `/api/tasks/{task_id}/status` | 查询异步任务进度 |
| `POST` | `/api/analyze-complexity` | 分析查询复杂度（纯本地计算） |
| `POST` | `/api/experiments/run` | 启动批量实验 |
| `GET` | `/api/experiments/status` | 实验进度 |
| `GET` | `/api/experiments/metrics` | 实验一 & 实验二指标 |
| `POST` | `/api/experiments/export/{session_id}` | 导出单条会话 JSON |
| `GET` | `/api/experiments/report` | 导出 Markdown 实验报告 |
| `GET` | `/health` | 健康检查 |

---

## 💰 成本控制策略

本项目严格遵循"仅答案生成调用 LLM"的成本约束：

| 模块 | 计算方式 | 是否调用 LLM |
|------|----------|-------------|
| Embedding | ONNX 本地模型 (`all-MiniLM-L6-v2`) | ❌ 否 |
| 向量检索 | ChromaDB 本地计算 | ❌ 否 |
| 质量评估 | 相似度/Jaccard/覆盖率 本地规则 | ❌ 否 |
| 复杂度分析 | 正则+规则本地计算 | ❌ 否 |
| 策略推荐 | 规则矩阵匹配 | ❌ 否 |
| 根因分析 | 规则引擎 | ❌ 否 |
| 扰动分析（近似阶段） | Embedding 余弦相似度 | ❌ 否 |
| 扰动分析（精确阶段） | 仅 Top-3 片段调用 LLM | ✅ 是 |
| 答案生成 | DeepSeek API | ✅ 是 |

---

## 📊 实验指标示例

系统内置批量实验模块，可自动计算以下指标：

- **故障检测准确率**：Macro Precision / Recall / F1，按 alert 类型细粒度拆解
- **跨策略对比**：向量检索 vs 混合检索在相关性、覆盖率、多样性、答案质量、响应耗时上的差异

---

## 📄 相关文档

- `PROJECT_STATUS.md` — 详细项目状态与技术细节
- `KIMI_START_PROMPT.md` — 开发对话启动模板（含 Phase 历史与约束）

---

## 📊 项目评估（2026-06 更新）

### 总体状态

项目已完成全链路功能开发与三轮工程化加固，是一个功能完整的 RAG 可视化诊断原型系统。

| 维度 | 评估 | 说明 |
|------|------|------|
| 功能完整性 | ★★★★★ | 全链路闭环：查询 → 检索 → 质量评估 → 根因诊断 → 扰动分析 → 因果归因 → 流式答案生成 |
| 代码质量 | ★★★★☆ | 分层架构清晰，Adapter+Registry 检索抽象，TypeScript + Pydantic 双端类型安全 |
| 测试覆盖 | ★★★★☆ | 114 个 pytest 用例覆盖核心模块与端到端流程，全部通过 |
| 工程化 | ★★★★☆ | Docker Compose 部署、Alembic 迁移、三级缓存、LLM 重试限流、阻塞调用线程池卸载 |
| 论文支撑 | ★★★★☆ | LLM-as-a-Judge 本地指标、Markdown/PDF/LaTeX 报告导出、批量实验流水线 |

### 关键指标

| 指标 | 数据 |
|------|------|
| 后端代码量 | ~5,400 行 Python（32 个源文件） |
| 前端代码量 | ~3,000 行 TypeScript/TSX（15 个源文件） |
| 测试代码量 | ~1,400 行（12 个文件，114 用例） |
| 检索策略 | 3 种（向量 / BM25+向量 RRF 混合 / networkx 图检索） |
| 故障检测类型 | 6 种（空结果/低相关性/低覆盖率/低多样性/上下文超长/幻觉风险） |
| 支持语言 | 中文 + 英文（双语复杂度分析、中文分词） |
| 测试数据集 | 英文 SQuAD 子集 225 条 + 中文唐诗宋词 50 条 |

### 待改进项

1. **任务队列**：扰动分析/批量实验使用内存级任务管理，服务重启后状态丢失（生产化方案：Redis + arq/Celery）
2. **评估体系**：忠实度为 embedding 近似判断，可升级为本地 NLI 事实一致性模型
3. **质量阈值**：告警阈值为经验值，可改为基于数据集分布动态标定
4. **单用户部署**：暂无用户认证与多租户隔离

---

## 📌 已知限制

1. **图检索依赖预建图**：`graph` 策略需要预先运行 `scripts/build_knowledge_graph.py` 构建实体共现图（见"快速开始"第 1 步），否则自动退化为向量检索。
2. **并发控制**：异步任务队列（扰动分析、批量实验）使用内存级信号量，服务重启后任务状态丢失。
3. **单用户部署**：暂无用户认证与多租户隔离，适合本地开发/演示场景。

---

## 📜 License

本项目为学术研究/大创项目用途，具体许可证待定。
