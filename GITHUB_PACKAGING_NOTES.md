# RAGInsight GitHub 打包注意事项

> 本文件由自动化整理生成，记录从原始项目提取到 `RAGInsight_Clean/` 的过程与后续上传 GitHub 的关键注意点。

---

## 1. 本文件夹已包含的内容（运行时所需）

| 目录/文件 | 说明 |
|-----------|------|
| `backend/app/` | FastAPI 应用核心代码（路由、服务、模型、配置） |
| `backend/scripts/` | 数据初始化、知识图谱构建等脚本 |
| `backend/tests/` | pytest 自动化测试（104 个用例） |
| `backend/alembic/` | 数据库迁移框架（初始迁移为空实现，实际建表由 `database.py` 兜底） |
| `backend/requirements.txt` | Python 依赖清单 |
| `backend/Dockerfile` | 后端 Docker 镜像 |
| `backend/.env.example` | 环境变量模板（安全，无真实密钥） |
| `frontend/src/` | React + TypeScript + Vite 前端源码 |
| `frontend/package*.json` | npm 依赖 |
| `frontend/vite.config.ts` 等 | 前端工程配置 |
| `frontend/Dockerfile`、`frontend/nginx.conf` | 前端生产部署配置 |
| `docker-compose.yml` | 一键启动前后端 |
| `.env.docker` | Docker 环境变量模板（安全） |
| `data/` | 英文测试数据集（SQuAD / NQ-Open） |
| `paper/` | LaTeX 论文源文件与图片 |
| `README.md` / `USER_MANUAL.md` / `PROJECT_STATUS.md` / `NEXT_STEPS.md` | 项目文档 |

---

## 2. 已排除的内容（请勿直接提交到 Git）

| 类别 | 排除原因 |
|------|---------|
| `backend/.env` | 包含真实 DeepSeek API Key，已泄漏风险 |
| `backend/venv/` | Python 虚拟环境，体积大且不可复现 |
| `frontend/node_modules/` | npm 依赖，应由 `npm install` 重新生成 |
| `frontend/dist/` | 前端构建产物，应由 `npm run build` 重新生成 |
| `backend/chroma_db/`、`backend/chroma_db_old/` | 向量库，体积大且可在本地重新生成 |
| `backend/raginsight.db` | SQLite 运行时数据库 |
| `*.log`、`.playwright-mcp/` | 运行时日志 |
| 根目录所有 `raginsight-*.png`、`phase2-test.png` | 演示截图，非运行必需 |
| `data/chinese-poetry/` | 嵌套 Git 仓库，需作为 submodule 处理 |
| `不佳回答的原因/`、`检索器方法创新/`、`检索过程可视化与可解释性/`、`简历面试资料/` | 调研资料、PDF、面试文档，与代码无关 |

---

## 3. 上传 GitHub 前的必要操作

### 3.1 立即轮换 API Key

原始项目 `backend/.env` 中存放了真实 DeepSeek API Key，存在泄露风险。请登录 DeepSeek 平台：
- 删除或轮换该 Key；
- 在新环境中复制 `backend/.env.example` 为 `backend/.env`，填入**新的 Key**。

```bash
cp backend/.env.example backend/.env
# 编辑 backend/.env，填入 RAGINSIGHT_DEEPSEEK_API_KEY=sk-你的新密钥
```

### 3.2 确认模型名称

当前配置使用 `RAGINSIGHT_DEEPSEEK_MODEL=deepseek-v4-flash`。DeepSeek 官方常见模型名为 `deepseek-chat`、`deepseek-reasoner` 等，请核实该模型名在你的账号下是否可用，否则调用会失败。

### 3.3 处理嵌套 Git 仓库 `data/chinese-petry/`

`data/chinese-poetry/` 内部有自己的 `.git/`，直接提交会导致异常。三种处理方式：
1. **作为 Git submodule**：`git submodule add https://github.com/chinese-poetry/chinese-poetry.git data/chinese-poetry`
2. **删除其 `.git/` 后作为普通目录提交**：
   ```bash
   rm -rf data/chinese-poetry/.git
   git add data/chinese-poetry
   ```
3. **不在仓库中存放**：在 README 中说明用户自行下载/运行 `scripts/init_data.py` 生成。

### 3.4 运行测试验证

```bash
cd backend
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python -m pytest tests/ -v
```

注意：
- `tests/test_e2e.py` 与 `tests/test_paper_experiments.py` 可能调用真实 DeepSeek API，运行前确保 `.env` 已配置且余额充足；
- Windows 路径含中文/非 ASCII 字符时，`pytest-asyncio` 可能出现 `OSError: could not get source code`，建议将项目放在纯英文路径下运行。

### 3.5 前端构建验证

```bash
cd frontend
npm install
npm run build
```

构建成功后会在 `frontend/dist/` 生成静态文件（该目录已被 `.gitignore` 忽略）。

### 3.6 首次运行数据初始化

```bash
cd backend
venv\Scripts\python scripts\init_data.py
# （可选）构建知识图谱以启用图检索策略
venv\Scripts\python scripts\build_knowledge_graph.py
```

### 3.7 README 图片链接

当前 `README.md` 中引用的 `raginsight-*.png` 等截图未包含在干净文件夹中，上传 GitHub 后图片会显示为 404。建议：
- 将关键截图放入 `docs/images/` 并更新 README 链接；或
- 使用 GitHub 拖拽上传图片到 Issue/PR，获取稳定外链后替换 README。

---

## 4. 推荐仓库结构（最终效果）

```
RAGInsight/
├── .gitignore
├── docker-compose.yml
├── .env.docker
├── README.md
├── USER_MANUAL.md
├── PROJECT_STATUS.md
├── NEXT_STEPS.md
├── KIMI_START_PROMPT.md
├── run_query.py
├── start_project.py
├── backend/
│   ├── app/
│   ├── scripts/
│   ├── tests/
│   ├── alembic/
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── alembic.ini
│   └── .env.example
├── frontend/
│   ├── src/
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── index.html
│   ├── nginx.conf
│   └── Dockerfile
├── data/
│   ├── nq_open_dev.jsonl
│   ├── squad_dev_v1.1.json
│   └── squad_test_queries.json
└── paper/
    ├── README.md
    ├── raginsight_paper.tex
    └── figures/
```

---

## 5. 已知限制（需在 README 中保持透明）

1. **图检索需预建知识图谱**：否则自动降级为向量检索；
2. **异步任务队列为内存级**：扰动分析、批量实验进度在服务重启后丢失；
3. **单用户部署**：无身份认证与多租户隔离，适合本地/演示场景；
4. **前端 API 路径为相对路径 `/api`**：部署到非本地环境需确保 nginx/代理正确配置；
5. **Alembic 初始迁移为空**：生产环境建议补充完整 migration 脚本。

---

## 6. 评估与评级摘要

| 维度 | 评级 | 说明 |
|------|------|------|
| 功能完整性 | ★★★★☆ | 6 个 Phase 功能闭环，含可视化、诊断、扰动、实验 |
| 代码质量 | ★★★★☆ | 分层清晰，TypeScript + Pydantic 双端类型安全 |
| 测试覆盖 | ★★★★☆ | 104 个测试用例，核心模块 82/83 通过 |
| 工程化 | ★★★☆☆ | 有 Docker、pytest，但缺完整迁移、持久化队列、CI |
| 文档 | ★★★★☆ | README、使用手册、项目状态文档齐全 |
| 安全/合规 | ★★☆☆☆ | `.env` 含真实 API Key，根目录无 `.gitignore`，需立即治理 |

**总体评级：B+ / 良好（GitHub 发布前需完成安全治理与 README 图片修复）**
