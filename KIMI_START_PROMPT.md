# Kimi2.6 对话启动提示词

请直接复制以下全部内容，粘贴到新对话中：

---

## 项目背景

我正在开发一个名为 **RAGInsight** 的 Web 应用，用于可视化和诊断 RAG（检索增强生成）系统的检索过程。项目根目录是 `d:\普通下载\大创\接下来主要的任务`。

## 已完成的功能

**Phase 1（基础设施）**：
- 后端：FastAPI + SQLAlchemy(async) + SQLite + ChromaDB(ONNX embedding)
- 前端：React + Vite + ReactFlow + Tailwind CSS 三栏布局
- 完整向量 RAG 流程：查询解析 → 向量检索 → 上下文构建 → 答案生成
- SSE 流式推送每步完成事件
- DeepSeek API（deepseek-v4-flash）仅用于最终答案生成
- 20 条合成 Wikipedia 文档作为测试知识库

**Phase 2（实时质量监控与故障预警）**：
- 质量评估器：相关性、多样性、覆盖率三维度评分
- 6 种故障检测：空结果、低相关性、低覆盖率、低多样性、上下文超长、幻觉风险
- 根因分析规则引擎：检索失败/知识不完整/幻觉风险/策略不匹配
- 前端根因分析面板、质量指标进度条、故障节点视觉编码

## 请先做的事

1. **阅读 `PROJECT_STATUS.md`**（项目根目录下），了解完整技术细节、项目结构、关键文件说明
2. **确认环境**：
   - 后端：`cd backend && .\venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`
   - 前端：`cd frontend && npm run dev`
   - 访问 `http://localhost:5173`
3. **运行测试**：输入 `"What is machine learning?"` 验证流程图、质量指标、根因分析是否正常工作

## 下一步任务

请帮我继续开发 **Phase 3：扰动分析与视觉编码增强**。

### Phase 3 核心目标
量化每个检索片段对最终答案的重要性，并将结果编码到流程图视觉元素中。

### Phase 3 后端任务
1. 创建 `backend/app/services/perturbation.py`
   - 移除单个 chunk → 重建上下文 → 调用 DeepSeek 生成新答案
   - 计算重要性 = 1 - 语义相似度（用 sklearn cosine_similarity）
   - **成本优化**：仅对 top-3 重要片段做完整 LLM 扰动，其余用 embedding 余弦相似度近似
2. 添加异步任务 API：
   - `POST /api/sessions/{id}/perturbation` — 触发全量扰动分析
   - `GET /api/tasks/{task_id}/status` — 查询进度
   - 请求队列控制并发（max 2 并发 DeepSeek 调用）
3. 将扰动结果写入数据库（新增 `perturbations` 表或在 `chunks` 表中增加 `importance_score`）

### Phase 3 前端任务
1. 节点视觉编码：重要性高的不透明+粗边框+稍大；重要性低的半透明+细边框
2. 边视觉编码：从 chunk 到答案的边，粗细映射贡献度
3. 悬停提示：鼠标悬停 chunk 节点显示重要性分数
4. what-if 交互：用户勾选要移除的 chunk → 执行扰动 → 显示对比视图（原答案 vs 新答案，高亮差异）
5. 右侧重要性排序面板：按重要性分数降序排列所有片段

### Phase 3 验收标准
- 扰动分析正确计算每个 chunk 的重要性分数
- 流程图节点透明度/边框正确映射重要性
- what-if 分析功能正常工作

## 约束条件
- **不改动技术栈**（FastAPI + ChromaDB + React + ReactFlow）
- **成本控制**：embedding/复杂度分析全部本地计算；扰动分析优先近似，仅 top-3 调 LLM
- **兼容性**：新代码与已有数据模型兼容（不删除已有表/字段）
- **使用 deepseek-v4-flash 模型**
- DeepSeek API Key 在 `backend/.env` 中

## 已知问题
- SSE 流式推送时 chunk 尚无数据库 id，FlowChart 使用 `chunk-${step.id}-${chunk.chunk_index}` 作为唯一 key（已修复）
- 首次运行 ChromaDB 会自动下载 ONNX 模型（约 80MB）到用户缓存目录
