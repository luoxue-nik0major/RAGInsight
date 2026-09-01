import { useState, useMemo } from 'react';
import { useHighlightContext } from '../contexts/HighlightContext';
import CausalGraph from './CausalGraph';
import type { Step, Session, Alert, PerturbationResult, PerturbationTask, WhatIfResponse, AttributionReport, AttributionTask } from '../types';

interface Props {
  step: Step | null;
  session: Session | null;
  streamingAnswer: string;
  perturbationResults: PerturbationResult[] | null;
  perturbationTask: PerturbationTask | null;
  whatIfResult: WhatIfResponse | null;
  attributionReport: AttributionReport | null;
  attributionTask: AttributionTask | null;
  attributionError: string | null;
  onTriggerPerturbation: () => void;
  onRunWhatIf: (removeIndices: number[]) => void;
  onTriggerAttribution: () => void;
}

function AlertCard({ alert }: { alert: Alert }) {
  return (
    <div className={`p-3 rounded-lg border ${
      alert.severity === 'error'
        ? 'bg-red-50 border-red-200'
        : 'bg-yellow-50 border-yellow-200'
    }`}>
      <div className="flex items-center gap-2">
        <span className="text-lg">
          {alert.severity === 'error' ? '❌' : '⚠️'}
        </span>
        <span className={`text-sm font-semibold ${
          alert.severity === 'error' ? 'text-red-700' : 'text-yellow-700'
        }`}>
          {alert.alert_type}
        </span>
      </div>
      <p className="text-sm text-gray-700 mt-1">{alert.message}</p>
      {alert.suggestion && (
        <p className="text-xs text-gray-500 mt-1">💡 {alert.suggestion}</p>
      )}
    </div>
  );
}

function QualityMetrics({ metrics }: { metrics: Record<string, any> }) {
  if (!metrics || Object.keys(metrics).length === 0) return null;

  const items = [
    { key: 'relevance', label: '相关性', color: 'bg-blue-500' },
    { key: 'diversity', label: '多样性', color: 'bg-green-500' },
    { key: 'coverage', label: '覆盖率', color: 'bg-purple-500' },
    { key: 'combined', label: '综合', color: 'bg-indigo-500' },
  ];

  return (
    <div>
      <label className="text-xs font-semibold text-gray-500 uppercase">质量指标</label>
      <div className="mt-1 space-y-2">
        {items.map(({ key, label, color }) => {
          const value = metrics[key];
          if (value === undefined) return null;
          const pct = Math.round(value * 100);
          return (
            <div key={key}>
              <div className="flex justify-between text-xs mb-0.5">
                <span className="text-gray-600">{label}</span>
                <span className={`font-medium ${pct < 30 ? 'text-red-500' : pct < 60 ? 'text-yellow-600' : 'text-green-600'}`}>
                  {pct}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-1.5">
                <div className={`${color} h-1.5 rounded-full`} style={{ width: `${pct}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function DiffView({ original, modified }: { original: string; modified: string }) {
  return (
    <div className="space-y-3">
      <div className="p-3 bg-gray-50 border border-gray-200 rounded text-sm whitespace-pre-wrap text-gray-700">
        <div className="text-xs font-semibold text-gray-500 mb-1">原答案</div>
        {original}
      </div>
      <div className="p-3 bg-orange-50 border border-orange-200 rounded text-sm whitespace-pre-wrap text-gray-800">
        <div className="text-xs font-semibold text-orange-600 mb-1">移除片段后</div>
        {modified}
      </div>
    </div>
  );
}

export default function DetailPanel({
  step,
  session,
  streamingAnswer,
  perturbationResults,
  perturbationTask,
  whatIfResult,
  attributionReport,
  attributionTask,
  attributionError,
  onTriggerPerturbation,
  onRunWhatIf,
  onTriggerAttribution,
}: Props) {
  const { highlightedSegmentIndex, highlightChunks, clearHighlights } = useHighlightContext();
  const [selectedRemoveIndices, setSelectedRemoveIndices] = useState<number[]>([]);
  const [whatIfLoading, setWhatIfLoading] = useState(false);

  const allChunks = useMemo(() => {
    if (!session) return [];
    const chunks: { chunk: any; importance?: number; isApproximate?: boolean }[] = [];
    session.steps.forEach((s) => {
      s.chunks?.forEach((c) => {
        const found = perturbationResults?.find(r => r.chunk_index === c.chunk_index);
        chunks.push({
          chunk: c,
          importance: found ? found.importance_score : (c.importance_score ?? undefined),
          isApproximate: found?.is_approximate,
        });
      });
    });
    return chunks;
  }, [session, perturbationResults]);

  const sortedChunks = useMemo(() => {
    return [...allChunks].sort((a, b) => (b.importance ?? 0) - (a.importance ?? 0));
  }, [allChunks]);

  const handleToggleChunk = (chunkIndex: number) => {
    setSelectedRemoveIndices((prev) =>
      prev.includes(chunkIndex)
        ? prev.filter((i) => i !== chunkIndex)
        : [...prev, chunkIndex]
    );
  };

  const handleRunWhatIf = async () => {
    if (selectedRemoveIndices.length === 0) return;
    setWhatIfLoading(true);
    await onRunWhatIf(selectedRemoveIndices);
    setWhatIfLoading(false);
  };

  if (!step && !session) {
    return (
      <div className="flex items-center justify-center h-full bg-white border-l border-gray-200">
        <p className="text-sm text-gray-400">点击流程节点查看详情</p>
      </div>
    );
  }

  if (!step && session) {
    const allAlerts = session.alerts || [];
    const isPerturbationRunning = !!perturbationTask && perturbationTask.status === 'running';
    const isPerturbationPending = !!perturbationTask && perturbationTask.status === 'pending';
    const isAttributionRunning = !!attributionTask && attributionTask.status === 'running';
    const isAttributionPending = !!attributionTask && attributionTask.status === 'pending';

    return (
      <div className="h-full bg-white border-l border-gray-200 overflow-y-auto p-4">
        <h3 className="text-lg font-bold text-gray-800 mb-3">查询概览</h3>
        <div className="space-y-3">
          <div>
            <label className="text-xs font-semibold text-gray-500 uppercase">查询</label>
            <p className="text-sm text-gray-800 mt-1 p-2 bg-gray-50 rounded">{session.query}</p>
          </div>
          <div>
            <label className="text-xs font-semibold text-gray-500 uppercase">状态</label>
            <span className={`ml-2 text-xs px-2 py-0.5 rounded ${
              session.status === 'completed' ? 'bg-green-100 text-green-700' :
              session.status === 'failed' ? 'bg-red-100 text-red-700' :
              'bg-yellow-100 text-yellow-700'
            }`}>
              {session.status}
            </span>
          </div>
          {(session.final_answer || streamingAnswer) && (
            <div>
              <label className="text-xs font-semibold text-gray-500 uppercase">最终答案</label>
              {session.execution_trace?.answer_segments ? (
                <div className="text-sm text-gray-800 mt-1 p-3 rounded-lg border border-gray-200 whitespace-pre-wrap">
                  {session.execution_trace.answer_segments.map((seg: any, idx: number) => (
                    <span
                      key={idx}
                      className={`rounded px-0.5 py-0.5 cursor-pointer transition-all ${
                        highlightedSegmentIndex === idx ? 'ring-2 ring-indigo-400 scale-105' : ''
                      }`}
                      style={{
                        background: seg.supported_by?.length > 0 ? '#dcfce7' : '#fee2e2',
                        display: 'inline',
                      }}
                      title={seg.supported_by?.length > 0 ? `引用: chunk_${seg.supported_by.join(', ')}` : '无引用支撑'}
                      onMouseEnter={() => seg.supported_by?.length > 0 && highlightChunks(seg.supported_by)}
                      onMouseLeave={clearHighlights}
                    >
                      {seg.text}
                    </span>
                  ))}
                </div>
              ) : streamingAnswer ? (
                <div className="text-sm text-gray-800 mt-1 p-3 bg-green-50 border border-green-200 rounded-lg whitespace-pre-wrap">
                  {streamingAnswer}
                  {session.status === 'running' && (
                    <span className="inline-block w-0.5 h-4 ml-0.5 bg-green-500 animate-pulse align-middle" />
                  )}
                </div>
              ) : (
                <div className="text-sm text-gray-800 mt-1 p-3 bg-green-50 border border-green-200 rounded-lg whitespace-pre-wrap">
                  {session.final_answer}
                </div>
              )}
            </div>
          )}

          {/* Quality Metrics from execution_trace */}
          {session.execution_trace?.quality_metrics && (
            <QualityMetrics metrics={session.execution_trace.quality_metrics} />
          )}

          {/* Perturbation Analysis Trigger */}
          {session.status === 'completed' && (
            <div className="p-3 bg-indigo-50 border border-indigo-200 rounded-lg">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-indigo-800">扰动分析</span>
                <button
                  onClick={onTriggerPerturbation}
                  disabled={isPerturbationRunning || isPerturbationPending}
                  className={`text-xs px-2 py-1 rounded ${
                    isPerturbationRunning || isPerturbationPending
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      : 'bg-indigo-600 text-white hover:bg-indigo-700'
                  }`}
                >
                  {isPerturbationPending ? '排队中...' : isPerturbationRunning ? '分析中...' : perturbationResults ? '重新分析' : '开始分析'}
                </button>
              </div>
              {perturbationTask && perturbationTask.status === 'running' && (
                <div className="mt-2">
                  <div className="text-xs text-indigo-600">
                    进度: {perturbationTask.progress} / {perturbationTask.total}
                  </div>
                  <div className="w-full bg-indigo-200 rounded-full h-1.5 mt-1">
                    <div
                      className="bg-indigo-600 h-1.5 rounded-full"
                      style={{ width: `${perturbationTask.total > 0 ? (perturbationTask.progress / perturbationTask.total) * 100 : 0}%` }}
                    />
                  </div>
                </div>
              )}
              {perturbationTask && perturbationTask.status === 'failed' && (
                <p className="text-xs text-red-600 mt-1">分析失败: {perturbationTask.error}</p>
              )}
            </div>
          )}

          {/* Importance Ranking Panel */}
          {sortedChunks.length > 0 && (
            <div>
              <label className="text-xs font-semibold text-gray-500 uppercase">
                片段重要性排序 ({sortedChunks.length})
              </label>
              <div className="mt-1 space-y-2 max-h-48 overflow-y-auto pr-1">
                {sortedChunks.map(({ chunk, importance, isApproximate }) => (
                  <div
                    key={chunk.id ?? chunk.chunk_index}
                    className="p-2 rounded border text-xs"
                    style={{
                      borderColor: (importance ?? 0) >= 0.7 ? '#fdba74' : (importance ?? 0) >= 0.4 ? '#93c5fd' : '#e5e7eb',
                      background: (importance ?? 0) >= 0.7 ? '#fff7ed' : (importance ?? 0) >= 0.4 ? '#eff6ff' : '#f9fafb',
                    }}
                  >
                    <div className="flex justify-between items-center">
                      <span className="font-medium text-gray-700">
                        片段 {chunk.chunk_index + 1}
                      </span>
                      <span
                        className={`font-bold ${
                          (importance ?? 0) >= 0.7 ? 'text-orange-600' :
                          (importance ?? 0) >= 0.4 ? 'text-blue-600' : 'text-gray-500'
                        }`}
                      >
                        {importance !== undefined ? `${(importance * 100).toFixed(0)}%` : '—'}
                      </span>
                    </div>
                    {isApproximate && (
                      <span className="text-[10px] text-gray-400">近似</span>
                    )}
                    <p className="text-gray-600 line-clamp-2 mt-0.5">{chunk.content}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* What-if Interaction */}
          {session.status === 'completed' && allChunks.length > 0 && (
            <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
              <label className="text-xs font-semibold text-gray-500 uppercase">What-if 分析</label>
              <p className="text-xs text-gray-500 mt-0.5 mb-2">勾选要移除的片段，查看对答案的影响</p>
              <div className="space-y-1 max-h-32 overflow-y-auto pr-1">
                {allChunks.map(({ chunk, importance }) => (
                  <label
                    key={chunk.id ?? chunk.chunk_index}
                    className="flex items-center gap-2 text-xs cursor-pointer hover:bg-gray-100 rounded px-1 py-0.5"
                  >
                    <input
                      type="checkbox"
                      checked={selectedRemoveIndices.includes(chunk.chunk_index ?? -1)}
                      onChange={() => handleToggleChunk(chunk.chunk_index)}
                      className="rounded border-gray-300"
                    />
                    <span className="text-gray-700 truncate flex-1">
                      片段 {chunk.chunk_index + 1}
                    </span>
                    {importance !== undefined && (
                      <span className={`text-[10px] font-medium ${
                        importance >= 0.7 ? 'text-orange-600' :
                        importance >= 0.4 ? 'text-blue-600' : 'text-gray-400'
                      }`}>
                        {(importance * 100).toFixed(0)}%
                      </span>
                    )}
                  </label>
                ))}
              </div>
              <button
                onClick={handleRunWhatIf}
                disabled={selectedRemoveIndices.length === 0 || whatIfLoading}
                className={`mt-2 text-xs px-3 py-1.5 rounded w-full ${
                  selectedRemoveIndices.length === 0 || whatIfLoading
                    ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                    : 'bg-blue-600 text-white hover:bg-blue-700'
                }`}
              >
                {whatIfLoading ? '分析中...' : '执行扰动并对比'}
              </button>
              {whatIfResult && (
                <div className="mt-3">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-500">语义相似度</span>
                    <span className={`font-medium ${
                      whatIfResult.similarity < 0.5 ? 'text-red-600' :
                      whatIfResult.similarity < 0.8 ? 'text-yellow-600' : 'text-green-600'
                    }`}>
                      {(whatIfResult.similarity * 100).toFixed(1)}%
                    </span>
                  </div>
                  <DiffView original={whatIfResult.original_answer} modified={whatIfResult.new_answer} />
                </div>
              )}
            </div>
          )}

          {/* Causal Attribution */}
          {session.status === 'completed' && (
            <div className="p-3 bg-indigo-50 border border-indigo-200 rounded-lg">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-indigo-800">因果归因</span>
                <button
                  onClick={onTriggerAttribution}
                  disabled={isAttributionRunning || isAttributionPending}
                  className={`text-xs px-2 py-1 rounded ${
                    isAttributionRunning || isAttributionPending
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      : 'bg-indigo-600 text-white hover:bg-indigo-700'
                  }`}
                >
                  {isAttributionPending ? '排队中...' : isAttributionRunning ? '分析中...' : attributionReport ? '重新分析' : '开始分析'}
                </button>
              </div>
              {attributionTask && (isAttributionRunning || isAttributionPending) && (
                <div className="mt-2">
                  <div className="text-xs text-indigo-600">
                    进度: {attributionTask.progress} / {attributionTask.total}
                  </div>
                  <div className="w-full bg-indigo-200 rounded-full h-1.5 mt-1">
                    <div
                      className="bg-indigo-600 h-1.5 rounded-full"
                      style={{ width: `${attributionTask.total > 0 ? (attributionTask.progress / attributionTask.total) * 100 : 0}%` }}
                    />
                  </div>
                </div>
              )}
              {attributionError && (
                <p className="text-xs text-red-600 mt-1">分析失败: {attributionError}</p>
              )}
              {(attributionReport || isAttributionRunning || isAttributionPending) && (
                <div className="mt-2 h-[420px] bg-white border border-indigo-200 rounded overflow-hidden">
                  <CausalGraph
                    report={attributionReport}
                    isLoading={!attributionReport}
                    onTriggerAttribution={onTriggerAttribution}
                  />
                </div>
              )}
            </div>
          )}

          {/* All Alerts */}
          {allAlerts.length > 0 && (
            <div>
              <label className="text-xs font-semibold text-gray-500 uppercase">
                故障警告 ({allAlerts.length})
              </label>
              <div className="mt-1 space-y-2">
                {[...allAlerts]
                  .sort((a, _b) => (a.severity === 'error' ? -1 : 1))
                  .map((alert) => (
                    <AlertCard key={alert.id} alert={alert} />
                  ))}
              </div>
            </div>
          )}

          <div>
            <label className="text-xs font-semibold text-gray-500 uppercase">执行信息</label>
            <pre className="text-xs text-gray-600 mt-1 p-2 bg-gray-50 rounded overflow-x-auto">
              {JSON.stringify(session.execution_trace, null, 2)}
            </pre>
          </div>
        </div>
      </div>
    );
  }

  if (!step) return null;

  const stepNames: Record<string, string> = {
    query_parse: '查询解析',
    vector_retrieve: '向量检索',
    rerank: '重排序',
    context_build: '上下文构建',
    answer_generate: '答案生成',
  };

  const qualityMetrics = step.output_data?.quality_metrics;

  return (
    <div className="h-full bg-white border-l border-gray-200 overflow-y-auto p-4">
      <h3 className="text-lg font-bold text-gray-800 mb-3">
        {stepNames[step.step_type] || step.step_type}
      </h3>

      <div className="space-y-3">
        <div>
          <label className="text-xs font-semibold text-gray-500 uppercase">步骤ID</label>
          <p className="text-sm text-gray-700">#{step.id}</p>
        </div>

        {step.quality_score !== undefined && (
          <div>
            <label className="text-xs font-semibold text-gray-500 uppercase">质量分数</label>
            <div className="mt-1">
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${
                    step.quality_score >= 0.7 ? 'bg-green-500' :
                    step.quality_score >= 0.4 ? 'bg-yellow-500' :
                    'bg-red-500'
                  }`}
                  style={{ width: `${step.quality_score * 100}%` }}
                />
              </div>
              <span className="text-xs text-gray-500 mt-0.5 block">
                {(step.quality_score * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        )}

        {qualityMetrics && <QualityMetrics metrics={qualityMetrics} />}

        {step.duration_ms !== undefined && (
          <div>
            <label className="text-xs font-semibold text-gray-500 uppercase">耗时</label>
            <p className="text-sm text-gray-700">{step.duration_ms}ms</p>
          </div>
        )}

        <div>
          <label className="text-xs font-semibold text-gray-500 uppercase">输入数据</label>
          <pre className="text-xs text-gray-600 mt-1 p-2 bg-gray-50 rounded overflow-x-auto">
            {JSON.stringify(step.input_data, null, 2)}
          </pre>
        </div>

        <div>
          <label className="text-xs font-semibold text-gray-500 uppercase">输出数据</label>
          <pre className="text-xs text-gray-600 mt-1 p-2 bg-gray-50 rounded overflow-x-auto">
            {JSON.stringify(step.output_data, null, 2)}
          </pre>
        </div>

        {step.chunks.length > 0 && (
          <div>
            <label className="text-xs font-semibold text-gray-500 uppercase">检索片段 ({step.chunks.length})</label>
            <div className="mt-1 space-y-2">
              {step.chunks.map((chunk) => (
                <div key={chunk.id} className="p-2 bg-blue-50 border border-blue-200 rounded text-xs">
                  <div className="flex justify-between text-gray-500 mb-1">
                    <span>来源: {chunk.source || '未知'}</span>
                    {chunk.relevance_score !== undefined && (
                      <span>相关: {(chunk.relevance_score * 100).toFixed(0)}%</span>
                    )}
                  </div>
                  {chunk.importance_score !== undefined && (
                    <div className="flex justify-between text-gray-500 mb-1">
                      <span>重要性</span>
                      <span className={`font-medium ${
                        chunk.importance_score >= 0.7 ? 'text-orange-600' :
                        chunk.importance_score >= 0.4 ? 'text-blue-600' : 'text-gray-400'
                      }`}>
                        {(chunk.importance_score * 100).toFixed(0)}%
                      </span>
                    </div>
                  )}
                  <p className="text-gray-700 line-clamp-4">{chunk.content}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {step.alerts.length > 0 && (
          <div>
            <label className="text-xs font-semibold text-gray-500 uppercase">警告 ({step.alerts.length})</label>
            <div className="mt-1 space-y-2">
              {step.alerts.map((alert) => (
                <AlertCard key={alert.id} alert={alert} />
              ))}
            </div>
          </div>
        )}

        <div>
          <label className="text-xs font-semibold text-gray-500 uppercase">时间戳</label>
          <p className="text-xs text-gray-500">{new Date(step.timestamp).toLocaleString()}</p>
        </div>
      </div>
    </div>
  );
}
