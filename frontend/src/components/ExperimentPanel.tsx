import { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';
import type { TestQuery, ExperimentResult, ExperimentMetrics } from '../types';

export default function ExperimentPanel() {
  const metricsRef = useRef<HTMLDivElement>(null);
  const [dataset, setDataset] = useState<TestQuery[]>([]);
  const [results, setResults] = useState<ExperimentResult[]>([]);
  const [metrics, setMetrics] = useState<ExperimentMetrics | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0, status: 'idle' });
  const [activeTab, setActiveTab] = useState<'dataset' | 'results' | 'metrics'>('dataset');
  const [selectedDataset, setSelectedDataset] = useState<'squad' | 'chinese'>('squad');
  const [loadingDataset, setLoadingDataset] = useState(false);
  const [datasetError, setDatasetError] = useState('');

  const loadDataset = useCallback(async (name: 'squad' | 'chinese' = 'squad') => {
    setLoadingDataset(true);
    setDatasetError('');
    try {
      const res = await axios.get(`/api/experiments/dataset?dataset=${name}`);
      setDataset(res.data.queries || []);
    } catch (err) {
      console.error('Failed to load dataset:', err);
      setDatasetError('加载数据集失败，请检查后端服务是否正在运行。');
    } finally {
      setLoadingDataset(false);
    }
  }, []);

  const loadResults = useCallback(async () => {
    try {
      const res = await axios.get('/api/experiments/results');
      setResults(res.data.results || []);
    } catch (err) {
      console.error('Failed to load results:', err);
    }
  }, []);

  const loadMetrics = useCallback(async () => {
    try {
      const res = await axios.get('/api/experiments/metrics');
      setMetrics(res.data);
    } catch (err) {
      console.error('Failed to load metrics:', err);
    }
  }, []);

  useEffect(() => {
    loadDataset(selectedDataset);
  }, [loadDataset, selectedDataset]);

  // Auto-load existing results and metrics on mount
  useEffect(() => {
    loadResults();
    loadMetrics();
  }, [loadResults, loadMetrics]);

  // Poll progress when running
  useEffect(() => {
    if (!isRunning) return;
    const interval = setInterval(async () => {
      try {
        const res = await axios.get('/api/experiments/status');
        setProgress(res.data);
        if (res.data.status === 'completed' || res.data.status === 'idle') {
          setIsRunning(false);
          loadResults();
          loadMetrics();
        }
      } catch (err) {
        console.error('Poll failed:', err);
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [isRunning, loadResults, loadMetrics]);

  const handleRun = useCallback(async () => {
    try {
      await axios.post(`/api/experiments/run?dataset=${selectedDataset}`);
      setIsRunning(true);
      setProgress({ current: 0, total: dataset.length * 2, status: 'running' });
    } catch (err: any) {
      alert(`启动实验失败: ${err.response?.data?.detail || err.message}`);
    }
  }, [dataset.length, selectedDataset]);

  const handleExport = useCallback(async (sessionId: number) => {
    try {
      const res = await axios.post(`/api/experiments/export/${sessionId}`);
      const blob = new Blob([JSON.stringify(res.data.session, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `session_${sessionId}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err: any) {
      alert(`导出失败: ${err.response?.data?.detail || err.message}`);
    }
  }, []);

  const handleExportReport = useCallback(async () => {
    try {
      const res = await axios.get('/api/experiments/report');
      const blob = new Blob([res.data.report], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `raginsight_report_${new Date().toISOString().slice(0, 10)}.md`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err: any) {
      alert(`导出报告失败: ${err.response?.data?.detail || err.message}`);
    }
  }, []);

  const handleExportPDF = useCallback(async () => {
    if (!metricsRef.current) return;
    try {
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const margin = 15;
      const today = new Date().toISOString().slice(0, 10);

      // Cover page
      pdf.setFontSize(24);
      pdf.text('RAGInsight', pageWidth / 2, pageHeight / 3, { align: 'center' });
      pdf.setFontSize(16);
      pdf.text('实验报告', pageWidth / 2, pageHeight / 3 + 12, { align: 'center' });
      pdf.setFontSize(11);
      pdf.text(`日期: ${today}`, pageWidth / 2, pageHeight / 3 + 25, { align: 'center' });
      pdf.text(`数据集: ${selectedDataset}`, pageWidth / 2, pageHeight / 3 + 32, { align: 'center' });

      // Capture metrics section
      const canvas = await html2canvas(metricsRef.current, {
        scale: 2,
        backgroundColor: '#ffffff',
        logging: false,
      });

      pdf.addPage();
      const imgWidth = pageWidth - margin * 2;
      const imgHeight = (canvas.height * imgWidth) / canvas.width;

      let remainingHeight = imgHeight;
      let sourceY = 0;
      const maxContentHeight = pageHeight - margin * 2;

      while (remainingHeight > 0) {
        const sliceHeight = Math.min(remainingHeight, maxContentHeight);
        const srcCanvas = document.createElement('canvas');
        srcCanvas.width = canvas.width;
        srcCanvas.height = Math.round(sliceHeight * (canvas.width / imgWidth));
        const ctx = srcCanvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(
            canvas,
            0, sourceY, canvas.width, srcCanvas.height,
            0, 0, srcCanvas.width, srcCanvas.height,
          );
          pdf.addImage(srcCanvas.toDataURL('image/png'), 'PNG', margin, margin, imgWidth, sliceHeight);
        }
        remainingHeight -= sliceHeight;
        sourceY += srcCanvas.height;
        if (remainingHeight > 0) pdf.addPage();
      }

      pdf.save(`raginsight_report_${today}.pdf`);
    } catch (err: any) {
      alert(`导出 PDF 失败: ${err.message}`);
    }
  }, [selectedDataset]);

  return (
    <div className="flex flex-col h-full bg-gray-50 border-r border-gray-200 overflow-y-auto">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 bg-white">
        <h2 className="text-lg font-bold text-gray-800 mb-2">实验验证面板</h2>
        <div className="flex items-center gap-2 mb-2">
          <select
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value as 'squad' | 'chinese')}
            disabled={isRunning}
            className="text-sm border border-gray-300 rounded px-2 py-1 bg-white"
          >
            <option value="squad">SQuAD (英文)</option>
            <option value="chinese">唐诗宋词 (中文)</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleRun}
            disabled={isRunning || dataset.length === 0}
            className={`px-3 py-1.5 rounded text-sm font-medium ${
              isRunning
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-indigo-600 text-white hover:bg-indigo-700'
            }`}
          >
            {isRunning ? '实验运行中...' : '运行批量实验'}
          </button>
          {isRunning && (
            <div className="flex-1">
              <div className="text-xs text-gray-600">
                进度: {progress.current} / {progress.total}
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                <div
                  className="bg-indigo-600 h-2 rounded-full transition-all"
                  style={{ width: `${progress.total > 0 ? (progress.current / progress.total) * 100 : 0}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200 bg-white">
        {(['dataset', 'results', 'metrics'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`flex-1 py-2 text-xs font-medium ${
              activeTab === tab
                ? 'text-indigo-600 border-b-2 border-indigo-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            {tab === 'dataset' && `数据集 (${dataset.length})`}
            {tab === 'results' && `结果 (${results.length})`}
            {tab === 'metrics' && '实验指标'}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {activeTab === 'dataset' && (
          <DatasetTab
            loading={loadingDataset}
            error={datasetError}
            dataset={dataset}
            onRetry={() => loadDataset(selectedDataset)}
          />
        )}

        {activeTab === 'results' && (
          <div className="space-y-2">
            {results.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-sm text-gray-400 mb-3">暂无实验结果</p>
                <p className="text-xs text-gray-400 mb-3">运行批量实验来生成结果和指标数据</p>
                <button
                  onClick={handleRun}
                  disabled={isRunning}
                  className="text-xs px-4 py-1.5 bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:bg-gray-300"
                >
                  运行批量实验
                </button>
              </div>
            ) : (
              results.map((r, idx) => (
                <div key={idx} className="p-2 bg-white rounded border border-gray-200 text-xs">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-gray-700">#{r.query_id} {r.strategy}</span>
                    <span className={`px-1.5 py-0.5 rounded ${
                      r.status === 'completed' ? 'bg-green-100 text-green-700' :
                      'bg-red-100 text-red-700'
                    }`}>
                      {r.status}
                    </span>
                  </div>
                  <p className="text-gray-600 line-clamp-2">{r.query}</p>
                  {r.actual_alerts.length > 0 && (
                    <div className="mt-1 flex gap-1 flex-wrap">
                      {r.actual_alerts.map((a) => (
                        <span key={a} className="text-[10px] px-1 py-0.5 bg-yellow-100 text-yellow-700 rounded">
                          {a}
                        </span>
                      ))}
                    </div>
                  )}
                  {r.session_id && (
                    <button
                      onClick={() => handleExport(r.session_id!)}
                      className="mt-1 text-[10px] text-indigo-600 hover:underline"
                    >
                      导出 JSON
                    </button>
                  )}
                </div>
              ))
            )}
          </div>
        )}

        {activeTab === 'metrics' && (
          <div ref={metricsRef} className="space-y-4">
            {!metrics || metrics.message ? (
              <div className="text-center py-8">
                <p className="text-sm text-gray-400 mb-2">
                  {metrics?.message || '暂无实验指标'}
                </p>
                <p className="text-xs text-gray-400 mb-3">运行批量实验后，这里将展示故障检测准确率和策略对比指标</p>
                <button
                  onClick={handleRun}
                  disabled={isRunning}
                  className="text-xs px-4 py-1.5 bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:bg-gray-300"
                >
                  运行批量实验
                </button>
              </div>
            ) : (
              <>
                {/* Experiment 1: Fault Detection */}
                <div className="bg-white rounded border border-gray-200 p-3">
                  <h3 className="text-sm font-bold text-gray-800 mb-2">实验1：故障检测准确率</h3>
                  {metrics.experiment_1_fault_detection && (
                    <>
                      <div className="grid grid-cols-3 gap-2 mb-3">
                        <div className="text-center p-2 bg-gray-50 rounded">
                          <div className="text-lg font-bold text-indigo-600">
                            {(metrics.experiment_1_fault_detection.macro.precision * 100).toFixed(1)}%
                          </div>
                          <div className="text-[10px] text-gray-500">Macro Precision</div>
                        </div>
                        <div className="text-center p-2 bg-gray-50 rounded">
                          <div className="text-lg font-bold text-indigo-600">
                            {(metrics.experiment_1_fault_detection.macro.recall * 100).toFixed(1)}%
                          </div>
                          <div className="text-[10px] text-gray-500">Macro Recall</div>
                        </div>
                        <div className="text-center p-2 bg-gray-50 rounded">
                          <div className="text-lg font-bold text-indigo-600">
                            {(metrics.experiment_1_fault_detection.macro.f1 * 100).toFixed(1)}%
                          </div>
                          <div className="text-[10px] text-gray-500">Macro F1</div>
                        </div>
                      </div>
                      <table className="w-full text-[10px]">
                        <thead>
                          <tr className="text-gray-500 border-b">
                            <th className="text-left py-1">Alert Type</th>
                            <th className="text-right py-1">P</th>
                            <th className="text-right py-1">R</th>
                            <th className="text-right py-1">F1</th>
                            <th className="text-right py-1">TP</th>
                            <th className="text-right py-1">FP</th>
                            <th className="text-right py-1">FN</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(metrics.experiment_1_fault_detection.per_alert_type).map(([type, m]) => (
                            <tr key={type} className="border-b border-gray-100">
                              <td className="py-1">{type}</td>
                              <td className="text-right">{(m.precision * 100).toFixed(0)}%</td>
                              <td className="text-right">{(m.recall * 100).toFixed(0)}%</td>
                              <td className="text-right">{(m.f1 * 100).toFixed(0)}%</td>
                              <td className="text-right">{m.tp}</td>
                              <td className="text-right">{m.fp}</td>
                              <td className="text-right">{m.fn}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </>
                  )}
                </div>

                {/* Experiment 2: Strategy Comparison */}
                <div className="bg-white rounded border border-gray-200 p-3">
                  <h3 className="text-sm font-bold text-gray-800 mb-2">实验2：跨检索架构对比</h3>
                  {metrics.experiment_2_strategy_comparison && (
                    <div className="space-y-2">
                      {Object.entries(metrics.experiment_2_strategy_comparison.strategies).map(([strategy, s]) => (
                        <div key={strategy} className="p-2 bg-gray-50 rounded">
                          <div className="font-medium text-gray-700 mb-1">{strategy} ({s.count} 次)</div>
                          <div className="grid grid-cols-4 gap-1 text-[10px] mb-1">
                            <div className="text-center">
                              <div className="font-bold text-blue-600">{(s.avg_relevance * 100).toFixed(0)}%</div>
                              <div className="text-gray-500">相关性</div>
                            </div>
                            <div className="text-center">
                              <div className="font-bold text-green-600">{(s.avg_coverage * 100).toFixed(0)}%</div>
                              <div className="text-gray-500">覆盖率</div>
                            </div>
                            <div className="text-center">
                              <div className="font-bold text-purple-600">{(s.avg_diversity * 100).toFixed(0)}%</div>
                              <div className="text-gray-500">多样性</div>
                            </div>
                            <div className="text-center">
                              <div className="font-bold text-gray-600">{s.avg_duration_ms.toFixed(0)}ms</div>
                              <div className="text-gray-500">耗时</div>
                            </div>
                          </div>
                          <div className="border-t border-gray-200 pt-1 mt-1">
                            <div className="text-[10px] text-gray-400 mb-0.5">LLM-as-a-Judge 评估</div>
                            <div className="grid grid-cols-2 gap-1 text-[10px]">
                              <div className="text-center">
                                <div className="font-bold text-emerald-600">{(s.avg_faithfulness * 100).toFixed(0)}%</div>
                                <div className="text-gray-500">忠实度</div>
                              </div>
                              <div className="text-center">
                                <div className="font-bold text-amber-600">{(s.avg_answer_relevance * 100).toFixed(0)}%</div>
                                <div className="text-gray-500">答案相关性</div>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                <div className="mt-4 space-y-2">
                  <button
                    onClick={handleExportReport}
                    className="w-full text-xs px-3 py-2 rounded bg-indigo-600 text-white hover:bg-indigo-700"
                  >
                    导出 Markdown 报告
                  </button>
                  <button
                    onClick={handleExportPDF}
                    className="w-full text-xs px-3 py-2 rounded bg-emerald-600 text-white hover:bg-emerald-700"
                  >
                    导出 PDF 报告
                  </button>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function DatasetTab({ loading, error, dataset, onRetry }: {
  loading: boolean;
  error: string;
  dataset: TestQuery[];
  onRetry: () => void;
}) {
  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-indigo-600 mr-2" />
        <span className="text-sm text-gray-400">加载数据集...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-8">
        <p className="text-sm text-red-500 mb-2">{error}</p>
        <button onClick={onRetry} className="text-xs px-3 py-1 bg-indigo-100 text-indigo-700 rounded hover:bg-indigo-200">
          重试
        </button>
      </div>
    );
  }

  if (dataset.length === 0) {
    return <p className="text-sm text-gray-400 text-center py-8">暂无测试数据</p>;
  }

  return (
    <div className="space-y-2">
      {dataset.map((q) => (
        <div key={q.id} className="p-2 bg-white rounded border border-gray-200 text-xs">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-medium text-gray-700">#{q.id}</span>
            <span className={`px-1.5 py-0.5 rounded text-[10px] ${
              q.category === 'simple_fact' ? 'bg-green-100 text-green-700' :
              q.category === 'multi_hop' ? 'bg-blue-100 text-blue-700' :
              q.category === 'comparison' ? 'bg-purple-100 text-purple-700' :
              'bg-red-100 text-red-700'
            }`}>
              {q.category}
            </span>
            <span className="text-gray-400">深度: {q.expected_depth}</span>
          </div>
          <p className="text-gray-800">{q.query}</p>
          {q.expected_alerts.length > 0 && (
            <div className="mt-1 flex gap-1 flex-wrap">
              {q.expected_alerts.map((a) => (
                <span key={a} className="text-[10px] px-1 py-0.5 bg-orange-100 text-orange-700 rounded">
                  预期: {a}
                </span>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
