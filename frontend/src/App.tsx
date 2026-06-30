import { useState, useCallback, useEffect, useRef } from 'react';
import Layout from './components/Layout';
import QueryPanel from './components/QueryPanel';
import FlowChart from './components/FlowChart';
import DetailPanel from './components/DetailPanel';
import RootCausePanel from './components/RootCausePanel';
import StrategyPanel from './components/StrategyPanel';
import CompareView from './components/CompareView';
import ExperimentPanel from './components/ExperimentPanel';
import { useSSE } from './hooks/useSSE';
import { HighlightProvider } from './contexts/HighlightContext';
import type {
  Session, SessionListItem, Step, StrategyInfo, RootCause,
  PerturbationResult, PerturbationTask, WhatIfResponse,
  ComplexityAnalysis,
} from './types';
import axios from 'axios';

const defaultStrategies: StrategyInfo[] = [
  { id: 'vector', name: '向量检索', description: '基于语义相似度的向量检索', icon: 'search' },
  { id: 'hybrid', name: '混合检索', description: '向量检索 + 关键词检索融合', icon: 'git-merge' },
  { id: 'graph', name: '图检索', description: '基于知识图谱的多跳推理检索', icon: 'share-2' },
];

function App() {
  const [sessions, setSessions] = useState<SessionListItem[]>([]);
  const [currentSession, setCurrentSession] = useState<Session | null>(null);
  const [selectedStep, setSelectedStep] = useState<Step | null>(null);
  const [rootCause, setRootCause] = useState<RootCause | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [strategies] = useState<StrategyInfo[]>(defaultStrategies);

  // Phase 3: Perturbation state
  const [perturbationTask, setPerturbationTask] = useState<PerturbationTask | null>(null);
  const [perturbationResults, setPerturbationResults] = useState<PerturbationResult[] | null>(null);
  const [whatIfResult, setWhatIfResult] = useState<WhatIfResponse | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Phase 4: Complexity & strategy comparison
  const [complexityAnalysis, setComplexityAnalysis] = useState<ComplexityAnalysis | null>(null);
  const [compareSession, setCompareSession] = useState<Session | null>(null);
  const [compareRootCause, setCompareRootCause] = useState<RootCause | null>(null);
  const [lastQuery, setLastQuery] = useState('');
  const [lastCollection, setLastCollection] = useState<string | undefined>(undefined);
  const [selectedStrategy, setSelectedStrategy] = useState('vector');

  const { connect } = useSSE();

  useEffect(() => {
    loadSessions();
  }, []);

  // Poll perturbation task status
  useEffect(() => {
    if (!perturbationTask || perturbationTask.status === 'completed' || perturbationTask.status === 'failed') {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }

    pollRef.current = setInterval(async () => {
      try {
        const res = await axios.get(`/api/tasks/${perturbationTask.task_id}/status`);
        const task = res.data as PerturbationTask;
        setPerturbationTask(task);
        if (task.status === 'completed' && task.result) {
          setPerturbationResults(task.result);
          setCurrentSession((prev) => {
            if (!prev) return prev;
            return {
              ...prev,
              steps: prev.steps.map((s) => ({
                ...s,
                chunks: s.chunks.map((c) => {
                  const found = task.result?.find(
                    (r: PerturbationResult) => r.chunk_index === c.chunk_index
                  );
                  return found ? { ...c, importance_score: found.importance_score } : c;
                }),
              })),
            };
          });
          clearInterval(pollRef.current!);
          pollRef.current = null;
        } else if (task.status === 'failed') {
          clearInterval(pollRef.current!);
          pollRef.current = null;
        }
      } catch (err) {
        console.error('Poll task failed:', err);
      }
    }, 1500);

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [perturbationTask?.task_id, perturbationTask?.status]);

  const loadSessions = useCallback(async () => {
    try {
      const res = await axios.get<SessionListItem[]>('/api/sessions');
      setSessions(res.data);
    } catch (err) {
      console.error('Failed to load sessions:', err);
    }
  }, []);

  const runQuery = useCallback((query: string, strategy: string, target: 'main' | 'compare', collection?: string) => {
    setIsLoading(true);
    if (target === 'main') {
      setCurrentSession(null);
      setRootCause(null);
      setCompareSession(null);
      setCompareRootCause(null);
      setPerturbationTask(null);
      setPerturbationResults(null);
      setWhatIfResult(null);
      setLastQuery(query);
      setSelectedStrategy(strategy);
      setLastCollection(collection);
    }

    const cleanup = connect(
      query,
      strategy,
      (event) => {
        const setter = target === 'main' ? setCurrentSession : setCompareSession;
        const rcSetter = target === 'main' ? setRootCause : setCompareRootCause;

        if (event.event === 'step') {
          setter((prev) => {
            if (!prev) {
              return {
                id: event.data.session_id || -1,
                query,
                status: 'running',
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
                execution_trace: {},
                steps: [event.data as Step],
                alerts: [],
              };
            }
            const exists = prev.steps.some(s => s.id === event.data.id);
            if (exists) {
              return {
                ...prev,
                steps: prev.steps.map(s => s.id === event.data.id ? event.data as Step : s),
              };
            }
            return {
              ...prev,
              steps: [...prev.steps, event.data as Step],
            };
          });
        } else if (event.event === 'alert') {
          setter((prev) => {
            if (!prev) return prev;
            return {
              ...prev,
              alerts: [...prev.alerts, event.data as any],
            };
          });
        } else if (event.event === 'root_cause') {
          rcSetter(event.data as RootCause);
          setter((prev) => {
            if (!prev) return prev;
            return {
              ...prev,
              execution_trace: {
                ...prev.execution_trace,
                root_cause: event.data,
              },
            };
          });
        } else if (event.event === 'done') {
          const sessionId = event.data.session_id;
          if (sessionId) {
            axios.get<Session>(`/api/sessions/${sessionId}`).then((res) => {
              setter(res.data);
              if (target === 'main') loadSessions();
            });
          }
          setIsLoading(false);
          cleanup?.();
        } else if (event.event === 'error') {
          setIsLoading(false);
          alert(`${target === 'compare' ? '对比查询' : '查询'}失败: ${event.data.message}`);
          cleanup?.();
        }
      },
      () => {
        if (target === 'main') setIsLoading(false);
      },
      (err) => {
        setIsLoading(false);
        alert(`${target === 'compare' ? '对比查询' : '连接'}错误: ${err}`);
      },
      collection,
    );
  }, [connect, loadSessions]);

  const handleQuery = useCallback((query: string, strategy: string, collection?: string) => {
    runQuery(query, strategy, 'main', collection);
  }, [runQuery]);

  const handleSelectSession = useCallback(async (id: number) => {
    try {
      setIsLoading(true);
      const res = await axios.get<Session>(`/api/sessions/${id}`);
      setCurrentSession(res.data);
      setRootCause(res.data.execution_trace?.root_cause || null);
      setCompareSession(null);
      setCompareRootCause(null);
      setSelectedStep(null);
      setPerturbationTask(null);
      setPerturbationResults(null);
      setWhatIfResult(null);
      setIsLoading(false);
    } catch (err) {
      setIsLoading(false);
      alert('加载会话失败');
    }
  }, []);

  const handleNodeClick = useCallback((step: Step) => {
    setSelectedStep(step);
  }, []);

  const handleTriggerPerturbation = useCallback(async () => {
    if (!currentSession) return;
    try {
      const res = await axios.post(`/api/sessions/${currentSession.id}/perturbation`);
      setPerturbationTask(res.data as PerturbationTask);
      setPerturbationResults(null);
      setWhatIfResult(null);
    } catch (err: any) {
      alert(`触发扰动分析失败: ${err.response?.data?.detail || err.message}`);
    }
  }, [currentSession]);

  const handleRunWhatIf = useCallback(async (removeIndices: number[]) => {
    if (!currentSession) return;
    try {
      const res = await axios.post<WhatIfResponse>(
        `/api/sessions/${currentSession.id}/perturbation/what-if`,
        { remove_chunk_indices: removeIndices }
      );
      setWhatIfResult(res.data);
    } catch (err: any) {
      alert(`What-if 分析失败: ${err.response?.data?.detail || err.message}`);
    }
  }, [currentSession]);

  const handleAnalyzeComplexity = useCallback((analysis: ComplexityAnalysis) => {
    setComplexityAnalysis(analysis);
  }, []);

  const handleCompareStrategy = useCallback((strategyId: string) => {
    if (!lastQuery) {
      alert('请先执行一次查询');
      return;
    }
    if (strategyId === selectedStrategy) {
      alert('当前已经是该策略，请选择其他策略对比');
      return;
    }
    setCompareSession(null);
    setCompareRootCause(null);
    runQuery(lastQuery, strategyId, 'compare', lastCollection);
  }, [lastQuery, selectedStrategy, lastCollection, runQuery]);

  const isComparing = !!compareSession;
  const [leftTab, setLeftTab] = useState<'query' | 'experiment'>('query');

  return (
    <HighlightProvider>
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-white">
      {/* Root Cause Panel */}
      <RootCausePanel rootCause={rootCause} />

      {/* Strategy Recommendation Panel */}
      <StrategyPanel
        analysis={complexityAnalysis}
        onSelectStrategy={(id) => {
          setComplexityAnalysis((prev) => prev ? { ...prev, recommended_strategy: id } : prev);
          setSelectedStrategy(id);
        }}
        onCompareStrategies={() => {
          if (complexityAnalysis) {
            const alt = complexityAnalysis.alternatives[0];
            if (alt) handleCompareStrategy(alt.id);
          }
        }}
        currentStrategy={selectedStrategy}
      />

      {/* Main Layout */}
      <div className="flex flex-1 min-h-0">
        <Layout
          leftPanel={
            <div className="flex flex-col h-full">
              <div className="flex border-b border-gray-200 bg-white">
                <button
                  onClick={() => setLeftTab('query')}
                  className={`flex-1 py-2 text-xs font-medium ${
                    leftTab === 'query'
                      ? 'text-indigo-600 border-b-2 border-indigo-600'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  查询
                </button>
                <button
                  onClick={() => setLeftTab('experiment')}
                  className={`flex-1 py-2 text-xs font-medium ${
                    leftTab === 'experiment'
                      ? 'text-indigo-600 border-b-2 border-indigo-600'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  实验
                </button>
              </div>
              <div className="flex-1 min-h-0">
                {leftTab === 'query' ? (
                  <QueryPanel
                    sessions={sessions}
                    onQuery={handleQuery}
                    onSelectSession={handleSelectSession}
                    onAnalyzeComplexity={handleAnalyzeComplexity}
                    isLoading={isLoading}
                    strategies={strategies}
                    selectedStrategy={selectedStrategy}
                    onStrategyChange={setSelectedStrategy}
                  />
                ) : (
                  <ExperimentPanel />
                )}
              </div>
            </div>
          }
          centerPanel={
            isComparing ? (
              <CompareView
                sessions={[
                  {
                    label: `${strategies.find(s => s.id === selectedStrategy)?.name || selectedStrategy} (当前)`,
                    session: currentSession,
                    rootCause,
                    perturbationResults,
                  },
                  {
                    label: `${strategies.find(s => s.id === compareSession?.execution_trace?.actual_strategy)?.name || '对比'} (对比)`,
                    session: compareSession,
                    rootCause: compareRootCause,
                    perturbationResults: null,
                  },
                ]}
                onNodeClick={handleNodeClick}
              />
            ) : (
              <FlowChart
                session={currentSession}
                rootCause={rootCause}
                perturbationResults={perturbationResults}
                onNodeClick={handleNodeClick}
              />
            )
          }
          rightPanel={
            <DetailPanel
              step={selectedStep}
              session={selectedStep ? null : currentSession}
              perturbationResults={perturbationResults}
              perturbationTask={perturbationTask}
              whatIfResult={whatIfResult}
              onTriggerPerturbation={handleTriggerPerturbation}
              onRunWhatIf={handleRunWhatIf}
            />
          }
        />
      </div>
    </div>
    </HighlightProvider>
  );
}

export default App;
