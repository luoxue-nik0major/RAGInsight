import { useState, useCallback, useEffect } from 'react';
import type { SessionListItem, StrategyInfo, ComplexityAnalysis, CollectionInfo } from '../types';
import RadarChart from './RadarChart';
import axios from 'axios';

interface Props {
  sessions: SessionListItem[];
  onQuery: (query: string, strategy: string, collection?: string) => void;
  onSelectSession: (id: number) => void;
  onAnalyzeComplexity?: (analysis: ComplexityAnalysis) => void;
  isLoading: boolean;
  strategies: StrategyInfo[];
  selectedStrategy?: string;
  onStrategyChange?: (strategy: string) => void;
}

export default function QueryPanel({ sessions, onQuery, onSelectSession, onAnalyzeComplexity, isLoading, strategies, selectedStrategy, onStrategyChange }: Props) {
  const [query, setQuery] = useState('');
  const [strategy, setStrategy] = useState(selectedStrategy || 'vector');
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [selectedCollection, setSelectedCollection] = useState('');

  // Sync with external strategy selection
  useEffect(() => {
    if (selectedStrategy && selectedStrategy !== strategy) {
      setStrategy(selectedStrategy);
    }
  }, [selectedStrategy]);

  // Load available collections
  useEffect(() => {
    axios.get('/api/collections').then((res) => {
      const cols = res.data.collections || [];
      setCollections(cols.map((name: string) => ({ name, count: 0 })));
      if (cols.length > 0 && !selectedCollection) {
        setSelectedCollection('');  // default: auto-detect
      }
    }).catch((err) => {
      console.error('Failed to load collections:', err);
    });
  }, []);

  const [analysis, setAnalysis] = useState<ComplexityAnalysis | null>(null);
  const [analyzing, setAnalyzing] = useState(false);

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;
    onQuery(query.trim(), strategy, selectedCollection || undefined);
  }, [query, strategy, selectedCollection, isLoading, onQuery]);

  const handleAnalyze = useCallback(async () => {
    if (!query.trim() || analyzing) return;
    setAnalyzing(true);
    try {
      const res = await axios.post<ComplexityAnalysis>('/api/analyze-complexity', { query: query.trim() });
      setAnalysis(res.data);
      onAnalyzeComplexity?.(res.data);
      // Auto-select recommended strategy
      const newStrategy = res.data.recommended_strategy;
      setStrategy(newStrategy);
      onStrategyChange?.(newStrategy);
    } catch (err) {
      console.error('Complexity analysis failed:', err);
    } finally {
      setAnalyzing(false);
    }
  }, [query, analyzing, onAnalyzeComplexity, onStrategyChange]);

  return (
    <div className="flex flex-col h-full bg-gray-50 border-r border-gray-200">
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-lg font-bold text-gray-800 mb-3">RAGInsight</h2>
        <form onSubmit={handleSubmit} className="space-y-2">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="输入查询问题..."
            className="w-full p-2 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-primary-500"
            rows={3}
            disabled={isLoading}
          />
          <select
            value={selectedCollection}
            onChange={(e) => setSelectedCollection(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-lg text-sm"
            disabled={isLoading}
          >
            <option value="">自动检测数据库（默认）</option>
            {collections.map((c) => (
              <option key={c.name} value={c.name}>{c.name}</option>
            ))}
          </select>
          <select
            value={strategy}
            onChange={(e) => setStrategy(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-lg text-sm"
            disabled={isLoading}
          >
            {strategies.map((s) => (
              <option key={s.id} value={s.id}>{s.name}</option>
            ))}
          </select>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={handleAnalyze}
              disabled={analyzing || !query.trim() || isLoading}
              className="flex-1 py-1.5 px-3 bg-indigo-50 text-indigo-700 border border-indigo-200 rounded-lg hover:bg-indigo-100 disabled:bg-gray-100 disabled:text-gray-400 disabled:border-gray-200 disabled:cursor-not-allowed transition-colors text-sm"
            >
              {analyzing ? '分析中...' : '分析复杂度'}
            </button>
            <button
              type="submit"
              disabled={isLoading || !query.trim()}
              className="flex-1 py-1.5 px-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors text-sm"
            >
              {isLoading ? '检索中...' : '执行查询'}
            </button>
          </div>
        </form>

        {/* Complexity Analysis Result */}
        {analysis && (
          <div className="mt-4 p-3 bg-white rounded-lg border border-gray-200">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-gray-500 uppercase">复杂度分析</span>
              <span className="text-xs text-gray-400">{analysis.question_type}</span>
            </div>

            <RadarChart features={analysis.features} size={140} />

            <div className="mt-2 space-y-1">
              {[
                { label: '长度', score: analysis.features.length_score },
                { label: '实体', score: analysis.features.entity_score },
                { label: '关系', score: analysis.features.relation_score },
                { label: '跳数', score: analysis.features.hop_demand_score },
                { label: '语义', score: analysis.features.semantic_score },
              ].map(({ label, score }) => (
                <div key={label} className="flex items-center gap-2">
                  <span className="text-[10px] text-gray-500 w-8">{label}</span>
                  <div className="flex-1 bg-gray-100 rounded-full h-1.5">
                    <div
                      className="bg-indigo-500 h-1.5 rounded-full"
                      style={{ width: `${(score * 100).toFixed(0)}%` }}
                    />
                  </div>
                  <span className="text-[10px] text-gray-500 w-8 text-right">{(score * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>

            <div className="mt-2 pt-2 border-t border-gray-100">
              <p className="text-xs text-gray-600">
                <span className="font-medium">推荐策略：</span>
                <span className="text-indigo-700 font-semibold">{analysis.recommended_strategy_name}</span>
              </p>
            </div>
          </div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        <h3 className="text-sm font-semibold text-gray-600 mb-2">历史查询</h3>
        {sessions.length === 0 ? (
          <p className="text-sm text-gray-400">暂无查询记录</p>
        ) : (
          <div className="space-y-2">
            {sessions.map((session) => (
              <button
                key={session.id}
                onClick={() => onSelectSession(session.id)}
                className="w-full text-left p-3 bg-white rounded-lg border border-gray-200 hover:border-primary-500 hover:shadow-sm transition-all"
              >
                <p className="text-sm text-gray-800 line-clamp-2">{session.query}</p>
                <div className="flex items-center justify-between mt-1">
                  <span className={`text-xs px-1.5 py-0.5 rounded ${
                    session.status === 'completed' ? 'bg-green-100 text-green-700' :
                    session.status === 'failed' ? 'bg-red-100 text-red-700' :
                    'bg-yellow-100 text-yellow-700'
                  }`}>
                    {session.status}
                  </span>
                  <span className="text-xs text-gray-400">
                    {new Date(session.created_at).toLocaleTimeString()}
                  </span>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
