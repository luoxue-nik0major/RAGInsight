import type { ComplexityAnalysis } from '../types';

interface Props {
  analysis: ComplexityAnalysis | null;
  onSelectStrategy: (strategyId: string) => void;
  onCompareStrategies?: () => void;
  currentStrategy?: string;
}

const strategyColor: Record<string, string> = {
  vector: 'bg-blue-100 text-blue-700 border-blue-300',
  hybrid: 'bg-purple-100 text-purple-700 border-purple-300',
  graph: 'bg-orange-100 text-orange-700 border-orange-300',
};

export default function StrategyPanel({ analysis, onSelectStrategy, onCompareStrategies, currentStrategy }: Props) {
  if (!analysis) return null;

  return (
    <div className="px-4 py-3 border-b border-gray-200 bg-white">
      <div className="flex items-start gap-4">
        {/* Complexity score badge */}
        <div className="shrink-0 text-center">
          <div className="text-2xl font-bold text-indigo-600">
            {(analysis.complexity_score * 100).toFixed(0)}%
          </div>
          <div className="text-[10px] text-gray-500">复杂度</div>
        </div>

        {/* Recommendation */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className={`text-xs px-2 py-0.5 rounded-full border font-semibold ${
              strategyColor[analysis.recommended_strategy] || 'bg-gray-100 text-gray-700 border-gray-300'
            }`}>
              推荐：{analysis.recommended_strategy_name}
            </span>
            <span className="text-xs text-gray-400">
              {analysis.question_type}
            </span>
          </div>
          <p className="text-sm text-gray-700 mt-1">{analysis.reason}</p>

          {/* Strategy buttons */}
          <div className="flex items-center gap-2 mt-2">
            {analysis.alternatives.map((alt) => (
              <button
                key={alt.id}
                onClick={() => onSelectStrategy(alt.id)}
                className={`text-xs px-2 py-1 rounded border transition-colors ${
                  currentStrategy === alt.id
                    ? 'bg-gray-800 text-white border-gray-800'
                    : 'bg-white text-gray-600 border-gray-300 hover:border-gray-400'
                }`}
                title={alt.description}
              >
                {alt.name}
              </button>
            ))}
            {onCompareStrategies && (
              <button
                onClick={onCompareStrategies}
                className="text-xs px-2 py-1 rounded border border-indigo-300 text-indigo-700 bg-indigo-50 hover:bg-indigo-100 transition-colors"
              >
                策略对比
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
