import type { RootCause } from '../types';

interface Props {
  rootCause: RootCause | null;
}

const causeTypeConfig: Record<string, { label: string; color: string; bgColor: string; icon: string }> = {
  retrieval_failure: { label: '检索失败', color: '#ef4444', bgColor: '#fef2f2', icon: '🔴' },
  incomplete_knowledge: { label: '知识不完整', color: '#f59e0b', bgColor: '#fffbeb', icon: '🟡' },
  hallucination_risk: { label: '幻觉风险', color: '#8b5cf6', bgColor: '#f5f3ff', icon: '🟣' },
  strategy_mismatch: { label: '策略不匹配', color: '#3b82f6', bgColor: '#eff6ff', icon: '🔵' },
  no_issue: { label: '无问题', color: '#10b981', bgColor: '#ecfdf5', icon: '🟢' },
};

export default function RootCausePanel({ rootCause }: Props) {
  if (!rootCause) return null;

  const config = causeTypeConfig[rootCause.root_cause_type] || {
    label: rootCause.root_cause_label,
    color: '#6b7280',
    bgColor: '#f9fafb',
    icon: '⚪',
  };

  return (
    <div
      className="px-4 py-3 border-b border-gray-200"
      style={{ backgroundColor: config.bgColor }}
    >
      <div className="flex items-center gap-2 mb-2">
        <span className="text-lg">{config.icon}</span>
        <span
          className="text-sm font-bold px-2 py-0.5 rounded"
          style={{ color: config.color, border: `1px solid ${config.color}` }}
        >
          {config.label}
        </span>
        <span className={`text-xs px-1.5 py-0.5 rounded ${
          rootCause.severity === 'error' ? 'bg-red-100 text-red-700' : 'bg-yellow-100 text-yellow-700'
        }`}>
          {rootCause.severity === 'error' ? '严重' : '警告'}
        </span>
        <span className="text-xs text-gray-500 ml-auto">
          {rootCause.error_count} 错误 / {rootCause.warning_count} 警告
        </span>
      </div>
      
      <p className="text-sm text-gray-700 mb-2">{rootCause.explanation}</p>
      
      {rootCause.suggestions.length > 0 && (
        <div className="mt-2">
          <p className="text-xs font-semibold text-gray-500 mb-1">修复建议：</p>
          <ul className="space-y-1">
            {rootCause.suggestions.map((suggestion, idx) => (
              <li key={idx} className="text-xs text-gray-600 flex items-start gap-1">
                <span className="text-blue-500 mt-0.5">💡</span>
                <span>{suggestion}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
