import { useState, useMemo } from 'react';
import type { AttributionReport, CausalGraphNode, CausalGraphEdge } from '../types';

interface Props {
  report: AttributionReport | null;
  isLoading?: boolean;
  onTriggerAttribution?: () => void;
}

const NODE_COLORS: Record<string, { fill: string; stroke: string; text: string }> = {
  input: { fill: '#e0e7ff', stroke: '#6366f1', text: '#3730a3' },
  decision: { fill: '#fef3c7', stroke: '#f59e0b', text: '#92400e' },
  process: { fill: '#d1fae5', stroke: '#10b981', text: '#065f46' },
  outcome: { fill: '#fce7f3', stroke: '#ec4899', text: '#9d174d' },
};

const NODE_POSITIONS: Record<string, { x: number; y: number }> = {
  query: { x: 250, y: 40 },
  strategy: { x: 100, y: 140 },
  retrieval: { x: 250, y: 140 },
  context: { x: 400, y: 140 },
  llm: { x: 250, y: 240 },
  quality: { x: 250, y: 340 },
};

const NODE_LABELS: Record<string, string> = {
  query: 'Query',
  strategy: 'Strategy',
  retrieval: 'Retrieval',
  context: 'Context',
  llm: 'LLM Gen',
  quality: 'Quality',
};

export default function CausalGraph({ report, isLoading, onTriggerAttribution }: Props) {
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  const nodes: (CausalGraphNode & { x: number; y: number })[] = useMemo(() => {
    if (!report?.causal_graph?.nodes) return [];
    return report.causal_graph.nodes
      .filter(n => NODE_POSITIONS[n.id])
      .map(n => ({
        ...n,
        x: NODE_POSITIONS[n.id]?.x ?? 0,
        y: NODE_POSITIONS[n.id]?.y ?? 0,
      }));
  }, [report]);

  const edges: CausalGraphEdge[] = useMemo(() => {
    if (!report?.causal_graph?.edges) return [];
    return report.causal_graph.edges;
  }, [report]);

  const maxAttribution = useMemo(() => {
    if (!nodes.length) return 0.01;
    return Math.max(...nodes.map(n => n.attribution || 0), 0.01);
  }, [nodes]);

  const nodeRadius = (attribution: number) => {
    const base = 24;
    const scale = maxAttribution > 0 ? (attribution || 0) / maxAttribution : 0;
    return base + scale * 18;
  };

  const getNodeColor = (nodeId: string, type: string) => {
    if (hoveredNode === nodeId) return { fill: '#e0e7ff', stroke: '#4f46e5', text: '#3730a3' };
    const colors = NODE_COLORS[type] || NODE_COLORS.process;
    return colors;
  };

  if (!report && !isLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-4">
        <p className="text-sm text-gray-500 mb-3 text-center">
          Run causal attribution to see which pipeline components contributed to the answer quality.
        </p>
        {onTriggerAttribution && (
          <button
            onClick={onTriggerAttribution}
            className="px-4 py-2 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 transition-colors"
          >
            Run Attribution Analysis
          </button>
        )}
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="flex flex-col items-center gap-2">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
          <p className="text-sm text-gray-500">Running attribution analysis...</p>
        </div>
      </div>
    );
  }

  const svgWidth = 500;
  const svgHeight = 420;

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-200">
        <h3 className="text-xs font-semibold text-gray-700 uppercase tracking-wide">
          Causal Attribution
        </h3>
        {report && (
          <span className="text-xs text-gray-500">
            Q={report.original_quality.toFixed(3)} | {report.total_interventions} interventions
          </span>
        )}
      </div>

      <div className="flex-1 overflow-auto p-2">
        <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="w-full h-auto">
          <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#9ca3af" />
            </marker>
            <filter id="glow">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>

          {/* Edges */}
          {edges.map((edge, i) => {
            const fromPos = NODE_POSITIONS[edge.from];
            const toPos = NODE_POSITIONS[edge.to];
            if (!fromPos || !toPos) return null;

            const dx = toPos.x - fromPos.x;
            const dy = toPos.y - fromPos.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            const nx = dx / dist;
            const ny = dy / dist;

            const fromR = nodeRadius(nodes.find(n => n.id === edge.from)?.attribution || 0);
            const toR = nodeRadius(nodes.find(n => n.id === edge.to)?.attribution || 0);

            const x1 = fromPos.x + nx * fromR;
            const y1 = fromPos.y + ny * fromR;
            const x2 = toPos.x - nx * (toR + 8);
            const y2 = toPos.y - ny * (toR + 8);

            const midX = (x1 + x2) / 2;
            const midY = (y1 + y2) / 2;

            const isHighlighted = hoveredNode === edge.from || hoveredNode === edge.to;

            return (
              <g key={`edge-${i}`}>
                <line
                  x1={x1} y1={y1} x2={x2} y2={y2}
                  stroke={isHighlighted ? '#6366f1' : '#d1d5db'}
                  strokeWidth={isHighlighted ? 2 : 1}
                  markerEnd="url(#arrowhead)"
                />
                <text
                  x={midX} y={midY - 6}
                  textAnchor="middle"
                  className="text-[9px] fill-gray-400"
                  style={{ fontSize: '9px' }}
                >
                  {edge.label}
                </text>
              </g>
            );
          })}

          {/* Nodes */}
          {nodes.map(node => {
            const r = nodeRadius(node.attribution);
            const colors = getNodeColor(node.id, node.type);
            const isHighlighted = hoveredNode === node.id;
            const attrPct = ((node.attribution || 0) * 100).toFixed(1);

            return (
              <g
                key={node.id}
                onMouseEnter={() => setHoveredNode(node.id)}
                onMouseLeave={() => setHoveredNode(null)}
                style={{ cursor: 'pointer' }}
                filter={isHighlighted ? 'url(#glow)' : undefined}
              >
                <circle
                  cx={node.x} cy={node.y} r={r}
                  fill={colors.fill}
                  stroke={colors.stroke}
                  strokeWidth={isHighlighted ? 3 : 1.5}
                />
                <text
                  x={node.x} y={node.y + 1}
                  textAnchor="middle"
                  dominantBaseline="central"
                  className="font-semibold"
                  style={{ fontSize: isHighlighted ? '12px' : '10px', fill: colors.text }}
                >
                  {NODE_LABELS[node.id] || node.label}
                </text>
                {/* Attribution badge */}
                {node.attribution > 0 && (
                  <text
                    x={node.x + r + 4}
                    y={node.y - r / 2}
                    className="font-mono"
                    style={{ fontSize: '9px', fill: '#6b7280' }}
                  >
                    {attrPct}%
                  </text>
                )}
              </g>
            );
          })}

          {/* Observables display */}
          {report?.causal_graph?.observables && Object.keys(report.causal_graph.observables).length > 0 && (
            <g transform="translate(10, 380)">
              <text x={0} y={0} style={{ fontSize: '9px', fill: '#9ca3af' }}>
                {Object.entries(report.causal_graph.observables)
                  .filter(([_, v]) => typeof v !== 'object')
                  .map(([k, v]) => `${k}: ${typeof v === 'number' ? v.toFixed(3) : v}`)
                  .join(' | ')}
              </text>
            </g>
          )}
        </svg>
      </div>

      {/* Legend & Component Attributions */}
      {report && (
        <div className="px-3 py-2 border-t border-gray-100">
          <div className="flex flex-wrap gap-2 text-xs">
            {Object.entries(report.component_attributions)
              .sort(([, a], [, b]) => b - a)
              .map(([comp, score]) => (
                <span
                  key={comp}
                  className="px-2 py-0.5 rounded-full bg-gray-100 text-gray-700 font-mono"
                  title={`${comp}: ${(score * 100).toFixed(1)}%`}
                >
                  {comp}: {(score * 100).toFixed(0)}%
                </span>
              ))}
          </div>
          {report.top_contributors.length > 0 && (
            <div className="mt-2 space-y-1 max-h-32 overflow-y-auto">
              {report.top_contributors.map((tc, i) => (
                <div key={i} className="flex items-center justify-between text-xs">
                  <span className="text-gray-600 truncate max-w-[75%]" title={tc.description}>
                    {tc.description}
                  </span>
                  <span className={`font-mono ml-2 ${tc.quality_delta >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {tc.quality_delta >= 0 ? '+' : ''}{tc.quality_delta.toFixed(4)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {onTriggerAttribution && report && (
        <div className="px-3 pb-2">
          <button
            onClick={onTriggerAttribution}
            className="w-full py-1.5 text-xs text-indigo-600 border border-indigo-200 rounded hover:bg-indigo-50 transition-colors"
          >
            Re-run Attribution
          </button>
        </div>
      )}
    </div>
  );
}
