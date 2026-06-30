import { useMemo, useCallback, useState, useEffect, useRef } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  Node,
  Edge,
  Position,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { useHighlightContext } from '../contexts/HighlightContext';
import type { Step, Session, RootCause, PerturbationResult } from '../types';

interface Props {
  session: Session | null;
  rootCause: RootCause | null;
  perturbationResults: PerturbationResult[] | null;
  onNodeClick: (step: Step) => void;
}

const SLOW_STEP_MS = 2000;
const PLAYBACK_INTERVAL_MS = 800;

const stepTypeConfig: Record<string, { label: string; color: string; icon: string }> = {
  query_parse: { label: 'Query Parse', color: '#6366f1', icon: '🔍' },
  vector_retrieve: { label: 'Vector Retrieve', color: '#3b82f6', icon: '📚' },
  rerank: { label: 'Rerank', color: '#8b5cf6', icon: '📊' },
  context_build: { label: 'Context Build', color: '#10b981', icon: '🧩' },
  answer_generate: { label: 'Answer Generate', color: '#f59e0b', icon: '✨' },
};

function getNodeStyle(step: Step, isFaulty: boolean): React.CSSProperties {
  const config = stepTypeConfig[step.step_type] || { color: '#6b7280', icon: '📄' };
  const alerts = step.alerts || [];
  const hasWarning = alerts.some(a => a.severity === 'warning');
  const hasError = alerts.some(a => a.severity === 'error');

  let borderColor = config.color;
  let borderWidth = 2;
  let opacity = 1;

  if (hasError) {
    borderColor = '#ef4444';
    borderWidth = 3;
  } else if (hasWarning) {
    borderColor = '#f59e0b';
    borderWidth = 3;
  }

  if (isFaulty) {
    borderColor = '#ef4444';
    borderWidth = 3;
    opacity = 1;
  }

  if (step.quality_score !== undefined && step.quality_score < 0.5 && !hasError && !hasWarning) {
    opacity = 0.6 + step.quality_score * 0.4;
  }

  const isSlow = (step.duration_ms ?? 0) > SLOW_STEP_MS;

  return {
    background: '#ffffff',
    border: `${borderWidth}px solid ${borderColor}`,
    borderRadius: '8px',
    padding: '10px 14px',
    width: 200,
    opacity,
    boxShadow: hasError || isFaulty
      ? '0 0 12px rgba(239,68,68,0.4)'
      : hasWarning
        ? '0 0 8px rgba(245,158,11,0.3)'
        : '0 1px 3px rgba(0,0,0,0.1)',
    animation: isSlow ? 'slowPulse 1.5s infinite' : 'none',
  };
}

function getChunkNodeStyle(importanceScore?: number): React.CSSProperties {
  const score = importanceScore ?? 0.5;
  const opacity = 0.35 + score * 0.65;
  const borderWidth = 1 + score * 2;
  const width = 160 + score * 20;

  let borderColor = '#3b82f6';
  if (score >= 0.7) borderColor = '#f97316';
  else if (score >= 0.4) borderColor = '#3b82f6';
  else borderColor = '#93c5fd';

  return {
    background: '#eff6ff',
    border: `${borderWidth.toFixed(1)}px solid ${borderColor}`,
    borderRadius: '6px',
    padding: '6px 8px',
    width,
    fontSize: 11,
    opacity,
    transition: 'all 0.3s ease',
  };
}

function getChunkToAnswerEdgeStyle(importanceScore?: number): { stroke: string; strokeWidth: number } {
  const score = importanceScore ?? 0.5;
  const strokeWidth = 1 + score * 3;
  let stroke = '#9ca3af';
  if (score >= 0.7) stroke = '#f97316';
  else if (score >= 0.4) stroke = '#3b82f6';
  return { stroke, strokeWidth };
}

function buildFlowData(
  steps: Step[],
  rootCause: RootCause | null,
  perturbationResults: PerturbationResult[] | null,
  maxVisibleSteps?: number,
  highlightedChunkIndices?: Set<number>,
): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  const visibleSteps = maxVisibleSteps !== undefined
    ? steps.slice(0, maxVisibleSteps)
    : steps;

  const faultyStepIds = new Set(rootCause?.involved_step_ids || []);

  const answerStep = visibleSteps.find(s => s.step_type === 'answer_generate');
  const answerStepId = answerStep ? `step-${answerStep.id}` : null;

  const importanceMap = new Map<number, number>();
  if (perturbationResults) {
    perturbationResults.forEach(r => {
      importanceMap.set(r.chunk_index, r.importance_score);
    });
  }
  steps.forEach(step => {
    step.chunks?.forEach(chunk => {
      if (chunk.importance_score !== undefined && chunk.importance_score !== null) {
        if (!importanceMap.has(chunk.chunk_index ?? -1)) {
          importanceMap.set(chunk.chunk_index ?? -1, chunk.importance_score);
        }
      }
    });
  });

  visibleSteps.forEach((step, index) => {
    const config = stepTypeConfig[step.step_type] || { label: step.step_type, color: '#6b7280', icon: '📄' };
    const alerts = step.alerts || [];
    const hasAlert = alerts.length > 0;
    const isFaulty = faultyStepIds.has(step.id);

    nodes.push({
      id: `step-${step.id}`,
      type: 'default',
      position: { x: 100, y: 80 + index * 140 },
      data: {
        label: (
          <div className="text-center">
            <div className="text-lg mb-1">{config.icon}</div>
            <div className="font-semibold text-gray-800 text-xs">{config.label}</div>
            {step.duration_ms !== undefined && step.duration_ms !== null && (
              <div className={`text-[10px] mt-0.5 ${step.duration_ms > SLOW_STEP_MS ? 'text-red-500 font-bold' : 'text-gray-400'}`}>
                {step.duration_ms > 1000 ? `${(step.duration_ms / 1000).toFixed(1)}s` : `${step.duration_ms}ms`}
              </div>
            )}
            {step.quality_score !== undefined && (
              <div className="text-[10px] mt-0.5 text-gray-500">
                Q: {(step.quality_score * 100).toFixed(0)}%
              </div>
            )}
            {hasAlert && (
              <div className="text-[10px] mt-0.5">
                {alerts.some(a => a.severity === 'error') ? (
                  <span className="text-red-500">ERROR</span>
                ) : (
                  <span className="text-yellow-600">WARN</span>
                )}
              </div>
            )}
            {isFaulty && !hasAlert && (
              <div className="text-[10px] mt-0.5 text-red-500">ROOT CAUSE</div>
            )}
          </div>
        ),
        step,
      },
      style: getNodeStyle(step, isFaulty),
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
    });

    if (index > 0) {
      const prevStep = visibleSteps[index - 1];
      const isFaultyEdge = faultyStepIds.has(prevStep.id) || faultyStepIds.has(step.id);
      edges.push({
        id: `e-${prevStep.id}-${step.id}`,
        source: `step-${prevStep.id}`,
        target: `step-${step.id}`,
        animated: step.step_type === 'answer_generate',
        style: {
          stroke: isFaultyEdge ? '#ef4444' : config.color,
          strokeWidth: isFaultyEdge ? 3 : 2,
        },
      });
    }

    if (step.step_type === 'vector_retrieve' && (step.chunks || []).length > 0) {
      (step.chunks || []).forEach((chunk, cIdx) => {
        const chunkId = chunk.id ? `chunk-${chunk.id}` : `chunk-${step.id}-${chunk.chunk_index ?? cIdx}`;
        const impScore = importanceMap.get(chunk.chunk_index ?? cIdx);

        const isHighlighted = highlightedChunkIndices?.has(chunk.chunk_index ?? cIdx) ?? false;

        nodes.push({
          id: chunkId,
          type: 'default',
          position: { x: 380, y: 80 + index * 140 + cIdx * 90 },
          data: {
            label: (
              <div className="text-[10px]">
                <div className="font-medium text-gray-700 truncate">Chunk {chunk.chunk_index ?? cIdx + 1}</div>
                <div className="text-gray-500 truncate" style={{ maxWidth: 140 }}>
                  {chunk.content.slice(0, 40)}...
                </div>
                {chunk.relevance_score !== undefined && (
                  <div className="text-blue-500 mt-0.5">
                    Rel: {(chunk.relevance_score * 100).toFixed(0)}%
                  </div>
                )}
                {impScore !== undefined && (
                  <div className="mt-0.5" style={{ color: impScore >= 0.7 ? '#f97316' : impScore >= 0.4 ? '#3b82f6' : '#93c5fd' }}>
                    Imp: {(impScore * 100).toFixed(0)}%
                  </div>
                )}
              </div>
            ),
            chunk,
          },
          style: {
            ...getChunkNodeStyle(impScore),
            ...(isHighlighted ? {
              boxShadow: '0 0 14px rgba(99,102,241,0.6)',
              border: '2.5px solid #6366f1',
              transform: 'scale(1.05)',
            } : {}),
          },
          sourcePosition: Position.Right,
          targetPosition: Position.Left,
        });

        edges.push({
          id: `e-step-${step.id}-${chunkId}`,
          source: `step-${step.id}`,
          target: chunkId,
          style: { stroke: '#93c5fd', strokeWidth: 1.5 },
        });

        if (answerStepId) {
          const edgeStyle = getChunkToAnswerEdgeStyle(impScore);
          edges.push({
            id: `e-${chunkId}-${answerStepId}`,
            source: chunkId,
            target: answerStepId,
            animated: impScore !== undefined && impScore > 0.5,
            style: {
              stroke: edgeStyle.stroke,
              strokeWidth: edgeStyle.strokeWidth,
              opacity: 0.7,
            },
          });
        }
      });
    }
  });

  return { nodes, edges };
}

export default function FlowChart({ session, rootCause, perturbationResults, onNodeClick }: Props) {
  const steps = session?.steps || [];
  const totalSteps = steps.length;
  const [playbackIndex, setPlaybackIndex] = useState(totalSteps);
  const [isPlaying, setIsPlaying] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const { highlightedChunkIndices, highlightAnswerSegment, clearHighlights } = useHighlightContext();

  // When session changes, reset to full view
  useEffect(() => {
    setPlaybackIndex(totalSteps);
    setIsPlaying(false);
  }, [session?.id, totalSteps]);

  // Auto-play timer
  useEffect(() => {
    if (!isPlaying) {
      if (timerRef.current) clearInterval(timerRef.current);
      return;
    }
    timerRef.current = setInterval(() => {
      setPlaybackIndex(prev => {
        if (prev >= totalSteps) {
          setIsPlaying(false);
          return totalSteps;
        }
        return prev + 1;
      });
    }, PLAYBACK_INTERVAL_MS);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isPlaying, totalSteps]);

  const { nodes, edges } = useMemo(
    () => buildFlowData(steps, rootCause, perturbationResults, playbackIndex, highlightedChunkIndices),
    [steps, rootCause, perturbationResults, playbackIndex, highlightedChunkIndices]
  );

  const handleNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    if (node.data?.step) {
      onNodeClick(node.data.step);
    }
  }, [onNodeClick]);

  const handleNodeMouseEnter = useCallback((_: React.MouseEvent, node: Node) => {
    if (node.data?.chunk) {
      const chunk = node.data.chunk;
      const cIdx = chunk.chunk_index;
      if (cIdx !== undefined) {
        // Find the answer segment that references this chunk
        const segments = session?.execution_trace?.answer_segments;
        if (segments) {
          for (let i = 0; i < segments.length; i++) {
            if (segments[i].supported_by?.includes(cIdx)) {
              highlightAnswerSegment(i);
              return;
            }
          }
        }
      }
    }
  }, [session, highlightAnswerSegment]);

  const handleNodeMouseLeave = useCallback(() => {
    clearHighlights();
  }, [clearHighlights]);

  const handlePlay = () => {
    if (playbackIndex >= totalSteps) setPlaybackIndex(0);
    setIsPlaying(true);
  };

  const handlePause = () => setIsPlaying(false);
  const handleReset = () => { setIsPlaying(false); setPlaybackIndex(totalSteps); };
  const handleStepForward = () => setPlaybackIndex(p => Math.min(p + 1, totalSteps));
  const handleStepBack = () => setPlaybackIndex(p => Math.max(p - 1, 0));

  if (!session) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-50">
        <div className="text-center text-gray-400">
          <div className="text-4xl mb-2">🔍</div>
          <p>Enter a query to start visualization</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full w-full bg-gray-50 relative">
      {/* Info overlay */}
      <div className="absolute top-4 left-4 z-10 bg-white/90 backdrop-blur px-3 py-2 rounded-lg shadow-sm border border-gray-200">
        <h3 className="text-sm font-semibold text-gray-700">Retrieval Pipeline</h3>
        <p className="text-xs text-gray-500 mt-0.5">Session #{session.id}</p>
      </div>

      {/* Playback controls */}
      <div className="absolute top-4 right-4 z-10 bg-white/90 backdrop-blur px-3 py-2 rounded-lg shadow-sm border border-gray-200">
        <div className="flex items-center gap-1.5">
          <button
            onClick={handleReset}
            className="text-xs px-1.5 py-0.5 rounded bg-gray-100 hover:bg-gray-200 text-gray-600"
            title="Show all"
          >
            ⏮
          </button>
          <button
            onClick={handleStepBack}
            disabled={playbackIndex === 0}
            className="text-xs px-1.5 py-0.5 rounded bg-gray-100 hover:bg-gray-200 text-gray-600 disabled:opacity-30"
          >
            ◀
          </button>
          {isPlaying ? (
            <button
              onClick={handlePause}
              className="text-xs px-2 py-0.5 rounded bg-orange-100 hover:bg-orange-200 text-orange-700 font-medium"
            >
              ⏸ Pause
            </button>
          ) : (
            <button
              onClick={handlePlay}
              disabled={totalSteps === 0}
              className="text-xs px-2 py-0.5 rounded bg-indigo-100 hover:bg-indigo-200 text-indigo-700 font-medium disabled:opacity-30"
            >
              ▶ Play
            </button>
          )}
          <button
            onClick={handleStepForward}
            disabled={playbackIndex >= totalSteps}
            className="text-xs px-1.5 py-0.5 rounded bg-gray-100 hover:bg-gray-200 text-gray-600 disabled:opacity-30"
          >
            ▶
          </button>
          <div className="text-[10px] text-gray-500 ml-1 min-w-[36px] text-center">
            {playbackIndex}/{totalSteps}
          </div>
        </div>
        {/* Progress bar */}
        <div className="w-full bg-gray-200 rounded-full h-1.5 mt-1.5">
          <div
            className="bg-indigo-500 h-1.5 rounded-full transition-all duration-300"
            style={{ width: `${totalSteps > 0 ? (playbackIndex / totalSteps) * 100 : 0}%` }}
          />
        </div>
        <div className="flex items-center gap-2 mt-1 text-[10px] text-gray-400">
          <span>⏱ Slow step &gt;2s</span>
          <span className="inline-block w-2 h-2 rounded-full bg-red-400 animate-pulse" />
        </div>
      </div>

      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodeClick={handleNodeClick}
        onNodeMouseEnter={handleNodeMouseEnter}
        onNodeMouseLeave={handleNodeMouseLeave}
        fitView
        attributionPosition="bottom-right"
      >
        <Background gap={16} size={1} color="#e5e7eb" />
        <Controls />
        <MiniMap nodeStrokeWidth={3} zoomable pannable />
      </ReactFlow>
    </div>
  );
}
