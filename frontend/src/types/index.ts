export type StepType = 
  | 'query_parse' 
  | 'vector_retrieve' 
  | 'rerank' 
  | 'context_build' 
  | 'answer_generate';

export type SessionStatus = 'pending' | 'running' | 'completed' | 'failed';

export type AlertSeverity = 'warning' | 'error';

export interface Chunk {
  id: number;
  step_id: number;
  content: string;
  source?: string;
  relevance_score?: number;
  importance_score?: number;
  chunk_index?: number;
}

export interface Alert {
  id: number;
  session_id: number;
  step_id?: number;
  alert_type: string;
  severity: AlertSeverity;
  message: string;
  suggestion?: string;
  created_at: string;
}

export interface Step {
  id: number;
  session_id: number;
  step_type: StepType;
  input_data: Record<string, any>;
  output_data: Record<string, any>;
  quality_score?: number;
  duration_ms?: number;
  timestamp: string;
  chunks: Chunk[];
  alerts: Alert[];
}

export interface RootCause {
  root_cause_type: string;
  root_cause_label: string;
  severity: string;
  involved_step_ids: number[];
  explanation: string;
  suggestions: string[];
  alert_count: number;
  error_count: number;
  warning_count: number;
}

export interface Session {
  id: number;
  query: string;
  final_answer?: string;
  complexity_score?: number;
  recommended_strategy?: string;
  status: SessionStatus;
  created_at: string;
  updated_at: string;
  execution_trace: Record<string, any>;
  steps: Step[];
  alerts: Alert[];
}

export interface SessionListItem {
  id: number;
  query: string;
  status: SessionStatus;
  created_at: string;
}

export interface StrategyInfo {
  id: string;
  name: string;
  description: string;
  icon: string;
}

export interface CollectionInfo {
  name: string;
  count: number;
}

export interface SSEEvent {
  event: string;
  data: Record<string, any>;
}

export interface PerturbationResult {
  chunk_index: number;
  chunk_id?: number;
  content: string;
  importance_score: number;
  is_approximate: boolean;
  perturbed_answer?: string;
}

export interface PerturbationTask {
  task_id: string;
  session_id: number;
  status: string;
  progress: number;
  total: number;
  result?: PerturbationResult[];
  error?: string;
}

export interface WhatIfResponse {
  original_answer: string;
  new_answer: string;
  similarity: number;
  removed_count: number;
  kept_count: number;
}

export interface ComplexityFeatures {
  length: number;
  length_score: number;
  sentence_count: number;
  sentence_score: number;
  entity_count: number;
  entity_score: number;
  relation_count: number;
  relation_score: number;
  semantic_score: number;
  hop_demand_score: number;
}

export interface ComplexityAnalysis {
  query: string;
  complexity_score: number;
  question_type: string;
  features: ComplexityFeatures;
  recommended_strategy: string;
  recommended_strategy_name: string;
  reason: string;
  alternatives: { id: string; name: string; description: string }[];
}

export interface TestQuery {
  id: number;
  query: string;
  category: string;
  expected_answer_type: string;
  expected_depth: number;
  recommended_strategy: string;
  expected_alerts: string[];
}

export interface ExperimentResult {
  query_id: number;
  query: string;
  strategy: string;
  category: string;
  expected_alerts: string[];
  session_id?: number;
  status: string;
  complexity_score?: number;
  recommended_strategy?: string;
  final_answer?: string;
  actual_alerts: string[];
  steps: any[];
  execution_trace?: Record<string, any>;
  error?: string;
}

export interface FaultMetrics {
  precision: number;
  recall: number;
  f1: number;
  tp: number;
  fp: number;
  fn: number;
}

export interface AnswerEvaluation {
  faithfulness: {
    score: number;
    claims: { text: string; supported: boolean; best_chunk_index: number; best_similarity: number }[];
    total_claims: number;
    supported_claims: number;
  };
  relevance: {
    score: number;
    method: string;
  };
  combined_score: number;
}

// ── Causal Attribution Types ────────────────────────────────────────────

export interface CausalGraphNode {
  id: string;
  label: string;
  type: string;
  attribution: number;
}

export interface CausalGraphEdge {
  from: string;
  to: string;
  label: string;
}

export interface CausalGraph {
  nodes: CausalGraphNode[];
  edges: CausalGraphEdge[];
  observables: Record<string, any>;
}

export interface AttributionIntervention {
  component: string;
  intervention: string;
  params: Record<string, any>;
  original_quality: number;
  perturbed_quality: number;
  quality_delta: number;
  attribution_score: number;
  is_approximate: boolean;
  description: string;
}

export interface AttributionReport {
  session_id: number;
  query: string;
  original_strategy: string;
  original_quality: number;
  interventions: AttributionIntervention[];
  component_attributions: Record<string, number>;
  top_contributors: Record<string, any>[];
  causal_graph: CausalGraph;
  total_interventions: number;
  llm_interventions: number;
  duration_ms: number;
}

export interface AttributionTask {
  task_id: string;
  session_id: number;
  status: string;
  progress: number;
  total: number;
  result?: AttributionReport;
  error?: string;
  extra_info?: Record<string, any>;
}

export interface ExperimentMetrics {
  experiment_1_fault_detection: {
    per_alert_type: Record<string, FaultMetrics>;
    macro: { precision: number; recall: number; f1: number };
    micro: { precision: number; recall: number; f1: number; tp: number; fp: number; fn: number };
  } | null;
  experiment_2_strategy_comparison: {
    strategies: Record<string, {
      count: number;
      avg_relevance: number;
      avg_coverage: number;
      avg_diversity: number;
      avg_answer_quality: number;
      avg_faithfulness: number;
      avg_answer_relevance: number;
      avg_duration_ms: number;
    }>;
  } | null;
  total_runs: number;
  message?: string;
}
