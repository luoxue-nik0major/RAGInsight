from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class StepType(str, Enum):
    QUERY_PARSE = "query_parse"
    VECTOR_RETRIEVE = "vector_retrieve"
    RERANK = "rerank"
    CONTEXT_BUILD = "context_build"
    ANSWER_GENERATE = "answer_generate"


class SessionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AlertSeverity(str, Enum):
    WARNING = "warning"
    ERROR = "error"


class ChunkBase(BaseModel):
    content: str
    source: Optional[str] = None
    relevance_score: Optional[float] = None
    importance_score: Optional[float] = None
    chunk_index: Optional[int] = None


class ChunkOut(ChunkBase):
    id: int
    step_id: int

    class Config:
        from_attributes = True


class AlertBase(BaseModel):
    alert_type: str
    severity: AlertSeverity
    message: str
    suggestion: Optional[str] = None


class AlertOut(AlertBase):
    id: int
    session_id: int
    step_id: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True


class StepBase(BaseModel):
    step_type: StepType
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    quality_score: Optional[float] = None
    duration_ms: Optional[int] = None


class StepOut(StepBase):
    id: int
    session_id: int
    timestamp: datetime
    chunks: List[ChunkOut] = Field(default_factory=list)
    alerts: List[AlertOut] = Field(default_factory=list)

    class Config:
        from_attributes = True


class SessionBase(BaseModel):
    query: str


class SessionCreate(SessionBase):
    pass


class SessionOut(SessionBase):
    id: int
    final_answer: Optional[str] = None
    complexity_score: Optional[float] = None
    recommended_strategy: Optional[str] = None
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    execution_trace: Dict[str, Any] = Field(default_factory=dict)
    steps: List[StepOut] = Field(default_factory=list)
    alerts: List[AlertOut] = Field(default_factory=list)

    class Config:
        from_attributes = True


class SessionListItem(BaseModel):
    id: int
    query: str
    status: SessionStatus
    created_at: datetime

    class Config:
        from_attributes = True


class SSEEvent(BaseModel):
    event: str
    data: Dict[str, Any]


class StrategyInfo(BaseModel):
    id: str
    name: str
    description: str
    icon: str


class PerturbationResult(BaseModel):
    chunk_index: int
    chunk_id: Optional[int] = None
    content: str
    importance_score: float
    is_approximate: bool
    perturbed_answer: Optional[str] = None


class WhatIfRequest(BaseModel):
    remove_chunk_indices: List[int]


class WhatIfResponse(BaseModel):
    original_answer: str
    new_answer: str
    similarity: float
    removed_count: int
    kept_count: int


class ComplexityFeatures(BaseModel):
    length: int
    length_score: float
    sentence_count: int
    sentence_score: float
    entity_count: int
    entity_score: float
    relation_count: int
    relation_score: float
    semantic_score: float
    hop_demand_score: float


class ComplexityAnalysis(BaseModel):
    query: str
    complexity_score: float
    question_type: str
    features: ComplexityFeatures
    recommended_strategy: str
    recommended_strategy_name: str
    reason: str
    alternatives: List[Dict[str, str]] = Field(default_factory=list)


# ── Causal Attribution Models ────────────────────────────────────────────

class InterventionResultSchema(BaseModel):
    component: str
    intervention: str
    params: Dict[str, Any] = Field(default_factory=dict)
    original_quality: float
    perturbed_quality: float
    quality_delta: float
    attribution_score: float = 0.0
    is_approximate: bool = True
    perturbed_answer: Optional[str] = None
    description: str = ""


class CausalGraphNode(BaseModel):
    id: str
    label: str
    type: str
    attribution: float = 0.0


class CausalGraphEdge(BaseModel):
    source: str = Field(alias="from_")
    target: str = Field(alias="to")
    label: str = ""

    class Config:
        populate_by_name = True


class CausalGraph(BaseModel):
    nodes: List[CausalGraphNode] = Field(default_factory=list)
    edges: List[Dict[str, str]] = Field(default_factory=list)
    observables: Dict[str, Any] = Field(default_factory=dict)


class AttributionReportSchema(BaseModel):
    session_id: int
    query: str
    original_strategy: str
    original_quality: float
    interventions: List[InterventionResultSchema] = Field(default_factory=list)
    component_attributions: Dict[str, float] = Field(default_factory=dict)
    top_contributors: List[Dict[str, Any]] = Field(default_factory=list)
    causal_graph: CausalGraph = Field(default_factory=lambda: CausalGraph())
    total_interventions: int = 0
    llm_interventions: int = 0
    duration_ms: int = 0


class RouterPrediction(BaseModel):
    recommended_strategy: str
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    router_mode: str = "heuristic"
