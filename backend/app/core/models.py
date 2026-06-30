import enum
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Enum, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class StepType(str, enum.Enum):
    QUERY_PARSE = "query_parse"
    VECTOR_RETRIEVE = "vector_retrieve"
    RERANK = "rerank"
    CONTEXT_BUILD = "context_build"
    ANSWER_GENERATE = "answer_generate"


class SessionStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    final_answer = Column(Text, nullable=True)
    complexity_score = Column(Float, nullable=True)
    recommended_strategy = Column(String, nullable=True)
    status = Column(Enum(SessionStatus), default=SessionStatus.PENDING)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    execution_trace = Column(JSON, default=dict)

    steps = relationship("Step", back_populates="session", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="session", cascade="all, delete-orphan")


class Step(Base):
    __tablename__ = "steps"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    step_type = Column(Enum(StepType), nullable=False)
    input_data = Column(JSON, default=dict)
    output_data = Column(JSON, default=dict)
    quality_score = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    duration_ms = Column(Integer, nullable=True)

    session = relationship("Session", back_populates="steps")
    chunks = relationship("Chunk", back_populates="step", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="step", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    step_id = Column(Integer, ForeignKey("steps.id"), nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String, nullable=True)
    relevance_score = Column(Float, nullable=True)
    importance_score = Column(Float, nullable=True)
    chunk_index = Column(Integer, nullable=True)

    step = relationship("Step", back_populates="chunks")


class AlertSeverity(str, enum.Enum):
    WARNING = "warning"
    ERROR = "error"


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    step_id = Column(Integer, ForeignKey("steps.id"), nullable=True)
    alert_type = Column(String, nullable=False)
    severity = Column(Enum(AlertSeverity), nullable=False)
    message = Column(Text, nullable=False)
    suggestion = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="alerts")
    step = relationship("Step", back_populates="alerts")


class AttributionResult(Base):
    """Stores causal attribution analysis results for a session."""
    __tablename__ = "attribution_results"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    component_name = Column(String, nullable=False)
    intervention_type = Column(String, nullable=False)
    intervention_params = Column(JSON, default=dict)
    original_quality = Column(Float, nullable=False, default=0.0)
    perturbed_quality = Column(Float, nullable=False, default=0.0)
    quality_delta = Column(Float, nullable=False, default=0.0)
    attribution_score = Column(Float, nullable=False, default=0.0)
    is_approximate = Column(Integer, default=1)  # 1=True, 0=False
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session")
