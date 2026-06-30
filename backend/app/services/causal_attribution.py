"""
Causal Attribution Framework for RAG Failure Diagnosis.

Extends perturbation analysis from chunk-level to component-level counterfactual
interventions. Builds a causal DAG over the pipeline and computes normalized
attribution scores for each component.

Two-stage cost optimization:
  1. Embedding approximation for all interventions (cheap, local)
  2. LLM exact verification for top-k highest-attribution components (expensive)
"""
import asyncio
import time
import numpy as np
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from app.services.deepseek import deepseek_client
from app.services.retriever import retriever_registry, _get_embedding_fn
from app.services.answer_evaluator import answer_evaluator


class ComponentType(str, Enum):
    STRATEGY = "strategy"
    TOP_K = "top_k"
    CHUNK_SELECTION = "chunk_selection"
    CONTEXT_ASSEMBLY = "context_assembly"
    LLM_GENERATION = "llm_generation"


class InterventionType(str, Enum):
    SWITCH_STRATEGY = "switch_strategy"
    CHANGE_TOPK = "change_topk"
    REMOVE_CHUNKS = "remove_chunks"
    COMPRESS_CONTEXT = "compress_context"


@dataclass
class InterventionResult:
    component: ComponentType
    intervention: InterventionType
    params: Dict[str, Any]
    original_quality: float
    perturbed_quality: float
    quality_delta: float              # positive = quality improved, negative = degraded
    attribution_score: float = 0.0    # normalized contribution
    is_approximate: bool = True
    perturbed_answer: Optional[str] = None
    description: str = ""


@dataclass
class AttributionReport:
    session_id: int
    query: str
    original_strategy: str
    original_quality: float
    interventions: List[InterventionResult] = field(default_factory=list)
    component_attributions: Dict[str, float] = field(default_factory=dict)
    top_contributors: List[Dict[str, Any]] = field(default_factory=list)
    causal_graph: Dict[str, Any] = field(default_factory=dict)
    total_interventions: int = 0
    llm_interventions: int = 0
    duration_ms: int = 0


class CausalAttributionAnalyzer:
    """Systematic counterfactual attribution for RAG pipeline components."""

    def __init__(self, max_concurrent_llm: int = 2, exact_top_k: int = 3):
        self.semaphore = asyncio.Semaphore(max_concurrent_llm)
        self.exact_top_k = exact_top_k
        self._embedding_fn = None

    def _get_emb_fn(self):
        if self._embedding_fn is None:
            self._embedding_fn = _get_embedding_fn()
        return self._embedding_fn

    def _embed(self, texts: List[str]) -> np.ndarray:
        emb_fn = self._get_emb_fn()
        embeddings = emb_fn(texts)
        return np.array(embeddings, dtype=np.float32)

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        a_norm = a / (np.linalg.norm(a) + 1e-10)
        b_norm = b / (np.linalg.norm(b) + 1e-10)
        return float(np.dot(a_norm, b_norm))

    def _compute_answer_quality(
        self, query: str, answer: str, context_chunks: List[Dict[str, Any]]
    ) -> float:
        """Compute combined answer quality score using local evaluator."""
        if not answer or not context_chunks:
            return 0.0
        try:
            eval_result = answer_evaluator.evaluate(query, answer, context_chunks)
            return eval_result["combined_score"]
        except Exception:
            return 0.0

    # ── Intervention Methods ──────────────────────────────────────────────

    async def intervene_strategy(
        self,
        query: str,
        original_answer: str,
        original_chunks: List[Dict[str, Any]],
        original_strategy: str,
        new_strategy: str,
        original_quality: float,
    ) -> InterventionResult:
        """Intervene by switching retrieval strategy."""
        try:
            retriever = retriever_registry.get(new_strategy)
            result = await retriever.retrieve(query, top_k=5)
            new_chunks = result["chunks"]

            if not new_chunks:
                return InterventionResult(
                    component=ComponentType.STRATEGY,
                    intervention=InterventionType.SWITCH_STRATEGY,
                    params={"from": original_strategy, "to": new_strategy},
                    original_quality=original_quality,
                    perturbed_quality=0.0,
                    quality_delta=-original_quality,
                    is_approximate=True,
                    description=f"Switch strategy {original_strategy}→{new_strategy}: no chunks returned",
                )

            new_answer = await deepseek_client.generate_answer(query, new_chunks)
            new_quality = self._compute_answer_quality(query, new_answer, new_chunks)
            delta = new_quality - original_quality

            return InterventionResult(
                component=ComponentType.STRATEGY,
                intervention=InterventionType.SWITCH_STRATEGY,
                params={"from": original_strategy, "to": new_strategy},
                original_quality=original_quality,
                perturbed_quality=new_quality,
                quality_delta=delta,
                is_approximate=False,
                perturbed_answer=new_answer,
                description=f"Switch strategy {original_strategy}→{new_strategy}: ΔQ={delta:+.4f}",
            )
        except Exception as e:
            return InterventionResult(
                component=ComponentType.STRATEGY,
                intervention=InterventionType.SWITCH_STRATEGY,
                params={"from": original_strategy, "to": new_strategy, "error": str(e)},
                original_quality=original_quality,
                perturbed_quality=original_quality,
                quality_delta=0.0,
                is_approximate=True,
                description=f"Switch strategy failed: {e}",
            )

    async def intervene_topk(
        self,
        query: str,
        original_answer: str,
        original_chunks: List[Dict[str, Any]],
        original_strategy: str,
        original_k: int,
        new_k: int,
        original_quality: float,
    ) -> InterventionResult:
        """Intervene by changing retrieval count."""
        try:
            retriever = retriever_registry.get(original_strategy)
            result = await retriever.retrieve(query, top_k=new_k)
            new_chunks = result["chunks"][:new_k]

            if not new_chunks:
                return InterventionResult(
                    component=ComponentType.TOP_K,
                    intervention=InterventionType.CHANGE_TOPK,
                    params={"from_k": original_k, "to_k": new_k},
                    original_quality=original_quality,
                    perturbed_quality=0.0,
                    quality_delta=-original_quality,
                    is_approximate=True,
                    description=f"Change top_k {original_k}→{new_k}: no chunks",
                )

            new_answer = await deepseek_client.generate_answer(query, new_chunks)
            new_quality = self._compute_answer_quality(query, new_answer, new_chunks)
            delta = new_quality - original_quality

            return InterventionResult(
                component=ComponentType.TOP_K,
                intervention=InterventionType.CHANGE_TOPK,
                params={"from_k": original_k, "to_k": new_k, "strategy": original_strategy},
                original_quality=original_quality,
                perturbed_quality=new_quality,
                quality_delta=delta,
                is_approximate=False,
                perturbed_answer=new_answer,
                description=f"Change top_k {original_k}→{new_k}: ΔQ={delta:+.4f}",
            )
        except Exception as e:
            return InterventionResult(
                component=ComponentType.TOP_K,
                intervention=InterventionType.CHANGE_TOPK,
                params={"from_k": original_k, "to_k": new_k, "error": str(e)},
                original_quality=original_quality,
                perturbed_quality=original_quality,
                quality_delta=0.0,
                is_approximate=True,
                description=f"Change top_k failed: {e}",
            )

    async def intervene_chunks(
        self,
        query: str,
        original_answer: str,
        chunks: List[Dict[str, Any]],
        remove_indices: Set[int],
        original_quality: float,
    ) -> InterventionResult:
        """Intervene by removing specific chunks (leave-N-out)."""
        kept = [c for i, c in enumerate(chunks) if i not in remove_indices]
        if not kept:
            return InterventionResult(
                component=ComponentType.CHUNK_SELECTION,
                intervention=InterventionType.REMOVE_CHUNKS,
                params={"removed_indices": sorted(remove_indices), "kept_count": 0},
                original_quality=original_quality,
                perturbed_quality=0.0,
                quality_delta=-original_quality,
                is_approximate=True,
                description=f"Remove chunks {sorted(remove_indices)}: no chunks remain",
            )

        new_answer = await deepseek_client.generate_answer(query, kept)
        new_quality = self._compute_answer_quality(query, new_answer, kept)
        delta = new_quality - original_quality

        return InterventionResult(
            component=ComponentType.CHUNK_SELECTION,
            intervention=InterventionType.REMOVE_CHUNKS,
            params={"removed_indices": sorted(remove_indices), "kept_count": len(kept)},
            original_quality=original_quality,
            perturbed_quality=new_quality,
            quality_delta=delta,
            is_approximate=False,
            perturbed_answer=new_answer,
            description=f"Remove chunks {sorted(remove_indices)}: ΔQ={delta:+.4f}",
        )

    async def intervene_context(
        self,
        query: str,
        original_answer: str,
        chunks: List[Dict[str, Any]],
        original_quality: float,
        max_chars: int = 1500,
    ) -> InterventionResult:
        """Intervene by truncating/compressing the context window."""
        truncated = []
        total_chars = 0
        for c in chunks:
            if total_chars + len(c["content"]) <= max_chars:
                truncated.append(c)
                total_chars += len(c["content"])
            else:
                remaining = max_chars - total_chars
                if remaining > 50:
                    truncated.append({**c, "content": c["content"][:remaining] + "..."})
                break

        if not truncated:
            return InterventionResult(
                component=ComponentType.CONTEXT_ASSEMBLY,
                intervention=InterventionType.COMPRESS_CONTEXT,
                params={"max_chars": max_chars, "original_chunks": len(chunks), "kept_chunks": 0},
                original_quality=original_quality,
                perturbed_quality=0.0,
                quality_delta=-original_quality,
                is_approximate=True,
                description=f"Compress context to {max_chars} chars: all chunks removed",
            )

        new_answer = await deepseek_client.generate_answer(query, truncated)
        new_quality = self._compute_answer_quality(query, new_answer, truncated)
        delta = new_quality - original_quality

        return InterventionResult(
            component=ComponentType.CONTEXT_ASSEMBLY,
            intervention=InterventionType.COMPRESS_CONTEXT,
            params={"max_chars": max_chars, "original_chunks": len(chunks), "kept_chunks": len(truncated)},
            original_quality=original_quality,
            perturbed_quality=new_quality,
            quality_delta=delta,
            is_approximate=False,
            perturbed_answer=new_answer,
            description=f"Compress context to {max_chars} chars ({len(truncated)}/{len(chunks)} chunks): ΔQ={delta:+.4f}",
        )

    # ── Approximate Attribution (embedding-based, no LLM) ──────────────────

    def _approx_intervene_strategy(
        self,
        query: str,
        original_chunks: List[Dict[str, Any]],
        new_chunks: List[Dict[str, Any]],
        original_quality: float,
    ) -> float:
        """Approximate quality delta for strategy switch using chunk embedding similarity."""
        if not new_chunks:
            return -original_quality

        orig_texts = [c["content"] for c in original_chunks]
        new_texts = [c["content"] for c in new_chunks]

        if not orig_texts:
            return 0.0

        try:
            orig_embs = self._embed(orig_texts)
            new_embs = self._embed(new_texts)

            # Average pairwise similarity between original and new chunk sets
            sims = []
            for oe in orig_embs:
                best = max(self._cosine_sim(oe, ne) for ne in new_embs)
                sims.append(best)
            avg_sim = float(np.mean(sims))

            approx_new_quality = original_quality * avg_sim
            return approx_new_quality - original_quality
        except Exception:
            return 0.0

    def _approx_intervene_chunks(
        self,
        original_answer: str,
        chunks: List[Dict[str, Any]],
        remove_indices: Set[int],
        original_quality: float,
    ) -> float:
        """Approximate quality delta for chunk removal using chunk-answer similarity."""
        kept_indices = [i for i in range(len(chunks)) if i not in remove_indices]
        if not kept_indices:
            return -original_quality

        try:
            answer_emb = self._embed([original_answer])[0]
            chunk_texts = [chunks[i]["content"] for i in kept_indices]
            chunk_embs = self._embed(chunk_texts)

            # Average similarity of kept chunks to answer
            sims = [self._cosine_sim(ce, answer_emb) for ce in chunk_embs]
            avg_sim = float(np.mean(sims)) if sims else 0.0

            # Also compute original average similarity for comparison
            all_chunk_texts = [c["content"] for c in chunks]
            all_chunk_embs = self._embed(all_chunk_texts)
            all_sims = [self._cosine_sim(ce, answer_emb) for ce in all_chunk_embs]
            orig_avg_sim = float(np.mean(all_sims)) if all_sims else 0.0

            if orig_avg_sim < 1e-8:
                return 0.0

            approx_new_quality = original_quality * (avg_sim / orig_avg_sim)
            return approx_new_quality - original_quality
        except Exception:
            return 0.0

    # ── Causal Graph Construction ──────────────────────────────────────────

    def build_causal_graph(self, session_trace: Dict[str, Any]) -> Dict[str, Any]:
        """Build the causal DAG from execution trace data."""
        nodes = [
            {"id": "query", "label": "Query", "type": "input"},
            {"id": "strategy", "label": "Retrieval Strategy", "type": "decision"},
            {"id": "retrieval", "label": "Document Retrieval", "type": "process"},
            {"id": "context", "label": "Context Assembly", "type": "process"},
            {"id": "llm", "label": "LLM Generation", "type": "process"},
            {"id": "quality", "label": "Answer Quality", "type": "outcome"},
        ]

        edges = [
            {"from": "query", "to": "strategy", "label": "informs"},
            {"from": "query", "to": "retrieval", "label": "search terms"},
            {"from": "strategy", "to": "retrieval", "label": "configures"},
            {"from": "retrieval", "to": "context", "label": "provides chunks"},
            {"from": "context", "to": "llm", "label": "feeds context"},
            {"from": "query", "to": "llm", "label": "direct question"},
            {"from": "llm", "to": "quality", "label": "determines"},
            {"from": "context", "to": "quality", "label": "influences"},
            {"from": "retrieval", "to": "quality", "label": "influences"},
        ]

        observables = {}
        if "complexity" in session_trace:
            observables["query_complexity"] = session_trace["complexity"].get("complexity_score", 0)
        if "recommendation" in session_trace:
            observables["recommended_strategy"] = session_trace["recommendation"].get("recommended_strategy", "")
        if "quality_metrics" in session_trace:
            observables["retrieval_quality"] = session_trace["quality_metrics"]
        if "answer_evaluation" in session_trace and session_trace["answer_evaluation"]:
            observables["answer_quality"] = session_trace["answer_evaluation"].get("combined_score", 0)

        return {"nodes": nodes, "edges": edges, "observables": observables}

    # ── Main Attribution Pipeline ──────────────────────────────────────────

    async def run_full_attribution(
        self,
        query: str,
        original_answer: str,
        chunks: List[Dict[str, Any]],
        original_strategy: str,
        original_topk: int,
        session_trace: Dict[str, Any],
        on_progress=None,
    ) -> AttributionReport:
        """
        Run full causal attribution analysis.

        Generates counterfactual interventions for each pipeline component,
        then uses two-stage ranking to identify the most impactful components.
        """
        start_time = time.time()

        if not chunks:
            return AttributionReport(
                session_id=0,
                query=query,
                original_strategy=original_strategy,
                original_quality=0.0,
                duration_ms=int((time.time() - start_time) * 1000),
            )

        original_quality = self._compute_answer_quality(query, original_answer, chunks)

        # Define all interventions to run
        all_strategies = ["vector", "hybrid", "graph"]
        alt_strategies = [s for s in all_strategies if s != original_strategy]
        topk_variants = [k for k in [3, 10, 15] if k != original_topk]

        approx_results: List[InterventionResult] = []

        # Phase 1: Approximate attribution for ALL interventions (local, fast)

        # Strategy interventions
        for alt_s in alt_strategies:
            try:
                retriever = retriever_registry.get(alt_s)
                alt_result = await retriever.retrieve(query, top_k=original_topk)
                alt_chunks = alt_result["chunks"]

                approx_delta = self._approx_intervene_strategy(
                    query, chunks, alt_chunks, original_quality
                )
                approx_results.append(InterventionResult(
                    component=ComponentType.STRATEGY,
                    intervention=InterventionType.SWITCH_STRATEGY,
                    params={"from": original_strategy, "to": alt_s},
                    original_quality=original_quality,
                    perturbed_quality=original_quality + approx_delta,
                    quality_delta=approx_delta,
                    is_approximate=True,
                    description=f"[APPROX] Switch strategy {original_strategy}→{alt_s}: ΔQ={approx_delta:+.4f}",
                ))
            except Exception as e:
                approx_results.append(InterventionResult(
                    component=ComponentType.STRATEGY,
                    intervention=InterventionType.SWITCH_STRATEGY,
                    params={"from": original_strategy, "to": alt_s, "error": str(e)},
                    original_quality=original_quality,
                    perturbed_quality=original_quality,
                    quality_delta=0.0,
                    is_approximate=True,
                    description=f"Strategy intervention failed: {e}",
                ))

        # Top-K interventions (approximate only by default — exact is expensive)
        for new_k in topk_variants:
            try:
                retriever = retriever_registry.get(original_strategy)
                alt_result = await retriever.retrieve(query, top_k=new_k)
                alt_chunks = alt_result["chunks"][:new_k]

                approx_delta = self._approx_intervene_strategy(
                    query, chunks, alt_chunks, original_quality
                )
                approx_results.append(InterventionResult(
                    component=ComponentType.TOP_K,
                    intervention=InterventionType.CHANGE_TOPK,
                    params={"from_k": original_topk, "to_k": new_k, "strategy": original_strategy},
                    original_quality=original_quality,
                    perturbed_quality=original_quality + approx_delta,
                    quality_delta=approx_delta,
                    is_approximate=True,
                    description=f"[APPROX] Change top_k {original_topk}→{new_k}: ΔQ={approx_delta:+.4f}",
                ))
            except Exception:
                pass

        # Chunk-level interventions: approximate importance for each chunk
        if len(chunks) > 1:
            for i in range(len(chunks)):
                approx_delta = self._approx_intervene_chunks(
                    original_answer, chunks, {i}, original_quality
                )
                approx_results.append(InterventionResult(
                    component=ComponentType.CHUNK_SELECTION,
                    intervention=InterventionType.REMOVE_CHUNKS,
                    params={"removed_indices": [i], "chunk_index": chunks[i].get("chunk_index", i)},
                    original_quality=original_quality,
                    perturbed_quality=original_quality + approx_delta,
                    quality_delta=approx_delta,
                    is_approximate=True,
                    description=f"[APPROX] Remove chunk {i}: ΔQ={approx_delta:+.4f}",
                ))

        # Context intervention (approximate via embedding of truncated context)
        try:
            truncated = chunks[:max(1, len(chunks) // 2)]
            approx_delta = self._approx_intervene_strategy(
                query, chunks, truncated, original_quality
            )
            approx_results.append(InterventionResult(
                component=ComponentType.CONTEXT_ASSEMBLY,
                intervention=InterventionType.COMPRESS_CONTEXT,
                params={"original_chunks": len(chunks), "truncated_chunks": len(truncated)},
                original_quality=original_quality,
                perturbed_quality=original_quality + approx_delta,
                quality_delta=approx_delta,
                is_approximate=True,
                description=f"[APPROX] Compress context {len(chunks)}→{len(truncated)} chunks: ΔQ={approx_delta:+.4f}",
            ))
        except Exception:
            pass

        if on_progress:
            on_progress("approximate_done", len(approx_results), 0)

        # Phase 2: Exact LLM verification for top-k highest-absolute-delta interventions
        # Sort by absolute quality delta (descending)
        approx_results.sort(key=lambda x: abs(x.quality_delta), reverse=True)

        exact_interventions = []
        llm_count = 0
        for i, approx in enumerate(approx_results):
            if i >= self.exact_top_k:
                exact_interventions.append(approx)
                continue
            if llm_count >= self.exact_top_k:
                exact_interventions.append(approx)
                continue

            # Run exact LLM intervention
            async with self.semaphore:
                try:
                    if approx.component == ComponentType.STRATEGY:
                        alt_s = approx.params.get("to", "")
                        if alt_s:
                            exact = await self.intervene_strategy(
                                query, original_answer, chunks,
                                original_strategy, alt_s, original_quality,
                            )
                            exact_interventions.append(exact)
                            llm_count += 1
                            if on_progress:
                                on_progress("llm_intervention", llm_count, self.exact_top_k)
                            continue
                    elif approx.component == ComponentType.CHUNK_SELECTION:
                        remove_set = set(approx.params.get("removed_indices", []))
                        if remove_set:
                            exact = await self.intervene_chunks(
                                query, original_answer, chunks, remove_set, original_quality,
                            )
                            exact_interventions.append(exact)
                            llm_count += 1
                            if on_progress:
                                on_progress("llm_intervention", llm_count, self.exact_top_k)
                            continue
                    elif approx.component == ComponentType.CONTEXT_ASSEMBLY:
                        exact = await self.intervene_context(
                            query, original_answer, chunks, original_quality,
                        )
                        exact_interventions.append(exact)
                        llm_count += 1
                        if on_progress:
                            on_progress("llm_intervention", llm_count, self.exact_top_k)
                        continue
                except Exception:
                    pass

                # Fallback: keep approximate result
                exact_interventions.append(approx)

        # Add remaining approximate results
        for i in range(self.exact_top_k, len(approx_results)):
            exact_interventions.append(approx_results[i])

        # Compute normalized attribution scores
        exact_interventions.sort(key=lambda x: abs(x.quality_delta), reverse=True)
        total_abs_delta = sum(abs(r.quality_delta) for r in exact_interventions)

        if total_abs_delta > 1e-10:
            for r in exact_interventions:
                r.attribution_score = round(abs(r.quality_delta) / total_abs_delta, 4)

        # Aggregate by component
        component_scores: Dict[str, float] = {}
        for r in exact_interventions:
            comp_key = r.component.value
            component_scores[comp_key] = component_scores.get(comp_key, 0.0) + r.attribution_score

        # Normalize component scores
        total_comp = sum(component_scores.values())
        if total_comp > 0:
            for k in component_scores:
                component_scores[k] = round(component_scores[k] / total_comp, 4)

        # Top contributors (both positive and negative)
        top_contributors = []
        for r in exact_interventions[:5]:
            top_contributors.append({
                "component": r.component.value,
                "intervention": r.intervention.value,
                "params": r.params,
                "quality_delta": r.quality_delta,
                "attribution_score": r.attribution_score,
                "is_approximate": r.is_approximate,
                "description": r.description,
            })

        causal_graph = self.build_causal_graph(session_trace)
        # Annotate graph nodes with attribution scores
        for node in causal_graph["nodes"]:
            node_id = node["id"]
            if node_id == "strategy":
                node["attribution"] = component_scores.get("strategy", 0)
            elif node_id == "retrieval":
                node["attribution"] = component_scores.get("chunk_selection", 0) + component_scores.get("top_k", 0)
            elif node_id == "context":
                node["attribution"] = component_scores.get("context_assembly", 0)
            elif node_id == "llm":
                node["attribution"] = component_scores.get("llm_generation", 0)

        report = AttributionReport(
            session_id=0,
            query=query,
            original_strategy=original_strategy,
            original_quality=original_quality,
            interventions=exact_interventions,
            component_attributions=component_scores,
            top_contributors=top_contributors,
            causal_graph=causal_graph,
            total_interventions=len(exact_interventions),
            llm_interventions=llm_count,
            duration_ms=int((time.time() - start_time) * 1000),
        )

        return report


causal_analyzer = CausalAttributionAnalyzer()
