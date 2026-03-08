from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class VerifierMode(str, Enum):
    predictive = "predictive"
    symbolic = "symbolic"
    simulation = "simulation"


class ObservationExample(BaseModel):
    features: dict[str, Any]
    target: Any


class SimulationCase(BaseModel):
    features: dict[str, Any]
    expected_target: Any


class BenchmarkTask(BaseModel):
    family: str
    task_id: str
    description: str
    feature_names: list[str]
    train: list[ObservationExample]
    heldout: list[ObservationExample]
    ground_truth_expression: str
    ground_truth_rule_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    simulation_cases: list[SimulationCase] = Field(default_factory=list)
    ood: list[ObservationExample] = Field(default_factory=list)
    stress: list[ObservationExample] = Field(default_factory=list)


class Hypothesis(BaseModel):
    hypothesis_id: str
    source: str
    round_index: int
    family: str
    task_id: str
    rule_text: str
    expression: str
    rationale: str
    confidence: float = 0.0
    parents: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)


class CollisionScore(BaseModel):
    structural_complementarity: float
    predictive_overlap: float
    explanatory_gain: float
    novelty: float
    collision_strength: float


class VerificationResult(BaseModel):
    hypothesis_id: str
    mode: VerifierMode
    passed: bool
    confidence: float
    reason: str
    metrics: dict[str, float] = Field(default_factory=dict)


class VerifierVerdict(BaseModel):
    hypothesis_id: str
    passed: bool
    confidence: float
    reason: str
    per_mode: list[VerificationResult]


class CandidateLogRecord(BaseModel):
    run_id: str
    method: str
    trial_index: int
    round_index: int
    family: str
    task_id: str
    hypothesis_id: str
    source: str
    expression: str
    confidence: float
    novelty: float
    predictive_accuracy: float
    symbolic_accuracy: float
    simulation_accuracy: float
    ood_accuracy: float = 0.0
    stress_accuracy: float = 0.0
    transfer_score: float = 0.0
    open_world_score: float = 0.0
    accepted: bool
    exact_match: bool


class RoundSummary(BaseModel):
    round_index: int
    family: str
    task_id: str
    candidate_count: int
    accepted_count: int
    validity_rate: float
    top_heldout_accuracy: float
    top_ood_accuracy: float = 0.0
    top_stress_accuracy: float = 0.0
    average_novelty: float
    time_to_valid_discovery: Optional[int]


class MethodSummary(BaseModel):
    method: str
    trial_index: int
    validity_rate: float
    heldout_predictive_accuracy: float
    ood_predictive_accuracy: float = 0.0
    stress_predictive_accuracy: float = 0.0
    transfer_generalization_score: float = 0.0
    open_world_readiness_score: float = 0.0
    rule_recovery_exact_match_rate: float
    compression_score: float
    novelty_score: float
    time_to_valid_discovery: float
    cumulative_improvement: float
    rounds: list[RoundSummary]


class ExperimentSummary(BaseModel):
    run_name: str
    config: dict[str, Any]
    method_summaries: list[MethodSummary]
    uplift: dict[str, dict[str, float]]
