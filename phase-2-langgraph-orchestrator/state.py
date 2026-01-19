"""
LangGraph Orchestrator State Management
Defines the state schema and persistence for the orchestration workflow.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from operator import add
import uuid

from pydantic import BaseModel, Field


class WorkloadType(str, Enum):
    """Type of workload being processed."""
    LATENCY_SENSITIVE = "latency_sensitive"
    BATCH = "batch"
    MODEL_SERVING = "model_serving"
    DATA_PIPELINE = "data_pipeline"


class ResourceType(str, Enum):
    """Type of compute resource."""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    MEMORY_OPTIMIZED = "memory_optimized"


class CloudProvider(str, Enum):
    """Supported cloud providers."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"


class WorkflowStatus(str, Enum):
    """Status of the orchestration workflow."""
    PENDING = "pending"
    PLANNING = "planning"
    ALLOCATING = "allocating"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    FINISHING = "finishing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class DecisionType(str, Enum):
    """Type of decision made by agents."""
    CLOUD_SELECTION = "cloud_selection"
    REGION_SELECTION = "region_selection"
    INSTANCE_SELECTION = "instance_selection"
    SPOT_DECISION = "spot_decision"
    SCALING_DECISION = "scaling_decision"
    FAILOVER_DECISION = "failover_decision"
    ROLLBACK_DECISION = "rollback_decision"


# ============== Pydantic Models for Input/Output ==============

class TaskRequest(BaseModel):
    """Input task request from user."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    workload_type: WorkloadType
    resource_requirements: Dict[str, Any] = Field(default_factory=dict)
    slo_requirements: Dict[str, Any] = Field(default_factory=dict)
    budget_limit: Optional[float] = None
    preferred_clouds: List[CloudProvider] = Field(default_factory=list)
    preferred_regions: List[str] = Field(default_factory=list)
    data_residency_requirements: List[str] = Field(default_factory=list)
    deadline: Optional[datetime] = None
    priority: int = Field(default=5, ge=1, le=10)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResourceSpec(BaseModel):
    """Specification for a cloud resource."""
    cloud: CloudProvider
    region: str
    instance_type: str
    count: int = 1
    spot: bool = False
    max_price: Optional[float] = None
    tags: Dict[str, str] = Field(default_factory=dict)


class AllocationPlan(BaseModel):
    """Resource allocation plan from Allocator."""
    resources: List[ResourceSpec]
    estimated_cost: float
    estimated_duration: Optional[int] = None  # seconds
    fallback_resources: Optional[List[ResourceSpec]] = None


class ProvisionedResource(BaseModel):
    """Record of a provisioned resource."""
    resource_id: str
    cloud: CloudProvider
    region: str
    instance_type: str
    status: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthCheckResult(BaseModel):
    """Result of a health check."""
    resource_id: str
    healthy: bool
    status: str
    latency_ms: Optional[float] = None
    error_rate: Optional[float] = None
    checked_at: datetime = Field(default_factory=datetime.utcnow)


class Decision(BaseModel):
    """Record of a decision made by an agent."""
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent: str
    decision_type: DecisionType
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    policy_checks: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WorkflowResult(BaseModel):
    """Final result of the orchestration workflow."""
    task_id: str
    status: WorkflowStatus
    provisioned_resources: List[ProvisionedResource] = Field(default_factory=list)
    total_cost: float = 0.0
    duration_seconds: float = 0.0
    decisions: List[Decision] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    completed_at: Optional[datetime] = None


# ============== LangGraph State Definition ==============

class OrchestratorState(TypedDict):
    """
    State schema for the LangGraph orchestration workflow.
    This state is passed between all agents in the workflow.
    """
    
    # Task Information
    task_id: str
    task_request: Dict[str, Any]
    
    # Workflow Status
    status: str
    current_agent: str
    started_at: str
    
    # Planning Phase Outputs
    workload_classification: Dict[str, Any]
    selected_clouds: List[str]
    selected_regions: List[str]
    compliance_checks: List[Dict[str, Any]]
    
    # Allocation Phase Outputs
    cost_estimates: List[Dict[str, Any]]
    quota_status: List[Dict[str, Any]]
    allocation_plan: Dict[str, Any]
    
    # Execution Phase Outputs
    provisioned_resources: Annotated[List[Dict[str, Any]], add]
    execution_logs: Annotated[List[str], add]
    
    # Verification Phase Outputs
    health_checks: List[Dict[str, Any]]
    slo_compliance: Dict[str, Any]
    needs_rollback: bool
    rollback_reason: Optional[str]
    
    # Finisher Phase Outputs
    cleanup_actions: List[Dict[str, Any]]
    cost_attribution: Dict[str, Any]
    final_result: Dict[str, Any]
    
    # Decision Tracking
    decisions: Annotated[List[Dict[str, Any]], add]
    
    # Error Handling
    errors: Annotated[List[str], add]
    retry_count: int
    max_retries: int


def create_initial_state(task_request: TaskRequest) -> OrchestratorState:
    """Create the initial state for a new orchestration workflow."""
    return OrchestratorState(
        task_id=task_request.task_id,
        task_request=task_request.model_dump(),
        status=WorkflowStatus.PENDING.value,
        current_agent="",
        started_at=datetime.utcnow().isoformat(),
        workload_classification={},
        selected_clouds=[],
        selected_regions=[],
        compliance_checks=[],
        cost_estimates=[],
        quota_status=[],
        allocation_plan={},
        provisioned_resources=[],
        execution_logs=[],
        health_checks=[],
        slo_compliance={},
        needs_rollback=False,
        rollback_reason=None,
        cleanup_actions=[],
        cost_attribution={},
        final_result={},
        decisions=[],
        errors=[],
        retry_count=0,
        max_retries=3
    )


# ============== State Persistence ==============

from abc import ABC, abstractmethod


class StateStore(ABC):
    """Abstract base class for state persistence."""
    
    @abstractmethod
    async def save_state(self, task_id: str, state: OrchestratorState) -> None:
        """Save workflow state."""
        pass
    
    @abstractmethod
    async def load_state(self, task_id: str) -> Optional[OrchestratorState]:
        """Load workflow state."""
        pass
    
    @abstractmethod
    async def delete_state(self, task_id: str) -> None:
        """Delete workflow state."""
        pass
    
    @abstractmethod
    async def list_active_workflows(self) -> List[str]:
        """List all active workflow task IDs."""
        pass


class InMemoryStateStore(StateStore):
    """In-memory state store for testing."""
    
    def __init__(self):
        self._states: Dict[str, OrchestratorState] = {}
    
    async def save_state(self, task_id: str, state: OrchestratorState) -> None:
        self._states[task_id] = state
    
    async def load_state(self, task_id: str) -> Optional[OrchestratorState]:
        return self._states.get(task_id)
    
    async def delete_state(self, task_id: str) -> None:
        self._states.pop(task_id, None)
    
    async def list_active_workflows(self) -> List[str]:
        return [
            task_id for task_id, state in self._states.items()
            if state['status'] not in [
                WorkflowStatus.COMPLETED.value,
                WorkflowStatus.FAILED.value
            ]
        ]


class PostgresStateStore(StateStore):
    """PostgreSQL state store for production."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._pool = None
    
    async def _get_pool(self):
        if self._pool is None:
            import asyncpg
            self._pool = await asyncpg.create_pool(self.connection_string)
        return self._pool
    
    async def save_state(self, task_id: str, state: OrchestratorState) -> None:
        import json
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO workflow_states (task_id, state, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (task_id) 
                DO UPDATE SET state = $2, updated_at = NOW()
            """, task_id, json.dumps(state))
    
    async def load_state(self, task_id: str) -> Optional[OrchestratorState]:
        import json
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT state FROM workflow_states WHERE task_id = $1",
                task_id
            )
            if row:
                return json.loads(row['state'])
        return None
    
    async def delete_state(self, task_id: str) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM workflow_states WHERE task_id = $1",
                task_id
            )
    
    async def list_active_workflows(self) -> List[str]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT task_id FROM workflow_states 
                WHERE state->>'status' NOT IN ('completed', 'failed')
            """)
            return [row['task_id'] for row in rows]


# SQL schema for state store
STATE_STORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS workflow_states (
    task_id VARCHAR(255) PRIMARY KEY,
    state JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_workflow_states_status 
ON workflow_states ((state->>'status'));

CREATE TABLE IF NOT EXISTS workflow_decisions (
    decision_id VARCHAR(255) PRIMARY KEY,
    task_id VARCHAR(255) REFERENCES workflow_states(task_id),
    agent VARCHAR(100) NOT NULL,
    decision_type VARCHAR(100) NOT NULL,
    input_data JSONB,
    output_data JSONB,
    confidence FLOAT,
    reasoning TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_workflow_decisions_task 
ON workflow_decisions (task_id);
"""
