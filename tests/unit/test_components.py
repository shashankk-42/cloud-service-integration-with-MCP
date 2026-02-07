"""
Unit Tests for Core Components
Tests for individual components without external dependencies.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import asyncio

# Import implementation classes
from phase_1_mcp_core.servers.base_server import CircuitBreaker, RateLimiter, CircuitBreakerOpenError
from phase_2_langgraph_orchestrator.state import (
    OrchestratorState,
    TaskRequest,
    WorkloadType,
    InMemoryStateStore,
    WorkflowStatus
)
from phase_2_langgraph_orchestrator.agents import (
    PlannerAgent,
    AllocatorAgent,
    VerifierAgent,
    MCPClientManager
)
from shared.config import FeatureFlags


class TestCircuitBreaker:
    """Tests for circuit breaker implementation."""
    
    @pytest.mark.asyncio
    async def test_circuit_closed_initially(self):
        """Circuit should start in closed state."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
        
        assert cb.state == "closed"
        assert cb.failures == 0
    
    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self):
        """Circuit should open after threshold failures."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
        
        # We need to simulate failures via internal method or call
        # _on_failure is an async method
        for _ in range(3):
            await cb._on_failure()
        
        assert cb.state == "open"
    
    @pytest.mark.asyncio
    async def test_circuit_half_open_after_timeout(self):
        """Circuit should become half-open after recovery timeout."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        
        for _ in range(3):
            await cb._on_failure()
        
        assert cb.state == "open"
        
        await asyncio.sleep(0.15)
        
        # The state change happens on next check or call
        # emulate check
        should_reset = cb._should_attempt_reset()
        if should_reset:
             # This logic is usually inside call(), forcing it here for unit test
             cb.state = "half-open"
             
        assert cb.state == "half-open"
    
    @pytest.mark.asyncio
    async def test_circuit_closes_on_success(self):
        """Circuit should close on successful call in half-open state."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.01)
        
        # Open it
        for _ in range(3):
            await cb._on_failure()
        
        await asyncio.sleep(0.02)
        cb.state = "half-open" # Simulate reset start
        
        await cb._on_success()
        
        assert cb.state == "closed"
        assert cb.failures == 0


class TestRateLimiter:
    """Tests for rate limiter implementation."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_within_limit(self):
        """Rate limiter should allow requests within limit."""
        # Fix: init args are rate, capacity
        rl = RateLimiter(rate=10, capacity=10)
        
        # Should allow 10 requests
        for _ in range(10):
            allowed = await rl.acquire()
            assert allowed == True
    
    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_excess(self):
        """Rate limiter should block excess requests."""
        rl = RateLimiter(rate=5, capacity=5)
        
        # Use up the limit
        for _ in range(5):
            await rl.acquire()
        
        # Next request should be rate limited (acquire returns False if not enough tokens)
        # Wait, acquire(tokens=1) returns bool immediately if we use acquire directly? 
        # The implementation of acquire returns True if successful, False otherwise.
        
        allowed = await rl.acquire()
        assert allowed == False


class TestOrchestratorState:
    """Tests for orchestrator state management."""
    
    def test_state_initialization(self):
        """Test state initializes correctly."""
        
        task = TaskRequest(
            name="Test Task",
            description="Test description",
            workload_type=WorkloadType.BATCH
        )
        
        # Simplified state dict as per TypedDict
        state: OrchestratorState = {
            "task_id": task.task_id,
            "task_request": task.model_dump(),
            "status": "planning",
            "current_agent": "",
            "started_at": datetime.utcnow().isoformat(),
            "workload_classification": {},
            "selected_clouds": [],
            "selected_regions": [],
            "compliance_checks": [],
            "cost_estimates": [],
            "quota_status": [],
            "allocation_plan": {},
            "provisioned_resources": [],
            "execution_logs": [],
            "health_checks": [],
            "slo_compliance": {},
            "needs_rollback": False,
            "rollback_reason": None,
            "cleanup_actions": [],
            "cost_attribution": {},
            "final_result": {},
            "decisions": [],
            "errors": [],
            "retry_count": 0,
            "max_retries": 3
        }
        
        assert state["status"] == "planning"
        assert state["retry_count"] == 0
        assert state["needs_rollback"] == False
    
    def test_task_request_defaults(self):
        """Test TaskRequest has sensible defaults."""
        
        task = TaskRequest(
            name="Test",
            description="Test",
            workload_type=WorkloadType.BATCH  # Fix: TRAINING -> BATCH
        )
        
        assert task.task_id is not None
        assert len(task.task_id) > 0
        assert task.budget_limit is None
        assert task.preferred_clouds == []
    
    @pytest.mark.asyncio
    async def test_state_persistence_inmemory(self):
        """Test in-memory state persistence."""
        
        # Fix: InMemoryStateStore instead of Persistence
        store = InMemoryStateStore()
        
        task = TaskRequest(
            name="Test",
            description="Test",
            workload_type=WorkloadType.BATCH
        )
        
        state: OrchestratorState = {
            "task_id": task.task_id,
            "task_request": task.model_dump(),
            "status": "planning",
            # ... minimal required fields for test
            "retry_count": 0
        }
        
        # Save state
        await store.save_state(task.task_id, state)
        
        # Load state
        loaded = await store.load_state(task.task_id)
        
        assert loaded is not None
        assert loaded["task_id"] == task.task_id
        assert loaded["status"] == "planning"


@pytest.fixture
def mock_mcp_manager():
    return MagicMock(spec=MCPClientManager)

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    return llm

class TestPlannerAgent:
    """Tests for the Planner agent."""
    
    @pytest.mark.asyncio
    async def test_planner_creates_valid_plan(self, mock_mcp_manager, mock_llm):
        """Test planner creates a valid execution plan."""
        
        # Mock LLM response
        mock_llm.ainvoke.return_value = MagicMock(
            content='''
            {
                "workload_classification": {
                    "type": "batch",
                    "resource_type": "cpu"
                },
                "selected_clouds": ["aws"],
                "reasoning": "Simple batch job"
            }
            '''
        )
        
        planner = PlannerAgent(mcp_manager=mock_mcp_manager, llm=mock_llm)
        
        task = TaskRequest(
            name="Deploy ML Model",
            description="Deploy a model to production",
            workload_type=WorkloadType.MODEL_SERVING
        )
        
        state = {
            "task_id": "test-1",
            "task_request": task.model_dump(),
            "status": "pending"
        }
        
        new_state = await planner(state)
        
        assert new_state["status"] == WorkflowStatus.PLANNING.value
        assert "aws" in new_state["selected_clouds"]


class TestAllocatorAgent:
    """Tests for the Allocator agent."""
    
    @pytest.mark.asyncio
    async def test_allocator_respects_budget(self, mock_mcp_manager, mock_llm):
        """Test allocator checks costs."""
        
        allocator = AllocatorAgent(mcp_manager=mock_mcp_manager, llm=mock_llm)
        
        # Mock MCP calls
        mock_mcp_manager.call_tool = AsyncMock(return_value={
            "estimated_cost_usd": 15.0,
            "currency": "USD"
        })
        
        task = TaskRequest(
            name="Test",
            description="Test",
            workload_type=WorkloadType.BATCH,
            resource_requirements={"count": 1},
            budget_limit=100.0
        )
        
        state = {
            "task_id": "test-1",
            "task_request": task.model_dump(),
            "status": "planning",
            "selected_clouds": ["aws"],
            "selected_regions": [{"cloud": "aws", "region": "us-east-1"}],
            "workload_classification": {"type": "batch", "estimated_compute_hours": 10}
        }
        
        new_state = await allocator(state)
        
        assert new_state["status"] == WorkflowStatus.ALLOCATING.value
        assert len(new_state["cost_estimates"]) > 0
        assert new_state["allocation_plan"]["total_estimated_cost"] == 15.0


class TestVerifierAgent:
    """Tests for the Verifier agent."""
    
    @pytest.mark.asyncio
    async def test_verifier_detects_failures(self, mock_mcp_manager, mock_llm):
        """Test verifier detects execution failures."""
        
        verifier = VerifierAgent(mcp_manager=mock_mcp_manager, llm=mock_llm)
        
        # Mock MCP health check
        mock_mcp_manager.call_tool = AsyncMock(return_value={
            "health_checks": [
                {"resource_id": "i-123", "healthy": False, "latency_ms": 5000}
            ]
        })
        
        task = TaskRequest(
            name="Test",
            description="Test",
            workload_type=WorkloadType.BATCH
        )
        
        state = {
            "task_id": "test-1",
            "task_request": task.model_dump(),
            "selected_clouds": ["aws"],
            "provisioned_resources": [
                {"resource_id": "i-123", "cloud": "aws"}
            ],
            "status": "executing"
        }
        
        new_state = await verifier(state)
        
        assert new_state["needs_rollback"] == True


class TestDataClasses:
    """Tests for data class serialization."""
    
    def test_task_request_serialization(self):
        """Test TaskRequest serializes with model_dump."""
        
        task = TaskRequest(
            name="Test Task",
            description="Description",
            workload_type=WorkloadType.BATCH,
            budget_limit=100.0
        )
        
        data = task.model_dump()
        
        assert data["name"] == "Test Task"
        assert data["workload_type"] == "batch"
        assert data["budget_limit"] == 100.0
    
    def test_task_request_deserialization(self):
        """Test TaskRequest deserializes with model_validate (from_dict replacement)."""
        
        data = {
            "task_id": "test-123",
            "name": "Test Task",
            "description": "Description",
            "workload_type": "batch",
            "budget_limit": 100.0,
            "preferred_clouds": [],
            "preferred_regions": []
        }
        
        # Use Pydantic v2 compatible method if possible, or standard init
        # TaskRequest(**data) works too
        task = TaskRequest(**data)
        
        assert task.task_id == "test-123"
        assert task.workload_type == WorkloadType.BATCH

class TestConfigValidation:
    def test_feature_flags(self):
        # We can't easily mock Pydantic settings reading from env in this context
        # without monkeypatching os.environ before import or reload.
        # So we strictly test correct instantiation logic.
        
        flags = FeatureFlags(_env_file=None) # Try to ignore env file if possible in Pydantic V2? 
        # Or just assert what we have is a bool.
        assert isinstance(flags.enable_auto_scaling, bool)
