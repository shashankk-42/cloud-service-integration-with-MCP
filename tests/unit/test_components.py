"""
Unit Tests for Core Components
Tests for individual components without external dependencies.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta


class TestCircuitBreaker:
    """Tests for circuit breaker implementation."""
    
    def test_circuit_closed_initially(self):
        """Circuit should start in closed state."""
        from phase_1_mcp_core.servers.base_server import CircuitBreaker
        
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
        
        assert cb.state == "closed"
        assert cb.failure_count == 0
    
    def test_circuit_opens_after_failures(self):
        """Circuit should open after threshold failures."""
        from phase_1_mcp_core.servers.base_server import CircuitBreaker
        
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
        
        for _ in range(3):
            cb.record_failure()
        
        assert cb.state == "open"
    
    def test_circuit_half_open_after_timeout(self):
        """Circuit should become half-open after recovery timeout."""
        from phase_1_mcp_core.servers.base_server import CircuitBreaker
        
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        
        for _ in range(3):
            cb.record_failure()
        
        assert cb.state == "open"
        
        import time
        time.sleep(0.15)
        
        assert cb.state == "half_open"
    
    def test_circuit_closes_on_success(self):
        """Circuit should close on successful call in half-open state."""
        from phase_1_mcp_core.servers.base_server import CircuitBreaker
        
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.01)
        
        for _ in range(3):
            cb.record_failure()
        
        import time
        time.sleep(0.02)
        
        cb.record_success()
        
        assert cb.state == "closed"
        assert cb.failure_count == 0


class TestRateLimiter:
    """Tests for rate limiter implementation."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_within_limit(self):
        """Rate limiter should allow requests within limit."""
        from phase_1_mcp_core.servers.base_server import RateLimiter
        
        rl = RateLimiter(requests_per_second=10)
        
        # Should allow 10 requests
        for _ in range(10):
            allowed = await rl.acquire()
            assert allowed == True
    
    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_excess(self):
        """Rate limiter should block excess requests."""
        from phase_1_mcp_core.servers.base_server import RateLimiter
        
        rl = RateLimiter(requests_per_second=5)
        
        # Use up the limit
        for _ in range(5):
            await rl.acquire()
        
        # Next request should be rate limited
        import asyncio
        allowed = await asyncio.wait_for(
            rl.try_acquire(),
            timeout=0.1
        )
        
        assert allowed == False


class TestOrchestratorState:
    """Tests for orchestrator state management."""
    
    def test_state_initialization(self):
        """Test state initializes correctly."""
        from phase_2_langgraph_orchestrator.state import OrchestratorState, TaskRequest, WorkloadType
        
        task = TaskRequest(
            name="Test Task",
            description="Test description",
            workload_type=WorkloadType.BATCH
        )
        
        state: OrchestratorState = {
            "task_request": task,
            "current_phase": "planning",
            "plan": None,
            "allocation": None,
            "execution_results": [],
            "verification_results": [],
            "final_output": None,
            "error": None,
            "retry_count": 0,
            "should_rollback": False
        }
        
        assert state["current_phase"] == "planning"
        assert state["retry_count"] == 0
        assert state["should_rollback"] == False
    
    def test_task_request_defaults(self):
        """Test TaskRequest has sensible defaults."""
        from phase_2_langgraph_orchestrator.state import TaskRequest, WorkloadType
        
        task = TaskRequest(
            name="Test",
            description="Test",
            workload_type=WorkloadType.TRAINING
        )
        
        assert task.task_id is not None
        assert len(task.task_id) > 0
        assert task.budget_limit == 0.0  # No budget constraint
        assert task.preferred_clouds == []  # Any cloud
    
    @pytest.mark.asyncio
    async def test_state_persistence_inmemory(self):
        """Test in-memory state persistence."""
        from phase_2_langgraph_orchestrator.state import (
            InMemoryStatePersistence,
            OrchestratorState,
            TaskRequest,
            WorkloadType
        )
        
        persistence = InMemoryStatePersistence()
        
        task = TaskRequest(
            name="Test",
            description="Test",
            workload_type=WorkloadType.BATCH
        )
        
        state: OrchestratorState = {
            "task_request": task,
            "current_phase": "planning",
            "plan": None,
            "allocation": None,
            "execution_results": [],
            "verification_results": [],
            "final_output": None,
            "error": None,
            "retry_count": 0,
            "should_rollback": False
        }
        
        # Save state
        await persistence.save_state(task.task_id, state)
        
        # Load state
        loaded = await persistence.load_state(task.task_id)
        
        assert loaded is not None
        assert loaded["task_request"].task_id == task.task_id
        assert loaded["current_phase"] == "planning"


class TestPlannerAgent:
    """Tests for the Planner agent."""
    
    @pytest.mark.asyncio
    async def test_planner_creates_valid_plan(self):
        """Test planner creates a valid execution plan."""
        from phase_2_langgraph_orchestrator.agents import PlannerAgent
        from phase_2_langgraph_orchestrator.state import TaskRequest, WorkloadType
        
        with patch('phase_2_langgraph_orchestrator.agents.ChatOpenAI') as mock_llm:
            mock_llm.return_value.agenerate.return_value = MagicMock(
                generations=[[MagicMock(text='''
                {
                    "steps": [
                        {"id": "1", "action": "create_cluster", "cloud": "aws", "params": {}},
                        {"id": "2", "action": "deploy_model", "cloud": "aws", "params": {}}
                    ],
                    "estimated_duration_minutes": 30,
                    "estimated_cost_usd": 5.0
                }
                ''')]]
            )
            
            planner = PlannerAgent()
            
            task = TaskRequest(
                name="Deploy ML Model",
                description="Deploy a model to production",
                workload_type=WorkloadType.INFERENCE
            )
            
            plan = await planner.create_plan(task)
            
            assert plan is not None
            assert len(plan.steps) >= 1


class TestAllocatorAgent:
    """Tests for the Allocator agent."""
    
    @pytest.mark.asyncio
    async def test_allocator_respects_budget(self):
        """Test allocator respects budget constraints."""
        from phase_2_langgraph_orchestrator.agents import AllocatorAgent
        
        allocator = AllocatorAgent()
        
        # Request exceeding budget
        allocation = await allocator.allocate_resources(
            plan_steps=[
                {"id": "1", "action": "provision_compute", "cloud": "aws", "params": {"instance_type": "p4d.24xlarge"}}
            ],
            budget_limit=10.0,  # Very low budget
            preferred_clouds=["aws"]
        )
        
        # Should either find cheaper alternative or fail gracefully
        assert allocation is not None
        if allocation.total_estimated_cost > 10.0:
            assert allocation.warnings is not None
    
    @pytest.mark.asyncio
    async def test_allocator_respects_cloud_preference(self):
        """Test allocator respects cloud preferences."""
        from phase_2_langgraph_orchestrator.agents import AllocatorAgent
        
        allocator = AllocatorAgent()
        
        allocation = await allocator.allocate_resources(
            plan_steps=[
                {"id": "1", "action": "provision_compute", "params": {"instance_type": "t3.medium"}}
            ],
            budget_limit=100.0,
            preferred_clouds=["gcp"]  # Only GCP
        )
        
        assert allocation is not None
        for resource in allocation.resources:
            assert resource.cloud == "gcp"


class TestVerifierAgent:
    """Tests for the Verifier agent."""
    
    @pytest.mark.asyncio
    async def test_verifier_detects_failures(self):
        """Test verifier detects execution failures."""
        from phase_2_langgraph_orchestrator.agents import VerifierAgent
        
        verifier = VerifierAgent()
        
        execution_results = [
            {"step_id": "1", "status": "success"},
            {"step_id": "2", "status": "failed", "error": "Resource not found"}
        ]
        
        verification = await verifier.verify(execution_results)
        
        assert verification.all_passed == False
        assert len(verification.failed_steps) == 1
        assert "2" in verification.failed_steps
    
    @pytest.mark.asyncio
    async def test_verifier_recommends_rollback(self):
        """Test verifier recommends rollback for critical failures."""
        from phase_2_langgraph_orchestrator.agents import VerifierAgent
        
        verifier = VerifierAgent()
        
        # Critical failure scenario
        execution_results = [
            {"step_id": "1", "status": "success"},
            {"step_id": "2", "status": "failed", "error": "Critical: Data corruption detected", "critical": True}
        ]
        
        verification = await verifier.verify(execution_results)
        
        assert verification.should_rollback == True


class TestCostCalculations:
    """Tests for cost calculation functions."""
    
    def test_spot_savings_calculation(self):
        """Test spot instance savings calculation."""
        from phase_4_cost_resilience.cost_optimizer import calculate_spot_savings
        
        on_demand_price = 10.0
        spot_price = 3.0
        
        savings = calculate_spot_savings(on_demand_price, spot_price)
        
        assert savings == 70.0  # 70% savings
    
    def test_monthly_cost_projection(self):
        """Test monthly cost projection."""
        from phase_4_cost_resilience.cost_optimizer import project_monthly_cost
        
        hourly_cost = 0.10
        hours_per_day = 8
        
        monthly = project_monthly_cost(hourly_cost, hours_per_day)
        
        # 0.10 * 8 * 30 = 24.0
        assert monthly == pytest.approx(24.0, rel=0.01)


class TestConfigValidation:
    """Tests for configuration validation."""
    
    def test_cloud_config_validation(self):
        """Test cloud configuration validation."""
        from shared.config import CloudConfig
        
        # Valid config
        config = CloudConfig(
            aws_enabled=True,
            aws_region="us-east-1",
            azure_enabled=False,
            gcp_enabled=True,
            gcp_project_id="my-project"
        )
        
        assert config.aws_enabled == True
        assert config.aws_region == "us-east-1"
    
    def test_security_config_defaults(self):
        """Test security configuration has safe defaults."""
        from shared.config import SecurityConfig
        
        config = SecurityConfig()
        
        # Should have secure defaults
        assert config.require_mfa_for_admin == True
        assert config.secret_rotation_days <= 90
        assert config.audit_log_retention_days >= 365
    
    def test_feature_flags(self):
        """Test feature flags configuration."""
        from shared.config import FeatureFlags
        
        flags = FeatureFlags(
            enable_spot_instances=True,
            enable_multi_cloud=True,
            enable_auto_scaling=False
        )
        
        assert flags.enable_spot_instances == True
        assert flags.enable_auto_scaling == False


class TestDataClasses:
    """Tests for data class serialization."""
    
    def test_task_request_to_dict(self):
        """Test TaskRequest serializes to dict."""
        from phase_2_langgraph_orchestrator.state import TaskRequest, WorkloadType
        
        task = TaskRequest(
            name="Test Task",
            description="Description",
            workload_type=WorkloadType.TRAINING,
            budget_limit=100.0
        )
        
        data = task.to_dict()
        
        assert data["name"] == "Test Task"
        assert data["workload_type"] == "training"
        assert data["budget_limit"] == 100.0
    
    def test_task_request_from_dict(self):
        """Test TaskRequest deserializes from dict."""
        from phase_2_langgraph_orchestrator.state import TaskRequest, WorkloadType
        
        data = {
            "task_id": "test-123",
            "name": "Test Task",
            "description": "Description",
            "workload_type": "training",
            "budget_limit": 100.0
        }
        
        task = TaskRequest.from_dict(data)
        
        assert task.task_id == "test-123"
        assert task.workload_type == WorkloadType.TRAINING


# Pytest configuration

@pytest.fixture
def sample_task():
    """Provide a sample task for tests."""
    from phase_2_langgraph_orchestrator.state import TaskRequest, WorkloadType
    
    return TaskRequest(
        name="Sample Task",
        description="A sample task for testing",
        workload_type=WorkloadType.BATCH,
        resource_requirements={
            "cpu": 4,
            "memory_gb": 16
        },
        budget_limit=50.0,
        preferred_clouds=["aws", "gcp"]
    )


@pytest.fixture
def mock_mcp_client():
    """Provide a mock MCP client."""
    client = AsyncMock()
    client.call_tool = AsyncMock(return_value={"status": "success"})
    return client
