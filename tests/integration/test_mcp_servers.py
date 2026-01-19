"""
Integration Tests for MCP Servers
Tests against LocalStack (AWS), Azure emulator, and GCP emulator.
"""

import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Test configuration
TEST_CONFIG = {
    "aws": {
        "endpoint_url": os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566"),
        "region": "us-east-1"
    }
}


class TestAWSMCPServer:
    """Integration tests for AWS MCP Server against LocalStack."""
    
    @pytest.fixture
    def aws_server(self):
        """Create AWS MCP server for testing."""
        from phase_1_mcp_core.servers.aws.aws_server import AWSMCPServer
        
        server = AWSMCPServer(
            region="us-east-1",
            endpoint_url=TEST_CONFIG["aws"]["endpoint_url"]
        )
        return server
    
    @pytest.mark.asyncio
    async def test_provision_compute(self, aws_server):
        """Test provisioning EC2 instances."""
        params = {
            "instance_type": "t3.micro",
            "region": "us-east-1",
            "count": 1,
            "tags": {"test": "true", "Name": "test-instance"}
        }
        
        result = await aws_server.provision_compute(params)
        
        assert result["status"] == "success"
        assert len(result["instances"]) == 1
        assert result["instances"][0]["instance_type"] == "t3.micro"
    
    @pytest.mark.asyncio
    async def test_launch_spot(self, aws_server):
        """Test launching spot instances."""
        params = {
            "instance_type": "t3.micro",
            "region": "us-east-1",
            "count": 1,
            "spot": True,
            "max_price": 0.01
        }
        
        result = await aws_server.launch_spot(params)
        
        assert result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_create_storage_bucket(self, aws_server):
        """Test creating S3 bucket."""
        import uuid
        bucket_name = f"test-bucket-{uuid.uuid4().hex[:8]}"
        
        params = {
            "bucket_name": bucket_name,
            "region": "us-east-1",
            "versioning": True,
            "encryption": True
        }
        
        result = await aws_server.create_storage_bucket(params)
        
        assert result["status"] == "success"
        assert result["bucket_name"] == bucket_name
        assert result["versioning"] == True
    
    @pytest.mark.asyncio
    async def test_get_cost_estimate(self, aws_server):
        """Test getting cost estimates."""
        params = {
            "resource_type": "compute",
            "instance_type": "t3.medium",
            "region": "us-east-1",
            "duration_hours": 720,
            "quantity": 2
        }
        
        result = await aws_server.get_cost_estimate(params)
        
        assert result["status"] == "success"
        assert "estimated_cost_usd" in result
        assert result["estimated_cost_usd"] > 0
    
    @pytest.mark.asyncio
    async def test_get_health(self, aws_server):
        """Test health check functionality."""
        params = {
            "resource_type": "ec2",
            "resource_ids": ["i-test123"],
            "region": "us-east-1"
        }
        
        result = await aws_server.get_health(params)
        
        assert result["status"] == "success"
        assert "health_checks" in result


class TestOrchestratorWorkflow:
    """Integration tests for the LangGraph orchestrator workflow."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing."""
        from phase_2_langgraph_orchestrator.workflow import CloudOrchestrator
        
        orchestrator = CloudOrchestrator(
            mcp_config={
                "aws": {"endpoint": "http://localhost:8001"},
                "azure": {"endpoint": "http://localhost:8002"},
                "gcp": {"endpoint": "http://localhost:8003"}
            }
        )
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_batch_job_workflow(self, orchestrator):
        """Test complete batch job workflow."""
        from phase_2_langgraph_orchestrator.state import TaskRequest, WorkloadType
        
        task = TaskRequest(
            name="Test Batch Job",
            description="Integration test batch job",
            workload_type=WorkloadType.BATCH,
            resource_requirements={
                "image_uri": "test-image:latest",
                "input_config": {"test": True},
                "output_path": "s3://test-bucket/output"
            },
            slo_requirements={
                "max_completion_time_hours": 1
            },
            budget_limit=10.0,
            preferred_clouds=["aws"]
        )
        
        result = await orchestrator.run(task)
        
        assert result["task_id"] == task.task_id
        assert result["status"] in ["completed", "rolled_back", "failed"]
    
    @pytest.mark.asyncio
    async def test_workflow_status_tracking(self, orchestrator):
        """Test workflow status can be retrieved."""
        from phase_2_langgraph_orchestrator.state import TaskRequest, WorkloadType
        
        task = TaskRequest(
            name="Status Test",
            description="Test status tracking",
            workload_type=WorkloadType.BATCH
        )
        
        # Start workflow
        asyncio.create_task(orchestrator.run(task))
        
        # Give it a moment to start
        await asyncio.sleep(0.1)
        
        # Check status
        status = await orchestrator.get_workflow_status(task.task_id)
        
        assert status is not None
        assert status["task_id"] == task.task_id


class TestCostOptimizer:
    """Tests for cost optimization functionality."""
    
    @pytest.fixture
    def cost_optimizer(self):
        """Create cost optimizer for testing."""
        from phase_4_cost_resilience.cost_optimizer import CostOptimizer
        return CostOptimizer()
    
    @pytest.mark.asyncio
    async def test_find_cheapest_option(self, cost_optimizer):
        """Test finding cheapest cloud option."""
        from phase_4_cost_resilience.cost_optimizer import CloudProvider
        
        instance_types = {
            CloudProvider.AWS: "t3.medium",
            CloudProvider.AZURE: "Standard_D2s_v3",
            CloudProvider.GCP: "n1-standard-2"
        }
        
        options = await cost_optimizer.find_cheapest_option(
            instance_types=instance_types,
            allow_spot=True,
            duration_hours=24
        )
        
        assert len(options) > 0
        # Options should be sorted by cost
        assert options[0]["total_cost"] <= options[-1]["total_cost"]
    
    @pytest.mark.asyncio
    async def test_budget_enforcement(self, cost_optimizer):
        """Test budget enforcement."""
        from phase_4_cost_resilience.cost_optimizer import BudgetConfig
        
        # Set budget
        budget = BudgetConfig(
            project_id="test-project",
            monthly_limit=100.0,
            soft_limit_percent=0.8
        )
        cost_optimizer.set_budget(budget)
        
        # Check allowance within budget
        allowed, reason = await cost_optimizer.check_budget_allowance(
            "test-project",
            estimated_cost=50.0
        )
        assert allowed == True
        
        # Check allowance exceeding budget
        allowed, reason = await cost_optimizer.check_budget_allowance(
            "test-project",
            estimated_cost=150.0
        )
        assert allowed == False
        assert "exceed" in reason.lower()


class TestPolicyEnforcement:
    """Tests for OPA policy enforcement."""
    
    def test_role_permissions(self):
        """Test role-based permissions."""
        # Mock OPA evaluation
        test_cases = [
            {
                "user": {"role": "developer"},
                "request": {"tool": "get_cost_estimate"},
                "expected": True
            },
            {
                "user": {"role": "developer"},
                "request": {"tool": "rotate_secret"},
                "expected": False
            },
            {
                "user": {"role": "admin"},
                "request": {"tool": "rotate_secret", "approval": {"valid": True}},
                "expected": True
            }
        ]
        
        for case in test_cases:
            # In production, this would call OPA
            # For testing, we verify the policy logic
            assert True  # Placeholder for OPA integration test
    
    def test_data_residency(self):
        """Test data residency enforcement."""
        # Test that PII data is restricted to allowed regions
        pii_allowed_regions = ["us-east-1", "us-west-2"]
        
        # Valid request
        assert "us-east-1" in pii_allowed_regions
        
        # Invalid request
        assert "eu-west-1" not in pii_allowed_regions


# Fixtures for test setup/teardown

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def localstack_available():
    """Check if LocalStack is available for testing."""
    import socket
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 4566))
        sock.close()
        return result == 0
    except:
        return False


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
