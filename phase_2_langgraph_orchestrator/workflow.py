"""
LangGraph Workflow Orchestrator
Defines and executes the main orchestration workflow graph.
"""

import asyncio
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import structlog

from .state import (
    OrchestratorState,
    WorkflowStatus,
    TaskRequest,
    create_initial_state,
    InMemoryStateStore,
    PostgresStateStore,
)
from .agents import (
    MCPClientManager,
    PlannerAgent,
    AllocatorAgent,
    ExecutorAgent,
    VerifierAgent,
    FinisherAgent,
)

logger = structlog.get_logger(__name__)


def should_rollback(state: OrchestratorState) -> str:
    """Determine if workflow should proceed to finisher or handle rollback."""
    if state.get("needs_rollback", False):
        return "finisher"  # Finisher handles rollback
    return "finisher"


def should_retry(state: OrchestratorState) -> str:
    """Determine if workflow should retry on error."""
    errors = state.get("errors", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    
    if errors and retry_count < max_retries:
        return "planner"  # Retry from beginning
    elif errors:
        return "finisher"  # Too many retries, cleanup and exit
    return "finisher"


class CloudOrchestrator:
    """
    Main orchestrator that builds and runs the LangGraph workflow.
    
    Workflow:
    Planner → Allocator → Executor → Verifier → Finisher
                                        ↓
                                   [Rollback if needed]
    """
    
    def __init__(
        self,
        mcp_config: Optional[Dict[str, Any]] = None,
        database_url: Optional[str] = None,
    ):
        self.mcp_config = mcp_config or {}
        self.mcp_manager = MCPClientManager(self.mcp_config)
        
        # State persistence
        if database_url:
            self.state_store = PostgresStateStore(database_url)
        else:
            self.state_store = InMemoryStateStore()
        
        # Initialize agents
        self.planner = PlannerAgent(self.mcp_manager)
        self.allocator = AllocatorAgent(self.mcp_manager)
        self.executor = ExecutorAgent(self.mcp_manager)
        self.verifier = VerifierAgent(self.mcp_manager)
        self.finisher = FinisherAgent(self.mcp_manager)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        # Memory for checkpointing
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(OrchestratorState)
        
        # Add nodes (agents)
        workflow.add_node("planner", self.planner)
        workflow.add_node("allocator", self.allocator)
        workflow.add_node("executor", self.executor)
        workflow.add_node("verifier", self.verifier)
        workflow.add_node("finisher", self.finisher)
        
        # Define edges (flow)
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "allocator")
        workflow.add_edge("allocator", "executor")
        workflow.add_edge("executor", "verifier")
        
        # Conditional edge from verifier
        workflow.add_conditional_edges(
            "verifier",
            should_rollback,
            {
                "finisher": "finisher"
            }
        )
        
        # End after finisher
        workflow.add_edge("finisher", END)
        
        return workflow
    
    async def run(self, task_request: TaskRequest) -> Dict[str, Any]:
        """
        Run the orchestration workflow for a task.
        
        Args:
            task_request: The task to execute
            
        Returns:
            Final result of the workflow
        """
        logger.info(
            "workflow_started",
            task_id=task_request.task_id,
            task_name=task_request.name
        )
        
        # Create initial state
        initial_state = create_initial_state(task_request)
        
        # Save initial state
        await self.state_store.save_state(task_request.task_id, initial_state)
        
        try:
            # Run the workflow
            config = {"configurable": {"thread_id": task_request.task_id}}
            final_state = await self.app.ainvoke(initial_state, config)
            
            # Save final state
            await self.state_store.save_state(task_request.task_id, final_state)
            
            logger.info(
                "workflow_completed",
                task_id=task_request.task_id,
                status=final_state.get("status")
            )
            
            return final_state.get("final_result", {})
            
        except Exception as e:
            logger.error(
                "workflow_error",
                task_id=task_request.task_id,
                error=str(e)
            )
            
            # Update state with error
            error_state = initial_state.copy()
            error_state["status"] = WorkflowStatus.FAILED.value
            error_state["errors"] = [str(e)]
            await self.state_store.save_state(task_request.task_id, error_state)
            
            return {
                "task_id": task_request.task_id,
                "status": WorkflowStatus.FAILED.value,
                "errors": [str(e)]
            }
    
    async def get_workflow_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a workflow."""
        state = await self.state_store.load_state(task_id)
        if state:
            return {
                "task_id": task_id,
                "status": state.get("status"),
                "current_agent": state.get("current_agent"),
                "errors": state.get("errors", []),
                "decisions_count": len(state.get("decisions", []))
            }
        return None
    
    async def get_workflow_decisions(self, task_id: str) -> list:
        """Get all decisions made during a workflow."""
        state = await self.state_store.load_state(task_id)
        if state:
            return state.get("decisions", [])
        return []
    
    async def cancel_workflow(self, task_id: str) -> bool:
        """Cancel a running workflow."""
        state = await self.state_store.load_state(task_id)
        if state and state.get("status") not in [
            WorkflowStatus.COMPLETED.value,
            WorkflowStatus.FAILED.value
        ]:
            state["status"] = WorkflowStatus.FAILED.value
            state["errors"] = state.get("errors", []) + ["Workflow cancelled by user"]
            await self.state_store.save_state(task_id, state)
            return True
        return False


# ============== Reference Workflow Implementations ==============

async def run_latency_service_deploy(
    orchestrator: CloudOrchestrator,
    model_path: str,
    endpoint_name: str,
    instance_type: str = "ml.m5.large",
    slo_latency_ms: int = 100,
    budget_limit: float = 500.0
) -> Dict[str, Any]:
    """
    Reference workflow for deploying a latency-sensitive model serving endpoint.
    """
    task = TaskRequest(
        name=f"Deploy {endpoint_name}",
        description="Deploy model serving endpoint with low latency requirements",
        workload_type="latency_sensitive",
        resource_requirements={
            "model_path": model_path,
            "instance_type": instance_type,
            "count": 2,  # Multi-AZ for availability
            "auto_scaling": True,
            "min_instances": 1,
            "max_instances": 10
        },
        slo_requirements={
            "max_latency_ms": slo_latency_ms,
            "min_availability": 0.999,
            "max_error_rate": 0.01
        },
        budget_limit=budget_limit,
        preferred_clouds=["aws", "gcp"],  # Best for ML serving
        priority=8
    )
    
    return await orchestrator.run(task)


async def run_batch_job_submit(
    orchestrator: CloudOrchestrator,
    job_name: str,
    image_uri: str,
    input_data_path: str,
    output_path: str,
    instance_type: str = "ml.p3.2xlarge",
    instance_count: int = 1,
    max_runtime_hours: int = 24,
    budget_limit: float = 200.0
) -> Dict[str, Any]:
    """
    Reference workflow for submitting a batch ML training job.
    """
    task = TaskRequest(
        name=f"Batch Job: {job_name}",
        description="ML training batch job with checkpointing",
        workload_type="batch",
        resource_requirements={
            "image_uri": image_uri,
            "input_config": {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3Uri": input_data_path,
                        "S3DataType": "S3Prefix"
                    }
                }
            },
            "output_path": output_path,
            "instance_type": instance_type,
            "count": instance_count,
            "max_runtime_seconds": max_runtime_hours * 3600,
            "checkpoint_enabled": True
        },
        slo_requirements={
            "max_completion_time_hours": max_runtime_hours,
            "checkpoint_interval_minutes": 30
        },
        budget_limit=budget_limit,
        preferred_clouds=["aws", "gcp", "azure"],
        priority=5
    )
    
    return await orchestrator.run(task)


# ============== CLI Entry Point ==============

async def main():
    """Main entry point for the orchestrator."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize orchestrator (using in-memory state store, no database required)
    orchestrator = CloudOrchestrator(
        mcp_config={
            "aws": {
                "endpoint": os.getenv("MCP_AWS_ENDPOINT", "http://localhost:8000")
            },
            "azure": {
                "endpoint": os.getenv("MCP_AZURE_ENDPOINT", "http://localhost:8002")
            },
            "gcp": {
                "endpoint": os.getenv("MCP_GCP_ENDPOINT", "http://localhost:8003")
            }
        },
        database_url=None  # Use in-memory state store
    )
    
    # Example: Run a batch job
    result = await run_batch_job_submit(
        orchestrator,
        job_name="example-training",
        image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.0-gpu-py38",
        input_data_path="s3://my-bucket/training-data/",
        output_path="s3://my-bucket/model-output/",
        budget_limit=100.0
    )
    
    print(f"Workflow completed: {result}")


if __name__ == "__main__":
    asyncio.run(main())
