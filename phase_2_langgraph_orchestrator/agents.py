"""
LangGraph Agent Implementations
Contains the Planner, Allocator, Executor, Verifier, and Finisher agents.
"""

import asyncio
import json
import socket
from datetime import datetime
from typing import Any, Dict, List, Optional

# ============== IPv4 Workaround ==============
# Force IPv4 to avoid DNS/IPv6 issues with aiohttp
_original_getaddrinfo = socket.getaddrinfo

def _getaddrinfo_ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
    return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

socket.getaddrinfo = _getaddrinfo_ipv4_only
# ============================================

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
import structlog

from .state import (
    OrchestratorState,
    WorkflowStatus,
    DecisionType,
    WorkloadType,
    CloudProvider,
)

logger = structlog.get_logger(__name__)


# ============== MCP Client Interface ==============

class MCPClientManager:
    """Manages connections to MCP servers for different clouds."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.clients: Dict[str, Any] = {}
    
    async def get_client(self, cloud: CloudProvider):
        """Get or create MCP client for a cloud provider."""
        if cloud.value not in self.clients:
            # Initialize MCP client connection via SSE
            from mcp.client.sse import sse_client
            
            # Get endpoint from config (e.g. http://localhost:8001)
            endpoint = self.config.get(cloud.value, {}).get("endpoint")
            if not endpoint:
                logger.warning("mcp_endpoint_not_configured", cloud=cloud.value)
                return None
            
            logger.info("connecting_to_mcp_server", cloud=cloud.value, endpoint=endpoint)
            
            try:
                # We need to maintain the context manager for the session
                # For simplicity in this manager, we'll store the session itself
                # In a production app, we'd manage the lifecycle more robustly
                self._cm = sse_client(url=f"{endpoint}/sse")
                read_stream, write_stream = await self._cm.__aenter__()
                
                from mcp import ClientSession
                session = ClientSession(read_stream, write_stream)
                await session.__aenter__()
                
                # Initialize session
                await session.initialize()
                
                self.clients[cloud.value] = session
                logger.info("mcp_client_connected", cloud=cloud.value)
                
            except Exception as e:
                logger.error("mcp_connection_failed", cloud=cloud.value, error=str(e))
                return None
                
        return self.clients.get(cloud.value)
    
    async def call_tool(
        self,
        cloud: CloudProvider,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a tool on a specific cloud's MCP server."""
        client = await self.get_client(cloud)
        if client:
            result = await client.call_tool(tool_name, arguments)
            # Parse the result content (assuming single text response containing JSON)
            if hasattr(result, 'content') and result.content:
                text_content = result.content[0].text
                try:
                    return json.loads(text_content)
                except json.JSONDecodeError:
                    return {"result": text_content}
            return {}
        else:
            # Fallback for testing without MCP connection
            logger.warning(
                "mcp_client_not_available",
                cloud=cloud.value,
                tool=tool_name
            )
            return {"status": "mock", "message": f"Mock response for {tool_name}"}


# ============== Agent Implementations ==============

class BaseAgent:
    """Base class for all agents."""
    
    def __init__(
        self,
        mcp_manager: MCPClientManager,
        llm: Optional[Any] = None
    ):
        self.mcp = mcp_manager
        self.llm = llm or ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    def _record_decision(
        self,
        state: OrchestratorState,
        decision_type: DecisionType,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        confidence: float,
        reasoning: str
    ) -> Dict[str, Any]:
        """Record a decision for audit trail."""
        decision = {
            "decision_id": f"dec-{datetime.utcnow().timestamp()}",
            "agent": self.__class__.__name__,
            "decision_type": decision_type.value,
            "input_data": input_data,
            "output_data": output_data,
            "confidence": confidence,
            "reasoning": reasoning,
            "timestamp": datetime.utcnow().isoformat()
        }
        return decision


class PlannerAgent(BaseAgent):
    """
    Planner Agent: Interprets tasks, classifies workloads, selects clouds/regions.
    """
    
    SYSTEM_PROMPT = """You are a cloud infrastructure planning expert. Your job is to:
1. Analyze the workload requirements
2. Classify the workload type (latency-sensitive, batch, model-serving, data-pipeline)
3. Select appropriate cloud providers and regions based on:
   - SLO requirements (latency, availability)
   - Cost constraints
   - Data residency requirements
   - Resource availability

Output your analysis as JSON with the following structure:
{
    "workload_classification": {
        "type": "latency_sensitive|batch|model_serving|data_pipeline",
        "resource_type": "cpu|gpu|tpu|memory_optimized",
        "estimated_compute_hours": number,
        "parallelizable": boolean
    },
    "selected_clouds": ["aws", "azure", "gcp"],
    "selected_regions": [
        {"cloud": "aws", "region": "us-east-1", "priority": 1, "reason": "..."}
    ],
    "compliance_status": {
        "data_residency": "compliant|non_compliant",
        "issues": []
    },
    "reasoning": "explanation of choices"
}"""
    
    async def __call__(self, state: OrchestratorState) -> OrchestratorState:
        """Execute the planning phase."""
        logger.info("planner_started", task_id=state["task_id"])
        
        state["status"] = WorkflowStatus.PLANNING.value
        state["current_agent"] = "PlannerAgent"
        
        task = state["task_request"]
        
        # Use LLM for intelligent planning
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=f"""
Analyze this task and provide planning decisions:

Task: {task.get('name')}
Description: {task.get('description')}
Workload Type Hint: {task.get('workload_type')}
Resource Requirements: {json.dumps(task.get('resource_requirements', {}))}
SLO Requirements: {json.dumps(task.get('slo_requirements', {}))}
Budget Limit: ${task.get('budget_limit', 'unlimited')}
Preferred Clouds: {task.get('preferred_clouds', ['any'])}
Preferred Regions: {task.get('preferred_regions', ['any'])}
Data Residency: {task.get('data_residency_requirements', ['none'])}
""")
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            plan = json.loads(response.content)
            
            # Update state with planning results
            state["workload_classification"] = plan.get("workload_classification", {})
            state["selected_clouds"] = plan.get("selected_clouds", ["aws"])
            state["selected_regions"] = plan.get("selected_regions", [])
            state["compliance_checks"] = [plan.get("compliance_status", {})]
            
            # Record decision
            decision = self._record_decision(
                state,
                DecisionType.CLOUD_SELECTION,
                {"task": task},
                plan,
                confidence=0.85,
                reasoning=plan.get("reasoning", "LLM-based planning")
            )
            state["decisions"] = state.get("decisions", []) + [decision]
            
            logger.info(
                "planner_completed",
                task_id=state["task_id"],
                clouds=state["selected_clouds"]
            )
            
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                logger.warning("planner_rate_limit_hit", detail="Gemini API Quota exceeded. Using safe defaults for demo.")
                state["errors"] = state.get("errors", []) + ["Planner fallback: Gemini API rate limit hit. Using default plan."]
            else:
                logger.error("planner_error", error=str(e))
                state["errors"] = state.get("errors", []) + [f"Planner error: {str(e)}"]
            
            # Fallback to defaults
            state["workload_classification"] = {
                "type": task.get("workload_type", "batch"),
                "resource_type": "cpu"
            }
            state["selected_clouds"] = task.get("preferred_clouds", ["aws"])
            state["selected_regions"] = [
                {"cloud": "aws", "region": "us-east-1", "priority": 1}
            ]
        
        return state


class AllocatorAgent(BaseAgent):
    """
    Allocator Agent: Queries costs/quotas, selects instance types, plans allocation.
    """
    
    async def __call__(self, state: OrchestratorState) -> OrchestratorState:
        """Execute the allocation phase."""
        logger.info("allocator_started", task_id=state["task_id"])
        
        state["status"] = WorkflowStatus.ALLOCATING.value
        state["current_agent"] = "AllocatorAgent"
        
        task = state["task_request"]
        selected_clouds = state["selected_clouds"]
        selected_regions = state["selected_regions"]
        workload = state["workload_classification"]
        
        cost_estimates = []
        quota_status = []
        
        # Query each selected cloud for costs and quotas
        for region_info in selected_regions:
            cloud = CloudProvider(region_info.get("cloud", "aws"))
            region = region_info.get("region", "us-east-1")
            
            # Determine instance type based on workload
            instance_type = self._select_instance_type(cloud, workload)
            
            # Get cost estimate from MCP
            cost_result = await self.mcp.call_tool(
                cloud,
                "get_cost_estimate",
                {
                    "resource_type": "compute",
                    "instance_type": instance_type,
                    "region": region,
                    "duration_hours": workload.get("estimated_compute_hours", 720),
                    "quantity": task.get("resource_requirements", {}).get("count", 1)
                }
            )
            cost_estimates.append({
                "cloud": cloud.value,
                "region": region,
                "instance_type": instance_type,
                **cost_result
            })
            
            # Get quota status from MCP
            quota_result = await self.mcp.call_tool(
                cloud,
                "get_quotas",
                {
                    "service": "ec2" if cloud == CloudProvider.AWS else "compute",
                    "region": region
                }
            )
            quota_status.append({
                "cloud": cloud.value,
                "region": region,
                **quota_result
            })
        
        state["cost_estimates"] = cost_estimates
        state["quota_status"] = quota_status
        
        # Create allocation plan
        allocation_plan = self._create_allocation_plan(
            task, workload, cost_estimates, quota_status
        )
        state["allocation_plan"] = allocation_plan
        
        # Record decision
        decision = self._record_decision(
            state,
            DecisionType.INSTANCE_SELECTION,
            {
                "workload": workload,
                "cost_estimates": cost_estimates,
                "quota_status": quota_status
            },
            allocation_plan,
            confidence=0.9,
            reasoning=f"Selected based on lowest cost with quota availability"
        )
        state["decisions"] = state.get("decisions", []) + [decision]
        
        logger.info(
            "allocator_completed",
            task_id=state["task_id"],
            estimated_cost=allocation_plan.get("total_estimated_cost")
        )
        
        return state
    
    def _select_instance_type(
        self,
        cloud: CloudProvider,
        workload: Dict[str, Any]
    ) -> str:
        """Select appropriate instance type based on workload and cloud."""
        resource_type = workload.get("resource_type", "cpu")
        
        instance_map = {
            CloudProvider.AWS: {
                "cpu": "t3.medium",
                "gpu": "p3.2xlarge",
                "memory_optimized": "r5.large"
            },
            CloudProvider.AZURE: {
                "cpu": "Standard_D2s_v3",
                "gpu": "Standard_NC6",
                "memory_optimized": "Standard_E2s_v3"
            },
            CloudProvider.GCP: {
                "cpu": "n1-standard-2",
                "gpu": "n1-standard-4-nvidia-tesla-t4",
                "memory_optimized": "n1-highmem-2"
            }
        }
        
        return instance_map.get(cloud, {}).get(resource_type, "t3.medium")
    
    def _create_allocation_plan(
        self,
        task: Dict[str, Any],
        workload: Dict[str, Any],
        cost_estimates: List[Dict],
        quota_status: List[Dict]
    ) -> Dict[str, Any]:
        """Create the final allocation plan."""
        # Sort by cost and pick the cheapest viable option
        viable_options = []
        
        for estimate in cost_estimates:
            # Check quota availability
            has_quota = True  # Simplified; would check actual quota
            
            if has_quota:
                viable_options.append(estimate)
        
        if not viable_options:
            viable_options = cost_estimates  # Fallback
        
        # Sort by estimated cost
        viable_options.sort(key=lambda x: x.get("estimated_cost_usd", float("inf")))
        
        best_option = viable_options[0] if viable_options else {}
        budget_limit = task.get("budget_limit")
        
        # Determine spot vs on-demand
        use_spot = (
            workload.get("type") == "batch" and
            (not budget_limit or best_option.get("estimated_cost_usd", 0) * 0.3 < budget_limit)
        )
        
        return {
            "primary": {
                "cloud": best_option.get("cloud", "aws"),
                "region": best_option.get("region", "us-east-1"),
                "instance_type": best_option.get("instance_type", "t3.medium"),
                "count": task.get("resource_requirements", {}).get("count", 1),
                "spot": use_spot
            },
            "fallback": viable_options[1] if len(viable_options) > 1 else None,
            "total_estimated_cost": best_option.get("estimated_cost_usd", 0),
            "use_spot": use_spot
        }


class ExecutorAgent(BaseAgent):
    """
    Executor Agent: Provisions resources, submits jobs, deploys models.
    """
    
    async def __call__(self, state: OrchestratorState) -> OrchestratorState:
        """Execute the provisioning phase."""
        logger.info("executor_started", task_id=state["task_id"])
        
        state["status"] = WorkflowStatus.EXECUTING.value
        state["current_agent"] = "ExecutorAgent"
        
        allocation = state["allocation_plan"]
        task = state["task_request"]
        workload = state["workload_classification"]
        
        primary = allocation.get("primary", {})
        cloud = CloudProvider(primary.get("cloud", "aws"))
        
        provisioned = []
        logs = []
        
        try:
            workload_type = workload.get("type", "batch")
            
            if workload_type in ["latency_sensitive", "model_serving"]:
                # Deploy model serving endpoint
                result = await self._deploy_model_serving(
                    cloud, primary, task
                )
                provisioned.append(result)
                logs.append(f"Deployed model serving endpoint: {result.get('endpoint_name')}")
                
            elif workload_type == "batch":
                # Submit batch training job
                result = await self._submit_batch_job(
                    cloud, primary, task
                )
                provisioned.append(result)
                logs.append(f"Submitted batch job: {result.get('job_name')}")
                
            else:
                # Provision compute instances
                result = await self._provision_compute(
                    cloud, primary, task
                )
                provisioned.extend(result.get("instances", []))
                logs.append(f"Provisioned {len(result.get('instances', []))} instances")
            
            state["provisioned_resources"] = provisioned
            state["execution_logs"] = logs
            
            logger.info(
                "executor_completed",
                task_id=state["task_id"],
                resources_count=len(provisioned)
            )
            
        except Exception as e:
            logger.error("executor_error", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Executor error: {str(e)}"]
            state["execution_logs"] = logs + [f"Error: {str(e)}"]
        
        return state
    
    async def _provision_compute(
        self,
        cloud: CloudProvider,
        config: Dict[str, Any],
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Provision compute instances."""
        tool_name = "launch_spot" if config.get("spot") else "provision_compute"
        
        return await self.mcp.call_tool(
            cloud,
            tool_name,
            {
                "instance_type": config.get("instance_type"),
                "region": config.get("region"),
                "count": config.get("count", 1),
                "tags": {
                    "task_id": task.get("task_id", "unknown"),
                    "name": task.get("name", "mcp-workload")
                },
                "spot": config.get("spot", False)
            }
        )
    
    async def _submit_batch_job(
        self,
        cloud: CloudProvider,
        config: Dict[str, Any],
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Submit ML training job."""
        return await self.mcp.call_tool(
            cloud,
            "submit_ml_job",
            {
                "job_name": f"job-{task.get('task_id', 'unknown')[:8]}",
                "instance_type": config.get("instance_type"),
                "instance_count": config.get("count", 1),
                "image_uri": task.get("resource_requirements", {}).get(
                    "image_uri",
                    "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.0-gpu-py38"
                ),
                "input_data_config": task.get("resource_requirements", {}).get(
                    "input_config",
                    {"ChannelName": "training", "DataSource": {"S3DataSource": {"S3Uri": "s3://bucket/data"}}}
                ),
                "output_path": task.get("resource_requirements", {}).get(
                    "output_path",
                    "s3://bucket/output"
                ),
                "spot_instances": config.get("spot", False)
            }
        )
    
    async def _deploy_model_serving(
        self,
        cloud: CloudProvider,
        config: Dict[str, Any],
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deploy model serving endpoint."""
        return await self.mcp.call_tool(
            cloud,
            "deploy_model",
            {
                "model_name": f"model-{task.get('task_id', 'unknown')[:8]}",
                "model_artifact_path": task.get("resource_requirements", {}).get(
                    "model_path",
                    "s3://bucket/model/model.tar.gz"
                ),
                "endpoint_name": f"endpoint-{task.get('task_id', 'unknown')[:8]}",
                "instance_type": config.get("instance_type"),
                "instance_count": config.get("count", 1),
                "auto_scaling": True
            }
        )


class VerifierAgent(BaseAgent):
    """
    Verifier Agent: Performs health checks, verifies SLOs, triggers rollbacks.
    """
    
    async def __call__(self, state: OrchestratorState) -> OrchestratorState:
        """Execute the verification phase."""
        logger.info("verifier_started", task_id=state["task_id"])
        
        state["status"] = WorkflowStatus.VERIFYING.value
        state["current_agent"] = "VerifierAgent"
        
        provisioned = state.get("provisioned_resources", [])
        task = state["task_request"]
        slo_requirements = task.get("slo_requirements", {})
        
        health_checks = []
        all_healthy = True
        
        # Perform health checks on provisioned resources
        for resource in provisioned:
            cloud = CloudProvider(resource.get("cloud", state["selected_clouds"][0]))
            
            result = await self.mcp.call_tool(
                cloud,
                "get_health",
                {
                    "resource_type": resource.get("resource_type", "compute"),
                    "resource_ids": [resource.get("resource_id", resource.get("instance_id", ""))],
                    "region": resource.get("region", "us-east-1")
                }
            )
            
            health_checks.append({
                "resource_id": resource.get("resource_id", resource.get("instance_id")),
                "cloud": cloud.value,
                **result
            })
            
            # Check if healthy
            checks = result.get("health_checks", [])
            for check in checks:
                if not check.get("healthy", False):
                    all_healthy = False
        
        state["health_checks"] = health_checks
        
        # Verify SLO compliance
        slo_compliance = self._check_slo_compliance(health_checks, slo_requirements)
        state["slo_compliance"] = slo_compliance
        
        # Determine if rollback is needed
        needs_rollback = not all_healthy or not slo_compliance.get("compliant", True)
        state["needs_rollback"] = needs_rollback
        
        if needs_rollback:
            state["rollback_reason"] = slo_compliance.get("violation_reason", "Health check failed")
            logger.warning(
                "verifier_rollback_needed",
                task_id=state["task_id"],
                reason=state["rollback_reason"]
            )
        else:
            logger.info(
                "verifier_completed",
                task_id=state["task_id"],
                all_healthy=all_healthy
            )
        
        return state
    
    def _check_slo_compliance(
        self,
        health_checks: List[Dict],
        slo_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if SLO requirements are met."""
        # Simplified SLO check
        max_latency = slo_requirements.get("max_latency_ms", 1000)
        min_availability = slo_requirements.get("min_availability", 0.99)
        
        compliant = True
        violations = []
        
        for check in health_checks:
            for hc in check.get("health_checks", []):
                if hc.get("latency_ms", 0) > max_latency:
                    compliant = False
                    violations.append(f"Latency {hc.get('latency_ms')}ms exceeds {max_latency}ms")
                
                if not hc.get("healthy", False):
                    compliant = False
                    violations.append(f"Resource {hc.get('resource_id')} is unhealthy")
        
        return {
            "compliant": compliant,
            "violations": violations,
            "violation_reason": "; ".join(violations) if violations else None
        }


class FinisherAgent(BaseAgent):
    """
    Finisher Agent: Cleanup, cost attribution, emit run records.
    """
    
    async def __call__(self, state: OrchestratorState) -> OrchestratorState:
        """Execute the finishing phase."""
        logger.info("finisher_started", task_id=state["task_id"])
        
        state["status"] = WorkflowStatus.FINISHING.value
        state["current_agent"] = "FinisherAgent"
        
        task = state["task_request"]
        provisioned = state.get("provisioned_resources", [])
        needs_rollback = state.get("needs_rollback", False)
        
        cleanup_actions = []
        
        if needs_rollback:
            # Perform rollback
            for resource in provisioned:
                cleanup_result = await self._cleanup_resource(resource)
                cleanup_actions.append({
                    "action": "rollback",
                    "resource_id": resource.get("resource_id", resource.get("instance_id")),
                    **cleanup_result
                })
            
            state["status"] = WorkflowStatus.ROLLED_BACK.value
        else:
            state["status"] = WorkflowStatus.COMPLETED.value
        
        state["cleanup_actions"] = cleanup_actions
        
        # Calculate cost attribution
        cost_attribution = self._calculate_cost_attribution(
            task,
            provisioned,
            state.get("allocation_plan", {})
        )
        state["cost_attribution"] = cost_attribution
        
        # Create final result
        final_result = {
            "task_id": state["task_id"],
            "status": state["status"],
            "provisioned_resources": provisioned,
            "total_cost": cost_attribution.get("total_cost", 0),
            "duration_seconds": self._calculate_duration(state),
            "decisions_count": len(state.get("decisions", [])),
            "errors": state.get("errors", []),
            "completed_at": datetime.utcnow().isoformat()
        }
        state["final_result"] = final_result
        
        logger.info(
            "finisher_completed",
            task_id=state["task_id"],
            status=state["status"],
            total_cost=cost_attribution.get("total_cost", 0)
        )
        
        return state
    
    async def _cleanup_resource(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """Cleanup a provisioned resource."""
        # In production, would call MCP to terminate/cleanup
        return {
            "status": "cleaned_up",
            "resource_id": resource.get("resource_id", resource.get("instance_id"))
        }
    
    def _calculate_cost_attribution(
        self,
        task: Dict[str, Any],
        provisioned: List[Dict],
        allocation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate cost attribution for the task."""
        estimated = allocation.get("total_estimated_cost", 0)
        
        return {
            "task_id": task.get("task_id"),
            "task_name": task.get("name"),
            "total_cost": estimated,
            "cost_by_cloud": {
                allocation.get("primary", {}).get("cloud", "aws"): estimated
            },
            "cost_by_resource_type": {
                "compute": estimated
            }
        }
    
    def _calculate_duration(self, state: OrchestratorState) -> float:
        """Calculate total workflow duration."""
        started = datetime.fromisoformat(state["started_at"])
        return (datetime.utcnow() - started).total_seconds()
