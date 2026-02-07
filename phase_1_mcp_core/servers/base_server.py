"""
Base MCP Server Implementation
Provides common functionality for all cloud-specific MCP servers.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import structlog
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent, CallToolResult
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from pydantic import BaseModel, Field

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)


class CloudProvider(str, Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"


class OperationStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class OperationResult:
    """Result of an MCP tool operation."""
    success: bool
    operation_id: str
    status: OperationStatus
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: Optional[float] = None


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_requests: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        self.failures = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        async with self._lock:
            if self.state == "open":
                if self._should_attempt_reset():
                    self.state = "half-open"
                else:
                    raise CircuitBreakerOpenError("Circuit breaker is open")
            
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        if self.last_failure_time is None:
            return True
        elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout
    
    async def _on_success(self):
        async with self._lock:
            self.failures = 0
            self.state = "closed"
    
    async def _on_failure(self):
        async with self._lock:
            self.failures += 1
            self.last_failure_time = datetime.utcnow()
            if self.failures >= self.failure_threshold:
                self.state = "open"


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = datetime.utcnow()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        async with self._lock:
            now = datetime.utcnow()
            elapsed = (now - self.last_update).total_seconds()
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    async def wait_for_token(self, tokens: int = 1):
        while not await self.acquire(tokens):
            await asyncio.sleep(1 / self.rate)


# Tool Parameter Models
class ProvisionComputeParams(BaseModel):
    """Parameters for provisioning compute resources."""
    instance_type: str = Field(..., description="Instance type (e.g., t3.medium, Standard_D2s_v3)")
    region: str = Field(..., description="Cloud region")
    count: int = Field(default=1, ge=1, le=100)
    tags: Dict[str, str] = Field(default_factory=dict)
    subnet_id: Optional[str] = None
    security_groups: List[str] = Field(default_factory=list)
    user_data: Optional[str] = None
    spot: bool = Field(default=False, description="Use spot/preemptible instances")
    max_price: Optional[float] = Field(default=None, description="Max price for spot instances")


class ScaleNodepoolParams(BaseModel):
    """Parameters for scaling Kubernetes node pools."""
    cluster_name: str
    nodepool_name: str
    desired_count: int = Field(ge=0, le=1000)
    min_count: Optional[int] = Field(default=None, ge=0)
    max_count: Optional[int] = Field(default=None, ge=1)
    region: str


class CreateStorageBucketParams(BaseModel):
    """Parameters for creating storage buckets."""
    bucket_name: str = Field(..., min_length=3, max_length=63)
    region: str
    versioning: bool = Field(default=True)
    encryption: bool = Field(default=True)
    public_access: bool = Field(default=False)
    lifecycle_rules: Optional[List[Dict[str, Any]]] = None


class SubmitMLJobParams(BaseModel):
    """Parameters for submitting ML training jobs."""
    job_name: str
    instance_type: str
    instance_count: int = Field(default=1, ge=1)
    image_uri: str
    hyperparameters: Dict[str, str] = Field(default_factory=dict)
    input_data_config: Dict[str, Any]
    output_path: str
    max_runtime_seconds: int = Field(default=86400)  # 24 hours
    spot_instances: bool = Field(default=False)


class DeployModelParams(BaseModel):
    """Parameters for deploying ML models."""
    model_name: str
    model_artifact_path: str
    endpoint_name: str
    instance_type: str
    instance_count: int = Field(default=1, ge=1)
    auto_scaling: bool = Field(default=True)
    min_instances: int = Field(default=1)
    max_instances: int = Field(default=10)


class GetCostEstimateParams(BaseModel):
    """Parameters for getting cost estimates."""
    resource_type: str
    instance_type: Optional[str] = None
    region: str
    duration_hours: int = Field(default=720)  # 30 days
    quantity: int = Field(default=1)


class GetQuotasParams(BaseModel):
    """Parameters for getting resource quotas."""
    service: str
    region: str
    quota_codes: Optional[List[str]] = None


class RotateSecretParams(BaseModel):
    """Parameters for rotating secrets."""
    secret_id: str
    rotation_lambda_arn: Optional[str] = None
    rotation_rules: Optional[Dict[str, Any]] = None


class GetHealthParams(BaseModel):
    """Parameters for health checks."""
    resource_type: str
    resource_ids: List[str]
    region: str


class FailoverRouteParams(BaseModel):
    """Parameters for configuring failover routing."""
    route_name: str
    primary_target: str
    secondary_target: str
    health_check_id: str
    failover_threshold: int = Field(default=3)


class BaseMCPServer(ABC):
    """
    Abstract base class for cloud-specific MCP servers.
    Provides common functionality and defines the interface for cloud operations.
    """
    
    def __init__(
        self,
        provider: CloudProvider,
        server_name: str,
        rate_limit: float = 10.0,
        rate_capacity: int = 100
    ):
        self.provider = provider
        self.server_name = server_name
        self.server = Server(server_name)
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter(rate_limit, rate_capacity)
        self.idempotency_store: Dict[str, OperationResult] = {}
        self.mock_mode = self._check_mock_mode()
        
        if self.mock_mode:
            logger.info("mcp_server_mock_mode_active", provider=self.provider.value)
        
        self._register_tools()
        self._setup_handlers()
    
    def _check_mock_mode(self) -> bool:
        """Check if mock mode should be enabled based on environment and credentials."""
        import os
        if os.getenv('MOCK_MODE', 'false').lower() == 'true':
            return True
        return False
    
    def _register_tools(self):
        """Register all MCP tools."""
        self.tools = [
            Tool(
                name="provision_compute",
                description="Provision compute instances (EC2/VM/GCE)",
                inputSchema=ProvisionComputeParams.model_json_schema()
            ),
            Tool(
                name="scale_nodepool",
                description="Scale Kubernetes node pools",
                inputSchema=ScaleNodepoolParams.model_json_schema()
            ),
            Tool(
                name="launch_spot",
                description="Launch spot/preemptible instances",
                inputSchema=ProvisionComputeParams.model_json_schema()
            ),
            Tool(
                name="create_storage_bucket",
                description="Create object storage buckets",
                inputSchema=CreateStorageBucketParams.model_json_schema()
            ),
            Tool(
                name="submit_ml_job",
                description="Submit ML training jobs",
                inputSchema=SubmitMLJobParams.model_json_schema()
            ),
            Tool(
                name="deploy_model",
                description="Deploy ML models for inference",
                inputSchema=DeployModelParams.model_json_schema()
            ),
            Tool(
                name="get_cost_estimate",
                description="Get cost estimates for resources",
                inputSchema=GetCostEstimateParams.model_json_schema()
            ),
            Tool(
                name="get_quotas",
                description="Check resource quotas and limits",
                inputSchema=GetQuotasParams.model_json_schema()
            ),
            Tool(
                name="rotate_secret",
                description="Rotate secrets in secret manager",
                inputSchema=RotateSecretParams.model_json_schema()
            ),
            Tool(
                name="get_health",
                description="Health check for cloud resources",
                inputSchema=GetHealthParams.model_json_schema()
            ),
            Tool(
                name="failover_route",
                description="Configure failover routing",
                inputSchema=FailoverRouteParams.model_json_schema()
            ),
        ]
    
    def _setup_handlers(self):
        """Set up MCP server handlers."""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return self.tools
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            return await self._handle_tool_call(name, arguments)
    
    async def _handle_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> CallToolResult:
        """Handle a tool call with rate limiting, circuit breaker, and idempotency."""
        
        # Generate operation ID for idempotency
        idempotency_key = arguments.get("idempotency_key")
        if idempotency_key and idempotency_key in self.idempotency_store:
            cached = self.idempotency_store[idempotency_key]
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Cached result: {cached.data}"
                )]
            )
        
        operation_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        with tracer.start_as_current_span(f"mcp.{tool_name}") as span:
            span.set_attribute("cloud.provider", self.provider.value)
            span.set_attribute("operation.id", operation_id)
            span.set_attribute("tool.name", tool_name)
            
            try:
                # Rate limiting
                await self.rate_limiter.wait_for_token()
                
                # Execute with circuit breaker
                handler = self._get_tool_handler(tool_name)
                
                if self.mock_mode:
                    result = await self._get_mock_response(tool_name, arguments)
                else:
                    result = await self.circuit_breaker.call(handler, arguments)
                
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                operation_result = OperationResult(
                    success=True,
                    operation_id=operation_id,
                    status=OperationStatus.COMPLETED,
                    data=result,
                    duration_ms=duration_ms
                )
                
                if idempotency_key:
                    self.idempotency_store[idempotency_key] = operation_result
                
                span.set_status(Status(StatusCode.OK))
                logger.info(
                    "tool_call_success",
                    tool=tool_name,
                    operation_id=operation_id,
                    duration_ms=duration_ms
                )
                
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=json.dumps(result)
                    )]
                )
                
            except CircuitBreakerOpenError as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error("circuit_breaker_open", tool=tool_name)
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Service temporarily unavailable: {e}"
                    )],
                    isError=True
                )
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error("tool_call_error", tool=tool_name, error=str(e))
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error: {e}"
                    )],
                    isError=True
                )
    
    def _get_tool_handler(self, tool_name: str) -> Callable:
        """Get the handler for a specific tool."""
        handlers = {
            "provision_compute": self.provision_compute,
            "scale_nodepool": self.scale_nodepool,
            "launch_spot": self.launch_spot,
            "create_storage_bucket": self.create_storage_bucket,
            "submit_ml_job": self.submit_ml_job,
            "deploy_model": self.deploy_model,
            "get_cost_estimate": self.get_cost_estimate,
            "get_quotas": self.get_quotas,
            "rotate_secret": self.rotate_secret,
            "get_health": self.get_health,
            "failover_route": self.failover_route,
        }
        return handlers.get(tool_name, self._unknown_tool)
    
    async def _get_mock_response(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Return a believable mock response for a tool call."""
        logger.info("generating_mock_response", tool=tool_name)
        
        # Default success response
        response = {
            "status": "success",
            "message": f"Mock {tool_name} execution successful",
            "provider": self.provider.value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Tool-specific extras to keep orchestrator happy
        if "provision" in tool_name:
            response["instances"] = [{
                "instance_id": f"mock-inst-{uuid.uuid4().hex[:8]}",
                "instance_type": arguments.get("instance_type", "t3.medium"),
                "state": "running"
            }]
        elif "storage" in tool_name:
            response["bucket_name"] = arguments.get("bucket_name", "mock-bucket")
        elif "cost" in tool_name:
            response["estimated_cost_usd"] = 0.50
            response["currency"] = "USD"
            
        return response
    
    async def _unknown_tool(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        raise ValueError(f"Unknown tool")
    
    # Abstract methods to be implemented by cloud-specific servers
    @abstractmethod
    async def provision_compute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Provision compute instances."""
        pass
    
    @abstractmethod
    async def scale_nodepool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Scale Kubernetes node pools."""
        pass
    
    @abstractmethod
    async def launch_spot(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Launch spot/preemptible instances."""
        pass
    
    @abstractmethod
    async def create_storage_bucket(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create storage buckets."""
        pass
    
    @abstractmethod
    async def submit_ml_job(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Submit ML training jobs."""
        pass
    
    @abstractmethod
    async def deploy_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy ML models."""
        pass
    
    @abstractmethod
    async def get_cost_estimate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get cost estimates."""
        pass
    
    @abstractmethod
    async def get_quotas(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get resource quotas."""
        pass
    
    @abstractmethod
    async def rotate_secret(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Rotate secrets."""
        pass
    
    @abstractmethod
    async def get_health(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Health check resources."""
        pass
    
    @abstractmethod
    async def failover_route(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Configure failover routing."""
        pass
    
    async def run(self, transport: str = "sse", host: str = "0.0.0.0", port: int = 8000):
        """Run the MCP server."""
        if transport == "stdio":
            from mcp.server.stdio import stdio_server
            logger.info("mcp_server_starting_stdio", provider=self.provider.value)
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name=self.server_name,
                        server_version="1.0.0",
                        capabilities={}
                    )
                )
        else:
            from mcp.server.sse import SseServerTransport
            from starlette.applications import Starlette
            from starlette.routing import Route, Mount
            import uvicorn
            
            logger.info("mcp_server_starting_sse", provider=self.provider.value, host=host, port=port)
            
            sse = SseServerTransport("/messages")
            
            async def handle_sse(scope, receive, send):
                try:
                    async with sse.connect_sse(scope, receive, send) as (read_stream, write_stream):
                        await self.server.run(
                            read_stream,
                            write_stream,
                            InitializationOptions(
                                server_name=self.server_name,
                                server_version="1.0.0",
                                capabilities={}
                            )
                        )
                except Exception as e:
                    logger.error("sse_connection_error", error=str(e), exc_info=True)
                    raise
            
            async def handle_messages(scope, receive, send):
                try:
                    await sse.handle_post_message(scope, receive, send)
                except Exception as e:
                    logger.error("message_handling_error", error=str(e), exc_info=True)
                    raise
                
            async def app(scope, receive, send):
                if scope['type'] == 'lifespan':
                    while True:
                        message = await receive()
                        if message['type'] == 'lifespan.startup':
                            await send({'type': 'lifespan.startup.complete'})
                        elif message['type'] == 'lifespan.shutdown':
                            await send({'type': 'lifespan.shutdown.complete'})
                            return
                
                if scope['type'] == 'http':
                    path = scope['path']
                    method = scope['method']
                    
                    if method == 'GET' and path.rstrip('/') == '/sse':
                        await handle_sse(scope, receive, send)
                        return
                        
                    if method == 'POST' and (path.rstrip('/') == '/sse/messages' or path.rstrip('/') == '/messages'):
                        await handle_messages(scope, receive, send)
                        return
                        
                    if path == '/health':
                        await send({
                            'type': 'http.response.start',
                            'status': 200,
                            'headers': [[b'content-type', b'application/json']],
                        })
                        await send({
                            'type': 'http.response.body',
                            'body': b'{"status": "healthy"}',
                        })
                        return

                    await send({'type': 'http.response.start', 'status': 404, 'headers': []})
                    await send({'type': 'http.response.body', 'body': b'Not Found'})
            
            config = uvicorn.Config(app, host=host, port=port, log_level="info")
            server = uvicorn.Server(config)
            await server.serve()
