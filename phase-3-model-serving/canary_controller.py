"""
Canary Deployment Controller
Manages progressive traffic shifting and automated rollback for model deployments.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect

logger = structlog.get_logger(__name__)


class DeploymentStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PROMOTING = "promoting"
    ROLLING_BACK = "rolling_back"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class CanaryMetrics:
    """Metrics for canary analysis."""
    latency_p50: float
    latency_p95: float
    latency_p99: float
    error_rate: float
    request_count: int
    success_rate: float
    timestamp: datetime


@dataclass
class CanaryConfig:
    """Configuration for canary deployment."""
    name: str
    namespace: str
    baseline_service: str
    canary_service: str
    
    # Traffic management
    initial_traffic_percent: int = 5
    max_traffic_percent: int = 50
    traffic_step_percent: int = 10
    step_interval_minutes: int = 5
    
    # Success criteria
    max_latency_p95_ms: float = 100.0
    max_error_rate: float = 0.01
    min_request_count: int = 100
    
    # Rollback settings
    consecutive_failures_for_rollback: int = 3
    analysis_interval_seconds: int = 60
    
    # Promotion criteria
    min_canary_duration_minutes: int = 30


class CanaryController:
    """
    Controller for managing canary deployments.
    
    Features:
    - Progressive traffic shifting
    - Automated metric analysis
    - Automatic rollback on degradation
    - Full promotion on success
    """
    
    def __init__(
        self,
        prometheus_url: str = "http://prometheus:9090",
        kubeconfig_path: Optional[str] = None
    ):
        # Initialize Kubernetes client
        if kubeconfig_path:
            config.load_kube_config(kubeconfig_path)
        else:
            try:
                config.load_incluster_config()
            except:
                config.load_kube_config()
        
        self.k8s_custom = client.CustomObjectsApi()
        self.k8s_apps = client.AppsV1Api()
        
        # Initialize Prometheus client
        self.prometheus = PrometheusConnect(url=prometheus_url)
        
        # Track deployment state
        self.deployments: Dict[str, Dict[str, Any]] = {}
    
    async def start_canary(self, config: CanaryConfig) -> str:
        """
        Start a new canary deployment.
        
        Args:
            config: Canary deployment configuration
            
        Returns:
            Deployment ID
        """
        deployment_id = f"canary-{config.name}-{datetime.utcnow().timestamp():.0f}"
        
        self.deployments[deployment_id] = {
            "config": config,
            "status": DeploymentStatus.PENDING,
            "current_traffic_percent": 0,
            "started_at": datetime.utcnow(),
            "metrics_history": [],
            "consecutive_failures": 0
        }
        
        logger.info(
            "canary_started",
            deployment_id=deployment_id,
            baseline=config.baseline_service,
            canary=config.canary_service
        )
        
        # Start the canary process
        asyncio.create_task(self._run_canary(deployment_id))
        
        return deployment_id
    
    async def _run_canary(self, deployment_id: str):
        """Run the canary deployment process."""
        deployment = self.deployments[deployment_id]
        config = deployment["config"]
        
        try:
            deployment["status"] = DeploymentStatus.IN_PROGRESS
            
            # Initial traffic shift
            await self._set_traffic_split(
                config.namespace,
                config.baseline_service,
                config.canary_service,
                config.initial_traffic_percent
            )
            deployment["current_traffic_percent"] = config.initial_traffic_percent
            
            # Progressive analysis and traffic shifting
            while deployment["current_traffic_percent"] < config.max_traffic_percent:
                # Wait for analysis interval
                await asyncio.sleep(config.analysis_interval_seconds)
                
                # Collect and analyze metrics
                metrics = await self._collect_metrics(config)
                deployment["metrics_history"].append(metrics)
                
                # Check success criteria
                analysis = self._analyze_metrics(config, metrics)
                
                if analysis["passed"]:
                    deployment["consecutive_failures"] = 0
                    
                    # Increase traffic if not at max
                    if deployment["current_traffic_percent"] < config.max_traffic_percent:
                        new_traffic = min(
                            deployment["current_traffic_percent"] + config.traffic_step_percent,
                            config.max_traffic_percent
                        )
                        
                        await self._set_traffic_split(
                            config.namespace,
                            config.baseline_service,
                            config.canary_service,
                            new_traffic
                        )
                        deployment["current_traffic_percent"] = new_traffic
                        
                        logger.info(
                            "canary_traffic_increased",
                            deployment_id=deployment_id,
                            traffic_percent=new_traffic
                        )
                else:
                    deployment["consecutive_failures"] += 1
                    logger.warning(
                        "canary_analysis_failed",
                        deployment_id=deployment_id,
                        failures=deployment["consecutive_failures"],
                        reasons=analysis["reasons"]
                    )
                    
                    # Check if rollback needed
                    if deployment["consecutive_failures"] >= config.consecutive_failures_for_rollback:
                        await self._rollback(deployment_id)
                        return
            
            # Check if ready for promotion
            elapsed = datetime.utcnow() - deployment["started_at"]
            if elapsed >= timedelta(minutes=config.min_canary_duration_minutes):
                await self._promote(deployment_id)
            else:
                # Continue monitoring until min duration
                remaining = config.min_canary_duration_minutes - elapsed.total_seconds() / 60
                logger.info(
                    "canary_waiting_for_promotion",
                    deployment_id=deployment_id,
                    remaining_minutes=remaining
                )
                await asyncio.sleep(remaining * 60)
                await self._promote(deployment_id)
                
        except Exception as e:
            logger.error("canary_error", deployment_id=deployment_id, error=str(e))
            deployment["status"] = DeploymentStatus.FAILED
            await self._rollback(deployment_id)
    
    async def _collect_metrics(self, config: CanaryConfig) -> CanaryMetrics:
        """Collect metrics from Prometheus for canary analysis."""
        interval = f"{config.analysis_interval_seconds}s"
        
        # Query latency percentiles
        latency_p50 = self._query_metric(
            f'histogram_quantile(0.50, rate(inference_latency_bucket{{service="{config.canary_service}"}}[{interval}]))'
        )
        latency_p95 = self._query_metric(
            f'histogram_quantile(0.95, rate(inference_latency_bucket{{service="{config.canary_service}"}}[{interval}]))'
        )
        latency_p99 = self._query_metric(
            f'histogram_quantile(0.99, rate(inference_latency_bucket{{service="{config.canary_service}"}}[{interval}]))'
        )
        
        # Query error rate
        error_rate = self._query_metric(
            f'sum(rate(inference_errors_total{{service="{config.canary_service}"}}[{interval}])) / '
            f'sum(rate(inference_requests_total{{service="{config.canary_service}"}}[{interval}]))'
        )
        
        # Query request count
        request_count = self._query_metric(
            f'sum(increase(inference_requests_total{{service="{config.canary_service}"}}[{interval}]))'
        )
        
        return CanaryMetrics(
            latency_p50=latency_p50 * 1000,  # Convert to ms
            latency_p95=latency_p95 * 1000,
            latency_p99=latency_p99 * 1000,
            error_rate=error_rate or 0,
            request_count=int(request_count or 0),
            success_rate=1 - (error_rate or 0),
            timestamp=datetime.utcnow()
        )
    
    def _query_metric(self, query: str) -> float:
        """Query a single metric from Prometheus."""
        try:
            result = self.prometheus.custom_query(query)
            if result and len(result) > 0:
                return float(result[0]["value"][1])
        except Exception as e:
            logger.warning("prometheus_query_failed", query=query, error=str(e))
        return 0.0
    
    def _analyze_metrics(
        self,
        config: CanaryConfig,
        metrics: CanaryMetrics
    ) -> Dict[str, Any]:
        """Analyze metrics against success criteria."""
        passed = True
        reasons = []
        
        # Check latency
        if metrics.latency_p95 > config.max_latency_p95_ms:
            passed = False
            reasons.append(f"P95 latency {metrics.latency_p95:.1f}ms exceeds {config.max_latency_p95_ms}ms")
        
        # Check error rate
        if metrics.error_rate > config.max_error_rate:
            passed = False
            reasons.append(f"Error rate {metrics.error_rate:.2%} exceeds {config.max_error_rate:.2%}")
        
        # Check minimum traffic
        if metrics.request_count < config.min_request_count:
            passed = False
            reasons.append(f"Request count {metrics.request_count} below minimum {config.min_request_count}")
        
        return {
            "passed": passed,
            "reasons": reasons,
            "metrics": metrics.__dict__
        }
    
    async def _set_traffic_split(
        self,
        namespace: str,
        baseline: str,
        canary: str,
        canary_percent: int
    ):
        """Set traffic split between baseline and canary."""
        # Update VirtualService for Istio traffic management
        virtual_service = {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "VirtualService",
            "metadata": {
                "name": f"{baseline}-vs",
                "namespace": namespace
            },
            "spec": {
                "hosts": [baseline],
                "http": [{
                    "route": [
                        {
                            "destination": {"host": baseline},
                            "weight": 100 - canary_percent
                        },
                        {
                            "destination": {"host": canary},
                            "weight": canary_percent
                        }
                    ]
                }]
            }
        }
        
        try:
            self.k8s_custom.patch_namespaced_custom_object(
                group="networking.istio.io",
                version="v1beta1",
                namespace=namespace,
                plural="virtualservices",
                name=f"{baseline}-vs",
                body=virtual_service
            )
        except client.exceptions.ApiException as e:
            if e.status == 404:
                self.k8s_custom.create_namespaced_custom_object(
                    group="networking.istio.io",
                    version="v1beta1",
                    namespace=namespace,
                    plural="virtualservices",
                    body=virtual_service
                )
            else:
                raise
    
    async def _promote(self, deployment_id: str):
        """Promote canary to production."""
        deployment = self.deployments[deployment_id]
        config = deployment["config"]
        
        deployment["status"] = DeploymentStatus.PROMOTING
        
        logger.info(
            "canary_promoting",
            deployment_id=deployment_id,
            canary=config.canary_service
        )
        
        # Shift all traffic to canary
        await self._set_traffic_split(
            config.namespace,
            config.baseline_service,
            config.canary_service,
            100
        )
        
        # Update baseline to use canary image
        # (In production, this would update the baseline deployment)
        
        deployment["status"] = DeploymentStatus.COMPLETED
        deployment["completed_at"] = datetime.utcnow()
        
        logger.info(
            "canary_promoted",
            deployment_id=deployment_id,
            duration_minutes=(deployment["completed_at"] - deployment["started_at"]).total_seconds() / 60
        )
    
    async def _rollback(self, deployment_id: str):
        """Rollback canary deployment."""
        deployment = self.deployments[deployment_id]
        config = deployment["config"]
        
        deployment["status"] = DeploymentStatus.ROLLING_BACK
        
        logger.warning(
            "canary_rolling_back",
            deployment_id=deployment_id,
            reason="Metrics degradation detected"
        )
        
        # Shift all traffic back to baseline
        await self._set_traffic_split(
            config.namespace,
            config.baseline_service,
            config.canary_service,
            0
        )
        
        deployment["status"] = DeploymentStatus.ROLLED_BACK
        deployment["completed_at"] = datetime.utcnow()
        
        logger.info(
            "canary_rolled_back",
            deployment_id=deployment_id
        )
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a canary deployment."""
        if deployment_id not in self.deployments:
            return None
        
        deployment = self.deployments[deployment_id]
        return {
            "deployment_id": deployment_id,
            "status": deployment["status"].value,
            "current_traffic_percent": deployment["current_traffic_percent"],
            "started_at": deployment["started_at"].isoformat(),
            "metrics_count": len(deployment["metrics_history"]),
            "consecutive_failures": deployment["consecutive_failures"]
        }


# Example usage
async def main():
    controller = CanaryController()
    
    config = CanaryConfig(
        name="model-v2",
        namespace="model-serving",
        baseline_service="cloud-orchestrator-model",
        canary_service="cloud-orchestrator-model-canary",
        initial_traffic_percent=5,
        max_traffic_percent=50,
        traffic_step_percent=10,
        max_latency_p95_ms=100.0,
        max_error_rate=0.01
    )
    
    deployment_id = await controller.start_canary(config)
    print(f"Started canary deployment: {deployment_id}")
    
    # Monitor status
    while True:
        status = controller.get_deployment_status(deployment_id)
        print(f"Status: {status}")
        if status["status"] in ["completed", "rolled_back", "failed"]:
            break
        await asyncio.sleep(30)


if __name__ == "__main__":
    asyncio.run(main())
