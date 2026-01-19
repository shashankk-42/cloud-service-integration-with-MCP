"""
Cost Optimizer Service
Implements cross-cloud arbitrage, budget management, and cost-aware scheduling.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class CloudProvider(str, Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"


class InstanceCategory(str, Enum):
    ON_DEMAND = "on_demand"
    SPOT = "spot"
    RESERVED = "reserved"
    PREEMPTIBLE = "preemptible"


@dataclass
class PricingInfo:
    """Pricing information for a resource type."""
    cloud: CloudProvider
    region: str
    instance_type: str
    on_demand_hourly: float
    spot_hourly: Optional[float] = None
    spot_availability: float = 1.0  # 0-1 probability
    egress_per_gb: float = 0.0
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BudgetConfig:
    """Budget configuration for a project/tenant."""
    project_id: str
    monthly_limit: float
    soft_limit_percent: float = 0.8
    hard_limit_percent: float = 1.0
    alert_emails: List[str] = field(default_factory=list)
    auto_shutdown_on_breach: bool = False


@dataclass
class CostRecord:
    """Record of incurred cost."""
    project_id: str
    cloud: CloudProvider
    region: str
    resource_type: str
    resource_id: str
    cost: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)


class CostOptimizer:
    """
    Cost optimization engine for multi-cloud resource allocation.
    
    Features:
    - Cross-cloud price comparison
    - Spot/preemptible optimization
    - Budget enforcement
    - Cost arbitrage scheduling
    """
    
    def __init__(self):
        # Pricing cache (would be populated from cloud APIs in production)
        self.pricing_cache: Dict[str, PricingInfo] = {}
        self._init_pricing_cache()
        
        # Budget tracking
        self.budgets: Dict[str, BudgetConfig] = {}
        self.cost_records: List[CostRecord] = []
        
        # Region latency data (ms from major locations)
        self.region_latency = {
            "aws": {
                "us-east-1": {"us": 20, "eu": 80, "asia": 180},
                "us-west-2": {"us": 40, "eu": 120, "asia": 150},
                "eu-west-1": {"us": 80, "eu": 20, "asia": 200},
                "ap-northeast-1": {"us": 150, "eu": 200, "asia": 30}
            },
            "azure": {
                "eastus": {"us": 20, "eu": 85, "asia": 175},
                "westeurope": {"us": 85, "eu": 15, "asia": 195},
                "southeastasia": {"us": 170, "eu": 190, "asia": 25}
            },
            "gcp": {
                "us-central1": {"us": 25, "eu": 90, "asia": 160},
                "europe-west1": {"us": 90, "eu": 18, "asia": 190},
                "asia-east1": {"us": 160, "eu": 185, "asia": 20}
            }
        }
    
    def _init_pricing_cache(self):
        """Initialize pricing cache with sample data."""
        # AWS pricing
        self._add_pricing(CloudProvider.AWS, "us-east-1", "t3.medium", 0.0416, 0.0125)
        self._add_pricing(CloudProvider.AWS, "us-east-1", "m5.large", 0.096, 0.035)
        self._add_pricing(CloudProvider.AWS, "us-east-1", "p3.2xlarge", 3.06, 0.918)
        self._add_pricing(CloudProvider.AWS, "us-west-2", "t3.medium", 0.0416, 0.0130)
        self._add_pricing(CloudProvider.AWS, "us-west-2", "m5.large", 0.096, 0.038)
        self._add_pricing(CloudProvider.AWS, "eu-west-1", "t3.medium", 0.0456, 0.0150)
        self._add_pricing(CloudProvider.AWS, "eu-west-1", "m5.large", 0.107, 0.042)
        
        # Azure pricing
        self._add_pricing(CloudProvider.AZURE, "eastus", "Standard_D2s_v3", 0.096, 0.029)
        self._add_pricing(CloudProvider.AZURE, "eastus", "Standard_D4s_v3", 0.192, 0.058)
        self._add_pricing(CloudProvider.AZURE, "westeurope", "Standard_D2s_v3", 0.108, 0.032)
        self._add_pricing(CloudProvider.AZURE, "southeastasia", "Standard_D2s_v3", 0.102, 0.030)
        
        # GCP pricing
        self._add_pricing(CloudProvider.GCP, "us-central1", "n1-standard-2", 0.095, 0.019, 0.95)
        self._add_pricing(CloudProvider.GCP, "us-central1", "n1-standard-4", 0.19, 0.038, 0.93)
        self._add_pricing(CloudProvider.GCP, "europe-west1", "n1-standard-2", 0.104, 0.021, 0.94)
        self._add_pricing(CloudProvider.GCP, "asia-east1", "n1-standard-2", 0.098, 0.020, 0.92)
    
    def _add_pricing(
        self,
        cloud: CloudProvider,
        region: str,
        instance_type: str,
        on_demand: float,
        spot: Optional[float] = None,
        spot_availability: float = 0.9
    ):
        """Add pricing info to cache."""
        key = f"{cloud.value}:{region}:{instance_type}"
        self.pricing_cache[key] = PricingInfo(
            cloud=cloud,
            region=region,
            instance_type=instance_type,
            on_demand_hourly=on_demand,
            spot_hourly=spot,
            spot_availability=spot_availability
        )
    
    def get_pricing(
        self,
        cloud: CloudProvider,
        region: str,
        instance_type: str
    ) -> Optional[PricingInfo]:
        """Get pricing information for a resource."""
        key = f"{cloud.value}:{region}:{instance_type}"
        return self.pricing_cache.get(key)
    
    async def find_cheapest_option(
        self,
        instance_types: Dict[CloudProvider, str],
        required_regions: Optional[List[str]] = None,
        allow_spot: bool = True,
        min_spot_availability: float = 0.8,
        duration_hours: int = 1,
        latency_requirement: Optional[Dict[str, int]] = None,
        data_residency: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find the cheapest cloud/region option for a workload.
        
        Args:
            instance_types: Mapping of cloud -> instance type
            required_regions: Specific regions to consider (optional)
            allow_spot: Whether to consider spot instances
            min_spot_availability: Minimum spot availability to consider
            duration_hours: Expected workload duration
            latency_requirement: Max latency by location (e.g., {"us": 50})
            data_residency: Allowed regions for data residency
            
        Returns:
            Sorted list of options from cheapest to most expensive
        """
        options = []
        
        for cloud, instance_type in instance_types.items():
            # Get all regions for this cloud
            regions = self._get_regions_for_cloud(cloud)
            
            if required_regions:
                regions = [r for r in regions if r in required_regions]
            
            if data_residency:
                regions = [r for r in regions if self._region_compliant(r, data_residency)]
            
            for region in regions:
                pricing = self.get_pricing(cloud, region, instance_type)
                if not pricing:
                    continue
                
                # Check latency requirement
                if latency_requirement and not self._meets_latency_requirement(
                    cloud, region, latency_requirement
                ):
                    continue
                
                # Calculate costs
                on_demand_cost = pricing.on_demand_hourly * duration_hours
                
                option = {
                    "cloud": cloud.value,
                    "region": region,
                    "instance_type": instance_type,
                    "pricing_type": "on_demand",
                    "hourly_rate": pricing.on_demand_hourly,
                    "total_cost": on_demand_cost,
                    "availability": 1.0
                }
                options.append(option)
                
                # Add spot option if available and allowed
                if allow_spot and pricing.spot_hourly:
                    if pricing.spot_availability >= min_spot_availability:
                        spot_cost = pricing.spot_hourly * duration_hours
                        spot_option = {
                            "cloud": cloud.value,
                            "region": region,
                            "instance_type": instance_type,
                            "pricing_type": "spot",
                            "hourly_rate": pricing.spot_hourly,
                            "total_cost": spot_cost,
                            "availability": pricing.spot_availability,
                            "savings_percent": (1 - pricing.spot_hourly / pricing.on_demand_hourly) * 100
                        }
                        options.append(spot_option)
        
        # Sort by total cost
        options.sort(key=lambda x: x["total_cost"])
        
        return options
    
    def _get_regions_for_cloud(self, cloud: CloudProvider) -> List[str]:
        """Get available regions for a cloud provider."""
        regions = {
            CloudProvider.AWS: ["us-east-1", "us-west-2", "eu-west-1", "ap-northeast-1"],
            CloudProvider.AZURE: ["eastus", "westeurope", "southeastasia"],
            CloudProvider.GCP: ["us-central1", "europe-west1", "asia-east1"]
        }
        return regions.get(cloud, [])
    
    def _region_compliant(self, region: str, data_residency: List[str]) -> bool:
        """Check if region complies with data residency requirements."""
        region_to_country = {
            "us-east-1": "us", "us-west-2": "us", "us-central1": "us", "eastus": "us",
            "eu-west-1": "eu", "westeurope": "eu", "europe-west1": "eu",
            "ap-northeast-1": "asia", "asia-east1": "asia", "southeastasia": "asia"
        }
        country = region_to_country.get(region, "unknown")
        return country in data_residency or "any" in data_residency
    
    def _meets_latency_requirement(
        self,
        cloud: CloudProvider,
        region: str,
        requirement: Dict[str, int]
    ) -> bool:
        """Check if region meets latency requirements."""
        cloud_regions = self.region_latency.get(cloud.value, {})
        region_latency = cloud_regions.get(region, {})
        
        for location, max_latency in requirement.items():
            actual_latency = region_latency.get(location, 999)
            if actual_latency > max_latency:
                return False
        return True
    
    # ============== Budget Management ==============
    
    def set_budget(self, config: BudgetConfig):
        """Set budget for a project."""
        self.budgets[config.project_id] = config
        logger.info("budget_set", project_id=config.project_id, limit=config.monthly_limit)
    
    def get_budget_status(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get current budget status for a project."""
        budget = self.budgets.get(project_id)
        if not budget:
            return None
        
        # Calculate current month spend
        current_spend = self._calculate_monthly_spend(project_id)
        
        soft_limit = budget.monthly_limit * budget.soft_limit_percent
        hard_limit = budget.monthly_limit * budget.hard_limit_percent
        
        return {
            "project_id": project_id,
            "monthly_limit": budget.monthly_limit,
            "current_spend": current_spend,
            "remaining": budget.monthly_limit - current_spend,
            "usage_percent": (current_spend / budget.monthly_limit) * 100,
            "soft_limit_breached": current_spend >= soft_limit,
            "hard_limit_breached": current_spend >= hard_limit
        }
    
    def _calculate_monthly_spend(self, project_id: str) -> float:
        """Calculate current month's spending for a project."""
        now = datetime.utcnow()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        return sum(
            record.cost for record in self.cost_records
            if record.project_id == project_id and record.timestamp >= month_start
        )
    
    async def check_budget_allowance(
        self,
        project_id: str,
        estimated_cost: float
    ) -> Tuple[bool, str]:
        """
        Check if a workload can proceed given budget constraints.
        
        Returns:
            Tuple of (allowed, reason)
        """
        budget = self.budgets.get(project_id)
        if not budget:
            return True, "No budget configured"
        
        current_spend = self._calculate_monthly_spend(project_id)
        projected_spend = current_spend + estimated_cost
        
        hard_limit = budget.monthly_limit * budget.hard_limit_percent
        
        if projected_spend > hard_limit:
            return False, f"Would exceed hard budget limit (${hard_limit:.2f})"
        
        soft_limit = budget.monthly_limit * budget.soft_limit_percent
        if projected_spend > soft_limit:
            return True, f"Warning: Approaching budget limit (${soft_limit:.2f})"
        
        return True, "Within budget"
    
    def record_cost(self, record: CostRecord):
        """Record an incurred cost."""
        self.cost_records.append(record)
        
        # Check for alerts
        budget = self.budgets.get(record.project_id)
        if budget:
            status = self.get_budget_status(record.project_id)
            if status and status["soft_limit_breached"]:
                self._send_budget_alert(budget, status)
    
    def _send_budget_alert(self, budget: BudgetConfig, status: Dict[str, Any]):
        """Send budget alert notification."""
        logger.warning(
            "budget_alert",
            project_id=budget.project_id,
            usage_percent=status["usage_percent"],
            current_spend=status["current_spend"],
            limit=budget.monthly_limit
        )
        # In production: send email/slack notifications


class SpotInterruptionHandler:
    """
    Handles spot/preemptible instance interruptions.
    
    Features:
    - Interruption notification monitoring
    - Checkpointing before termination
    - Automatic fallback to on-demand
    """
    
    def __init__(self, mcp_manager):
        self.mcp = mcp_manager
        self.active_spot_instances: Dict[str, Dict[str, Any]] = {}
        self.interruption_callbacks: Dict[str, callable] = {}
    
    async def register_spot_instance(
        self,
        instance_id: str,
        cloud: CloudProvider,
        checkpoint_callback: Optional[callable] = None
    ):
        """Register a spot instance for interruption monitoring."""
        self.active_spot_instances[instance_id] = {
            "cloud": cloud,
            "registered_at": datetime.utcnow()
        }
        
        if checkpoint_callback:
            self.interruption_callbacks[instance_id] = checkpoint_callback
        
        # Start monitoring for this instance
        asyncio.create_task(self._monitor_interruption(instance_id, cloud))
    
    async def _monitor_interruption(self, instance_id: str, cloud: CloudProvider):
        """Monitor for spot interruption notices."""
        while instance_id in self.active_spot_instances:
            # Check for interruption notice (varies by cloud)
            interrupted = await self._check_interruption_notice(instance_id, cloud)
            
            if interrupted:
                logger.warning(
                    "spot_interruption_detected",
                    instance_id=instance_id,
                    cloud=cloud.value
                )
                
                # Trigger checkpoint if callback registered
                if instance_id in self.interruption_callbacks:
                    try:
                        await self.interruption_callbacks[instance_id]()
                    except Exception as e:
                        logger.error("checkpoint_failed", instance_id=instance_id, error=str(e))
                
                # Clean up
                del self.active_spot_instances[instance_id]
                break
            
            await asyncio.sleep(5)  # Check every 5 seconds
    
    async def _check_interruption_notice(
        self,
        instance_id: str,
        cloud: CloudProvider
    ) -> bool:
        """Check if instance has received interruption notice."""
        # In production: query cloud metadata service
        # AWS: http://169.254.169.254/latest/meta-data/spot/termination-time
        # GCP: http://metadata.google.internal/computeMetadata/v1/instance/preempted
        return False


class IdleResourceCleaner:
    """
    Cleans up idle cloud resources to reduce costs.
    
    Features:
    - TTL-based cleanup
    - Usage-based detection
    - Automated termination
    """
    
    def __init__(self, mcp_manager):
        self.mcp = mcp_manager
        self.cleanup_rules: List[Dict[str, Any]] = []
    
    def add_cleanup_rule(
        self,
        resource_type: str,
        max_idle_minutes: int,
        tags: Optional[Dict[str, str]] = None
    ):
        """Add a cleanup rule for idle resources."""
        self.cleanup_rules.append({
            "resource_type": resource_type,
            "max_idle_minutes": max_idle_minutes,
            "tags": tags or {}
        })
    
    async def scan_and_cleanup(self) -> List[Dict[str, Any]]:
        """Scan for and cleanup idle resources."""
        cleaned = []
        
        for cloud in CloudProvider:
            # Get resource health/usage
            for rule in self.cleanup_rules:
                idle_resources = await self._find_idle_resources(
                    cloud,
                    rule["resource_type"],
                    rule["max_idle_minutes"],
                    rule["tags"]
                )
                
                for resource in idle_resources:
                    # Terminate idle resource
                    result = await self._cleanup_resource(cloud, resource)
                    cleaned.append({
                        "cloud": cloud.value,
                        "resource_id": resource["id"],
                        "resource_type": rule["resource_type"],
                        "idle_minutes": resource["idle_minutes"],
                        "cleanup_result": result
                    })
                    
                    logger.info(
                        "idle_resource_cleaned",
                        cloud=cloud.value,
                        resource_id=resource["id"],
                        idle_minutes=resource["idle_minutes"]
                    )
        
        return cleaned
    
    async def _find_idle_resources(
        self,
        cloud: CloudProvider,
        resource_type: str,
        max_idle_minutes: int,
        tags: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Find resources that have been idle beyond threshold."""
        # In production: query cloud APIs for usage metrics
        return []
    
    async def _cleanup_resource(
        self,
        cloud: CloudProvider,
        resource: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Cleanup a single resource."""
        # In production: call MCP to terminate resource
        return {"status": "cleaned"}
