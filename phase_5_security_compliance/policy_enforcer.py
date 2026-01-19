"""
Policy Enforcer
Integrates with OPA for policy decision making.
"""

import asyncio
import httpx
import structlog
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = structlog.get_logger(__name__)


@dataclass
class PolicyDecision:
    """Result of a policy evaluation."""
    allowed: bool
    violated_policies: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    suggested_regions: Optional[List[str]] = None
    required_approvals: Optional[List[str]] = None


class PolicyEnforcer:
    """
    Enforces security and compliance policies using OPA.
    
    Features:
    - RBAC enforcement
    - Data residency compliance
    - Budget policy checks
    - Audit logging
    """
    
    def __init__(
        self,
        opa_endpoint: str = "http://localhost:8181",
        policy_path: str = "v1/data/mcp"
    ):
        self.opa_endpoint = opa_endpoint
        self.policy_path = policy_path
        self.http_client = httpx.AsyncClient(timeout=10.0)
        
        # Allowed regions for PII data (US-only by default)
        self.pii_allowed_regions = ["us-east-1", "us-west-2", "us-central1", "eastus", "westus"]
        
        # Role permissions matrix
        self.role_permissions = {
            "viewer": ["get_cost_estimate", "get_health", "list_resources"],
            "developer": [
                "get_cost_estimate", "get_health", "list_resources",
                "provision_compute", "submit_ml_job", "deploy_model",
                "create_storage_bucket", "scale_nodepool"
            ],
            "operator": [
                "get_cost_estimate", "get_health", "list_resources",
                "provision_compute", "submit_ml_job", "deploy_model",
                "create_storage_bucket", "scale_nodepool",
                "terminate_compute", "configure_logging"
            ],
            "admin": [
                "get_cost_estimate", "get_health", "list_resources",
                "provision_compute", "submit_ml_job", "deploy_model",
                "create_storage_bucket", "scale_nodepool",
                "terminate_compute", "configure_logging",
                "rotate_secret", "manage_iam"
            ]
        }
    
    async def evaluate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a request against policies.
        
        Args:
            request: Request containing action, user, data classification, etc.
            
        Returns:
            Policy decision with allowed status and details
        """
        violations = []
        suggested_regions = None
        
        # Check RBAC
        user = request.get("user", {})
        action = request.get("action", "")
        tool = request.get("tool", action)
        
        if not self._check_rbac(user.get("role", "viewer"), tool):
            violations.append("rbac")
        
        # Check data residency for PII
        data_classification = request.get("data_classification", "")
        target_region = request.get("target_region", "")
        
        if data_classification == "pii" and target_region:
            if not self._check_data_residency(target_region):
                violations.append("data_residency")
                suggested_regions = self.pii_allowed_regions
        
        # Check budget if applicable
        estimated_cost = request.get("estimated_cost", 0)
        budget_limit = request.get("budget_limit", float("inf"))
        
        if estimated_cost > budget_limit:
            violations.append("budget")
        
        # Try OPA evaluation if available
        try:
            opa_decision = await self._evaluate_opa(request)
            if not opa_decision.get("allow", True):
                violations.extend(opa_decision.get("violations", []))
        except Exception as e:
            logger.warning("opa_evaluation_failed", error=str(e))
        
        allowed = len(violations) == 0
        
        # Log the decision
        await self._audit_log(request, allowed, violations)
        
        return {
            "allowed": allowed,
            "violated_policies": violations,
            "reason": f"Violated policies: {', '.join(violations)}" if violations else None,
            "suggested_regions": suggested_regions
        }
    
    def _check_rbac(self, role: str, tool: str) -> bool:
        """Check if role has permission for the tool."""
        allowed_tools = self.role_permissions.get(role, [])
        return tool in allowed_tools or role == "admin"
    
    def _check_data_residency(self, region: str) -> bool:
        """Check if region is allowed for PII data."""
        return region in self.pii_allowed_regions
    
    async def _evaluate_opa(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate request against OPA policies."""
        try:
            response = await self.http_client.post(
                f"{self.opa_endpoint}/{self.policy_path}/allow",
                json={"input": request}
            )
            
            if response.status_code == 200:
                result = response.json()
                return {"allow": result.get("result", False)}
            
            return {"allow": True}  # Default allow if OPA unavailable
            
        except httpx.RequestError:
            return {"allow": True}  # Default allow if OPA unavailable
    
    async def _audit_log(
        self,
        request: Dict[str, Any],
        allowed: bool,
        violations: List[str]
    ):
        """Log policy decision for audit."""
        logger.info(
            "policy_decision",
            action=request.get("action"),
            user=request.get("user", {}).get("role"),
            allowed=allowed,
            violations=violations,
            timestamp=datetime.utcnow().isoformat()
        )
    
    async def check_approval_required(
        self,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if a request requires approval.
        
        Args:
            request: The request to check
            
        Returns:
            Dict with approval requirements
        """
        user_role = request.get("user", {}).get("role", "viewer")
        action = request.get("action", "")
        estimated_cost = request.get("estimated_cost", 0)
        
        required_approvals = []
        
        # High-cost actions require approval
        if estimated_cost > 1000:
            required_approvals.append("budget_owner")
        
        # Sensitive actions require admin approval
        sensitive_actions = ["rotate_secret", "manage_iam", "delete_data"]
        if action in sensitive_actions and user_role != "admin":
            required_approvals.append("admin")
        
        # Production deployments require approval
        if request.get("environment") == "production":
            required_approvals.append("release_manager")
        
        return {
            "approval_required": len(required_approvals) > 0,
            "required_approvals": required_approvals
        }
    
    async def close(self):
        """Close HTTP client."""
        await self.http_client.aclose()
