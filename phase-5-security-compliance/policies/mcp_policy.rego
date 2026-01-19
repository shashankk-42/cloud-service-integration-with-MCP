# OPA Policy for MCP Cloud Operations
# Enforces security, compliance, and cost guardrails

package cloud_orchestrator.mcp

import future.keywords.if
import future.keywords.in

# Default deny
default allow := false

# ============== RBAC Definitions ==============

roles := {
    "developer": {
        "allowed_tools": [
            "get_cost_estimate",
            "get_quotas",
            "get_health",
            "submit_ml_job"
        ],
        "requires_approval": []
    },
    "operator": {
        "allowed_tools": [
            "get_cost_estimate",
            "get_quotas",
            "get_health",
            "submit_ml_job",
            "provision_compute",
            "scale_nodepool",
            "launch_spot",
            "create_storage_bucket",
            "deploy_model"
        ],
        "requires_approval": []
    },
    "admin": {
        "allowed_tools": [
            "get_cost_estimate",
            "get_quotas",
            "get_health",
            "submit_ml_job",
            "provision_compute",
            "scale_nodepool",
            "launch_spot",
            "create_storage_bucket",
            "deploy_model",
            "rotate_secret",
            "failover_route",
            "attach_volume"
        ],
        "requires_approval": ["rotate_secret", "failover_route"]
    },
    "security": {
        "allowed_tools": [
            "get_cost_estimate",
            "get_quotas",
            "get_health"
        ],
        "requires_approval": []
    }
}

# ============== Data Residency Rules ==============

# Allowed regions by data classification
data_residency_rules := {
    "pii": ["us-east-1", "us-west-2", "eastus", "us-central1"],  # US only for PII
    "financial": ["us-east-1", "eu-west-1", "westeurope"],  # US/EU for financial
    "public": ["*"],  # Any region for public data
    "restricted": ["us-east-1", "us-west-2"]  # Specific regions only
}

# EU GDPR regions
gdpr_regions := ["eu-west-1", "eu-west-2", "eu-central-1", "westeurope", "northeurope", "europe-west1", "europe-west2"]

# ============== Budget Limits ==============

# Maximum budget per role (USD per operation)
budget_limits := {
    "developer": 100,
    "operator": 1000,
    "admin": 10000
}

# ============== Instance Type Restrictions ==============

# Allowed instance families by role
allowed_instance_families := {
    "developer": ["t3", "t2", "Standard_B", "e2"],
    "operator": ["t3", "m5", "c5", "Standard_D", "Standard_E", "n1", "n2"],
    "admin": ["*"]
}

# GPU instance restrictions (require admin)
gpu_instance_patterns := ["p3", "p4", "g4", "Standard_NC", "Standard_ND", "a2", "n1-.*-nvidia"]

# ============== Main Authorization Rule ==============

allow if {
    # Check if user has permission for the tool
    tool_allowed
    
    # Check data residency compliance
    region_compliant
    
    # Check budget limits
    budget_compliant
    
    # Check instance type restrictions
    instance_allowed
    
    # Check approval if required
    approval_satisfied
}

# ============== Tool Authorization ==============

tool_allowed if {
    user_role := input.user.role
    tool := input.request.tool
    tool in roles[user_role].allowed_tools
}

# ============== Region Compliance ==============

region_compliant if {
    not requires_region_check
}

region_compliant if {
    requires_region_check
    data_class := input.request.data_classification
    region := input.request.region
    
    # Check if region is allowed for data classification
    allowed_regions := data_residency_rules[data_class]
    region in allowed_regions
}

region_compliant if {
    requires_region_check
    data_class := input.request.data_classification
    allowed_regions := data_residency_rules[data_class]
    "*" in allowed_regions
}

# GDPR compliance check
region_compliant if {
    requires_region_check
    input.request.gdpr_subject == true
    input.request.region in gdpr_regions
}

requires_region_check if {
    input.request.region != null
    input.request.data_classification != null
}

# ============== Budget Compliance ==============

budget_compliant if {
    not input.request.estimated_cost
}

budget_compliant if {
    user_role := input.user.role
    estimated_cost := input.request.estimated_cost
    max_budget := budget_limits[user_role]
    estimated_cost <= max_budget
}

# ============== Instance Type Restrictions ==============

instance_allowed if {
    not input.request.instance_type
}

instance_allowed if {
    user_role := input.user.role
    instance_type := input.request.instance_type
    
    # Admin can use any instance
    allowed_instance_families[user_role][_] == "*"
}

instance_allowed if {
    user_role := input.user.role
    instance_type := input.request.instance_type
    
    # Check if instance family is allowed
    some family in allowed_instance_families[user_role]
    startswith(instance_type, family)
    
    # Check GPU restrictions
    not is_gpu_instance(instance_type)
}

instance_allowed if {
    user_role := input.user.role
    instance_type := input.request.instance_type
    
    # GPU instances require admin
    is_gpu_instance(instance_type)
    user_role == "admin"
}

is_gpu_instance(instance_type) if {
    some pattern in gpu_instance_patterns
    regex.match(pattern, instance_type)
}

# ============== Approval Requirements ==============

approval_satisfied if {
    user_role := input.user.role
    tool := input.request.tool
    
    # Tool doesn't require approval
    not tool in roles[user_role].requires_approval
}

approval_satisfied if {
    user_role := input.user.role
    tool := input.request.tool
    
    # Tool requires approval but approval is present
    tool in roles[user_role].requires_approval
    input.request.approval != null
    valid_approval(input.request.approval)
}

valid_approval(approval) if {
    # Approval must be from an admin
    approval.approver_role == "admin"
    
    # Approval must not be expired (24 hours)
    time.now_ns() < approval.expires_ns
    
    # Approval must match the request
    approval.tool == input.request.tool
    approval.resource_type == input.request.resource_type
}

# ============== Audit Rules ==============

# Generate audit log entry
audit_log := {
    "timestamp": time.now_ns(),
    "user": input.user,
    "request": input.request,
    "decision": allow,
    "reasons": reasons
}

reasons[reason] if {
    not tool_allowed
    reason := sprintf("Tool '%s' not allowed for role '%s'", [input.request.tool, input.user.role])
}

reasons[reason] if {
    not region_compliant
    reason := sprintf("Region '%s' not compliant for data classification '%s'", [input.request.region, input.request.data_classification])
}

reasons[reason] if {
    not budget_compliant
    reason := sprintf("Estimated cost $%v exceeds budget limit $%v", [input.request.estimated_cost, budget_limits[input.user.role]])
}

reasons[reason] if {
    not instance_allowed
    reason := sprintf("Instance type '%s' not allowed for role '%s'", [input.request.instance_type, input.user.role])
}

reasons[reason] if {
    not approval_satisfied
    reason := sprintf("Tool '%s' requires approval", [input.request.tool])
}

# ============== Cost Estimation Rules ==============

# Estimate hourly cost based on instance type
estimated_hourly_cost(instance_type) := cost if {
    pricing := {
        "t3.micro": 0.0104,
        "t3.small": 0.0208,
        "t3.medium": 0.0416,
        "m5.large": 0.096,
        "p3.2xlarge": 3.06,
        "Standard_D2s_v3": 0.096,
        "Standard_NC6": 0.90,
        "n1-standard-2": 0.095
    }
    cost := pricing[instance_type]
}

estimated_hourly_cost(instance_type) := 0.10 if {
    not pricing[instance_type]
}

# ============== Helper Functions ==============

# Check if operation is read-only
is_read_only(tool) if {
    startswith(tool, "get_")
}

# Check if operation modifies state
is_write_operation(tool) if {
    not is_read_only(tool)
}

# High-risk operations
is_high_risk(tool) if {
    tool in ["rotate_secret", "failover_route", "attach_volume"]
}
