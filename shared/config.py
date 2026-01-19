"""
Shared Configuration Module
Central configuration management for the cloud orchestrator platform.
"""

import os
from typing import Any, Dict, List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class CloudConfig(BaseSettings):
    """Configuration for cloud provider connections."""
    
    # Cloud enable flags
    aws_enabled: bool = Field(default=True)
    azure_enabled: bool = Field(default=True)
    gcp_enabled: bool = Field(default=True)
    
    # AWS Configuration
    aws_access_key_id: Optional[str] = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", alias="AWS_DEFAULT_REGION")
    aws_endpoint_url: Optional[str] = Field(default=None, alias="AWS_ENDPOINT_URL")
    
    # Azure Configuration
    azure_subscription_id: Optional[str] = Field(default=None, alias="AZURE_SUBSCRIPTION_ID")
    azure_tenant_id: Optional[str] = Field(default=None, alias="AZURE_TENANT_ID")
    azure_client_id: Optional[str] = Field(default=None, alias="AZURE_CLIENT_ID")
    azure_client_secret: Optional[str] = Field(default=None, alias="AZURE_CLIENT_SECRET")
    azure_resource_group: str = Field(default="cloud-orchestrator-rg", alias="AZURE_RESOURCE_GROUP")
    
    # GCP Configuration
    gcp_project_id: Optional[str] = Field(default=None, alias="GCP_PROJECT_ID")
    gcp_credentials_path: Optional[str] = Field(default=None, alias="GOOGLE_APPLICATION_CREDENTIALS")
    gcp_region: str = Field(default="us-central1", alias="GCP_REGION")
    
    class Config:
        env_file = ".env"
        extra = "ignore"


class MCPConfig(BaseSettings):
    """Configuration for MCP servers."""
    
    aws_server_port: int = Field(default=8001, alias="MCP_AWS_SERVER_PORT")
    azure_server_port: int = Field(default=8002, alias="MCP_AZURE_SERVER_PORT")
    gcp_server_port: int = Field(default=8003, alias="MCP_GCP_SERVER_PORT")
    orchestrator_port: int = Field(default=8000, alias="MCP_ORCHESTRATOR_PORT")
    
    # Rate limiting
    rate_limit_requests_per_second: float = Field(default=10.0)
    rate_limit_burst: int = Field(default=100)
    
    # Circuit breaker
    circuit_breaker_failure_threshold: int = Field(default=5)
    circuit_breaker_recovery_timeout: int = Field(default=60)
    
    class Config:
        env_file = ".env"
        extra = "ignore"


class DatabaseConfig(BaseSettings):
    """Configuration for database connections."""
    
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/cloud_orchestrator",
        alias="DATABASE_URL"
    )
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    
    # Connection pool
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)
    pool_timeout: int = Field(default=30)
    
    class Config:
        env_file = ".env"
        extra = "ignore"


class ObservabilityConfig(BaseSettings):
    """Configuration for observability stack."""
    
    otlp_endpoint: str = Field(default="http://localhost:4317", alias="OTLP_ENDPOINT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    # Prometheus
    prometheus_port: int = Field(default=9090)
    metrics_enabled: bool = Field(default=True)
    
    # Tracing
    tracing_enabled: bool = Field(default=True)
    trace_sample_rate: float = Field(default=0.1)  # 10% sampling in production
    
    class Config:
        env_file = ".env"
        extra = "ignore"


class SecurityConfig(BaseSettings):
    """Configuration for security settings."""
    
    jwt_secret_key: str = Field(default="change-me-in-production", alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expiration_minutes: int = Field(default=30, alias="JWT_EXPIRATION_MINUTES")
    
    # mTLS
    mtls_enabled: bool = Field(default=True)
    ca_cert_path: Optional[str] = Field(default=None)
    server_cert_path: Optional[str] = Field(default=None)
    server_key_path: Optional[str] = Field(default=None)
    
    # OPA Policy
    opa_endpoint: str = Field(default="http://localhost:8181")
    policy_bundle_path: str = Field(default="/policies")
    
    # Security defaults
    require_mfa_for_admin: bool = Field(default=True)
    secret_rotation_days: int = Field(default=90)
    audit_log_retention_days: int = Field(default=365)
    
    class Config:
        env_file = ".env"
        extra = "ignore"


class CostConfig(BaseSettings):
    """Configuration for cost management."""
    
    default_budget_limit: float = Field(default=1000.0, alias="DEFAULT_BUDGET_LIMIT")
    cost_alert_threshold: float = Field(default=0.8, alias="COST_ALERT_THRESHOLD")
    spot_max_price_multiplier: float = Field(default=0.5, alias="SPOT_MAX_PRICE_MULTIPLIER")
    
    # Idle resource cleanup
    idle_resource_ttl_minutes: int = Field(default=60)
    cleanup_enabled: bool = Field(default=True)
    
    class Config:
        env_file = ".env"
        extra = "ignore"


class FeatureFlags(BaseSettings):
    """Feature flags for gradual rollout."""
    
    enable_spot_instances: bool = Field(default=True, alias="ENABLE_SPOT_INSTANCES")
    enable_cross_cloud_arbitrage: bool = Field(default=True, alias="ENABLE_CROSS_CLOUD_ARBITRAGE")
    enable_auto_scaling: bool = Field(default=True, alias="ENABLE_AUTO_SCALING")
    enable_multi_cloud: bool = Field(default=True)
    enable_canary_deployments: bool = Field(default=True)
    enable_cost_optimization: bool = Field(default=True)
    
    class Config:
        env_file = ".env"
        extra = "ignore"


class AppConfig(BaseSettings):
    """Main application configuration aggregating all configs."""
    
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")
    
    cloud: CloudConfig = Field(default_factory=CloudConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    
    class Config:
        env_file = ".env"
        extra = "ignore"


# Singleton config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the application configuration singleton."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def reload_config():
    """Reload configuration from environment."""
    global _config
    _config = AppConfig()
    return _config
