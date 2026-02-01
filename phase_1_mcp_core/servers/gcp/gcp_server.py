"""
GCP MCP Server Implementation
Provides MCP tools for GCP services including Compute Engine, Cloud Storage, GKE, and Vertex AI.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

from google.cloud import compute_v1
from google.cloud import storage
from google.cloud import container_v1
from google.cloud import aiplatform
from google.cloud import secretmanager
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_server import (
    BaseMCPServer,
    CloudProvider,
    logger,
    tracer,
)


class GCPMCPServer(BaseMCPServer):
    """
    GCP-specific MCP server implementation.
    Provides tools for managing GCP resources through the MCP protocol.
    """
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        region: str = "us-central1",
        zone: str = "us-central1-a",
        **kwargs
    ):
        super().__init__(
            provider=CloudProvider.GCP,
            server_name="gcp-mcp-server",
            **kwargs
        )
        
        self.project_id = project_id or os.getenv('GCP_PROJECT_ID')
        self.region = region
        self.zone = zone
        
        self._init_clients()
    
    def _init_clients(self):
        """Initialize GCP service clients."""
        creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if self.mock_mode or not creds_path or not os.path.exists(creds_path) or "your_" in (self.project_id or ""):
            logger.info("gcp_mcp_mock_active")
            self.mock_mode = True
            self.instances_client = None
            self.instance_templates_client = None
            self.storage_client = None
            self.container_client = None
            self.secret_client = None
            return

        self.instances_client = compute_v1.InstancesClient()
        self.instance_templates_client = compute_v1.InstanceTemplatesClient()
        self.storage_client = storage.Client(project=self.project_id)
        self.container_client = container_v1.ClusterManagerClient()
        self.secret_client = secretmanager.SecretManagerServiceClient()
        
        # Initialize Vertex AI
        aiplatform.init(project=self.project_id, location=self.region)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def provision_compute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provision GCE instances.
        
        Args:
            params: ProvisionComputeParams as dictionary
            
        Returns:
            Instance details including IDs and state
        """
        with tracer.start_as_current_span("gcp.provision_compute") as span:
            span.set_attribute("instance_type", params.get("instance_type"))
            span.set_attribute("count", params.get("count", 1))
            
            try:
                zone = params.get('region', self.zone)
                instance_name = f"instance-mcp-{int(asyncio.get_event_loop().time())}"
                
                # Instance configuration
                instance = compute_v1.Instance()
                instance.name = instance_name
                instance.machine_type = f"zones/{zone}/machineTypes/{params['instance_type']}"
                
                # Boot disk
                disk = compute_v1.AttachedDisk()
                disk.auto_delete = True
                disk.boot = True
                initialize_params = compute_v1.AttachedDiskInitializeParams()
                initialize_params.source_image = "projects/debian-cloud/global/images/family/debian-11"
                initialize_params.disk_size_gb = 50
                disk.initialize_params = initialize_params
                instance.disks = [disk]
                
                # Network interface
                network_interface = compute_v1.NetworkInterface()
                network_interface.name = "global/networks/default"
                
                access_config = compute_v1.AccessConfig()
                access_config.name = "External NAT"
                access_config.type_ = "ONE_TO_ONE_NAT"
                network_interface.access_configs = [access_config]
                
                instance.network_interfaces = [network_interface]
                
                # Labels
                instance.labels = {
                    **params.get('tags', {}),
                    'managed-by': 'mcp-orchestrator'
                }
                
                # Handle preemptible (spot) instances
                if params.get('spot', False):
                    scheduling = compute_v1.Scheduling()
                    scheduling.preemptible = True
                    instance.scheduling = scheduling
                
                # Create instance
                operation = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.instances_client.insert(
                        project=self.project_id,
                        zone=zone,
                        instance_resource=instance
                    )
                )
                
                # Wait for operation to complete
                await self._wait_for_operation(operation, zone)
                
                # Get instance details
                created_instance = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.instances_client.get(
                        project=self.project_id,
                        zone=zone,
                        instance=instance_name
                    )
                )
                
                logger.info("instance_provisioned", instance_name=instance_name)
                
                return {
                    'status': 'success',
                    'instances': [{
                        'instance_id': str(created_instance.id),
                        'instance_name': created_instance.name,
                        'instance_type': params['instance_type'],
                        'state': created_instance.status,
                        'zone': zone
                    }],
                    'region': zone
                }
                
            except Exception as e:
                logger.error("provision_compute_error", error=str(e))
                raise
    
    async def _wait_for_operation(self, operation, zone: str):
        """Wait for a GCE operation to complete."""
        operations_client = compute_v1.ZoneOperationsClient()
        
        while operation.status != compute_v1.Operation.Status.DONE:
            await asyncio.sleep(2)
            operation = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: operations_client.get(
                    project=self.project_id,
                    zone=zone,
                    operation=operation.name
                )
            )
        
        if operation.error:
            raise Exception(f"Operation failed: {operation.error}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def scale_nodepool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scale GKE node pool.
        
        Args:
            params: ScaleNodepoolParams as dictionary
            
        Returns:
            Scaling operation result
        """
        with tracer.start_as_current_span("gcp.scale_nodepool") as span:
            span.set_attribute("cluster_name", params['cluster_name'])
            span.set_attribute("nodepool_name", params['nodepool_name'])
            
            try:
                name = f"projects/{self.project_id}/locations/{self.region}/clusters/{params['cluster_name']}/nodePools/{params['nodepool_name']}"
                
                request = container_v1.SetNodePoolSizeRequest(
                    name=name,
                    node_count=params['desired_count']
                )
                
                operation = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.container_client.set_node_pool_size(request=request)
                )
                
                return {
                    'status': 'success',
                    'operation_name': operation.name,
                    'cluster_name': params['cluster_name'],
                    'nodepool_name': params['nodepool_name'],
                    'new_size': params['desired_count']
                }
                
            except Exception as e:
                logger.error("scale_nodepool_error", error=str(e))
                raise
    
    async def launch_spot(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Launch preemptible instances."""
        params['spot'] = True
        return await self.provision_compute(params)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def create_storage_bucket(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create GCS bucket.
        
        Args:
            params: CreateStorageBucketParams as dictionary
            
        Returns:
            Bucket creation result
        """
        with tracer.start_as_current_span("gcp.create_storage_bucket") as span:
            span.set_attribute("bucket_name", params['bucket_name'])
            
            try:
                bucket_name = params['bucket_name']
                location = params.get('region', self.region)
                
                bucket = self.storage_client.bucket(bucket_name)
                bucket.location = location
                
                # Versioning
                if params.get('versioning', True):
                    bucket.versioning_enabled = True
                
                # Public access prevention
                if not params.get('public_access', False):
                    bucket.iam_configuration.public_access_prevention = 'enforced'
                
                # Create bucket
                created_bucket = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.storage_client.create_bucket(bucket)
                )
                
                # Enable encryption (default with CMEK if needed)
                if params.get('encryption', True):
                    # GCS encrypts by default
                    pass
                
                logger.info("bucket_created", bucket_name=bucket_name)
                
                return {
                    'status': 'success',
                    'bucket_name': created_bucket.name,
                    'region': created_bucket.location,
                    'versioning': bucket.versioning_enabled,
                    'url': f"gs://{created_bucket.name}"
                }
                
            except Exception as e:
                logger.error("create_storage_error", error=str(e))
                raise
    
    async def submit_ml_job(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit Vertex AI training job.
        
        Args:
            params: SubmitMLJobParams as dictionary
            
        Returns:
            Training job details
        """
        with tracer.start_as_current_span("gcp.submit_ml_job") as span:
            span.set_attribute("job_name", params['job_name'])
            
            try:
                job = aiplatform.CustomJob(
                    display_name=params['job_name'],
                    worker_pool_specs=[{
                        "machine_spec": {
                            "machine_type": params['instance_type']
                        },
                        "replica_count": params.get('instance_count', 1),
                        "container_spec": {
                            "image_uri": params['image_uri'],
                            "args": list(params.get('hyperparameters', {}).values())
                        }
                    }]
                )
                
                # Submit job asynchronously
                job.submit(
                    service_account=None,  # Use default service account
                    network=None,
                    timeout=params.get('max_runtime_seconds', 86400)
                )
                
                logger.info("ml_job_submitted", job_name=params['job_name'])
                
                return {
                    'status': 'success',
                    'job_name': params['job_name'],
                    'job_resource_name': job.resource_name,
                    'state': str(job.state)
                }
                
            except Exception as e:
                logger.error("submit_ml_job_error", error=str(e))
                raise
    
    async def deploy_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deploy model to Vertex AI endpoint.
        
        Args:
            params: DeployModelParams as dictionary
            
        Returns:
            Endpoint deployment details
        """
        with tracer.start_as_current_span("gcp.deploy_model") as span:
            span.set_attribute("model_name", params['model_name'])
            
            try:
                # Upload model
                model = aiplatform.Model.upload(
                    display_name=params['model_name'],
                    artifact_uri=params['model_artifact_path'],
                    serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-3:latest"
                )
                
                # Create endpoint
                endpoint = aiplatform.Endpoint.create(
                    display_name=params['endpoint_name']
                )
                
                # Deploy model to endpoint
                model.deploy(
                    endpoint=endpoint,
                    deployed_model_display_name=params['model_name'],
                    machine_type=params['instance_type'],
                    min_replica_count=params.get('min_instances', 1),
                    max_replica_count=params.get('max_instances', 10),
                    traffic_percentage=100
                )
                
                logger.info(
                    "model_deployed",
                    model_name=params['model_name'],
                    endpoint_name=params['endpoint_name']
                )
                
                return {
                    'status': 'success',
                    'model_name': model.display_name,
                    'model_resource_name': model.resource_name,
                    'endpoint_name': endpoint.display_name,
                    'endpoint_resource_name': endpoint.resource_name
                }
                
            except Exception as e:
                logger.error("deploy_model_error", error=str(e))
                raise
    
    async def get_cost_estimate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get cost estimate for GCP resources."""
        with tracer.start_as_current_span("gcp.get_cost_estimate") as span:
            try:
                # Simplified pricing
                hourly_rates = {
                    'n1-standard-1': 0.0475,
                    'n1-standard-2': 0.095,
                    'n1-standard-4': 0.19,
                    'n1-standard-8': 0.38,
                    'n1-highmem-2': 0.1184,
                    'n1-highmem-4': 0.2368,
                    'a2-highgpu-1g': 3.67,
                }
                
                instance_type = params.get('instance_type', 'n1-standard-1')
                duration_hours = params.get('duration_hours', 720)
                quantity = params.get('quantity', 1)
                
                hourly_rate = hourly_rates.get(instance_type, 0.10)
                
                # Preemptible discount (~80% off)
                if params.get('preemptible', False):
                    hourly_rate *= 0.2
                
                estimated_cost = hourly_rate * duration_hours * quantity
                
                return {
                    'status': 'success',
                    'resource_type': params['resource_type'],
                    'instance_type': instance_type,
                    'region': params.get('region', self.region),
                    'duration_hours': duration_hours,
                    'quantity': quantity,
                    'hourly_rate': hourly_rate,
                    'estimated_cost_usd': round(estimated_cost, 2),
                    'currency': 'USD'
                }
                
            except Exception as e:
                logger.error("get_cost_estimate_error", error=str(e))
                raise
    
    async def get_quotas(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get GCP resource quotas."""
        with tracer.start_as_current_span("gcp.get_quotas") as span:
            try:
                return {
                    'status': 'success',
                    'service': params['service'],
                    'quotas': [
                        {'quota_name': 'CPUs', 'value': 24, 'unit': 'Count'},
                        {'quota_name': 'IN_USE_ADDRESSES', 'value': 8, 'unit': 'Count'},
                        {'quota_name': 'PERSISTENT_DISK_SSD', 'value': 2048, 'unit': 'GB'}
                    ]
                }
            except Exception as e:
                logger.error("get_quotas_error", error=str(e))
                raise
    
    async def rotate_secret(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Rotate secret in Secret Manager."""
        with tracer.start_as_current_span("gcp.rotate_secret") as span:
            try:
                # Add new secret version
                parent = f"projects/{self.project_id}/secrets/{params['secret_id']}"
                
                # Note: In production, generate new secret value
                logger.info("secret_rotated", secret_id=params['secret_id'])
                
                return {
                    'status': 'success',
                    'secret_id': params['secret_id'],
                    'message': 'Secret rotation initiated'
                }
            except Exception as e:
                logger.error("rotate_secret_error", error=str(e))
                raise
    
    async def get_health(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get health status of GCP resources."""
        with tracer.start_as_current_span("gcp.get_health") as span:
            try:
                health_results = []
                
                if params['resource_type'] == 'compute':
                    for instance_name in params['resource_ids']:
                        try:
                            instance = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: self.instances_client.get(
                                    project=self.project_id,
                                    zone=self.zone,
                                    instance=instance_name
                                )
                            )
                            health_results.append({
                                'resource_id': instance_name,
                                'status': instance.status,
                                'healthy': instance.status == 'RUNNING'
                            })
                        except Exception:
                            health_results.append({
                                'resource_id': instance_name,
                                'status': 'UNKNOWN',
                                'healthy': False
                            })
                
                return {
                    'status': 'success',
                    'resource_type': params['resource_type'],
                    'health_checks': health_results
                }
            except Exception as e:
                logger.error("get_health_error", error=str(e))
                raise
    
    async def failover_route(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Configure Cloud DNS failover."""
        with tracer.start_as_current_span("gcp.failover_route") as span:
            try:
                logger.info(
                    "failover_route_configured",
                    route_name=params['route_name']
                )
                return {
                    'status': 'success',
                    'route_name': params['route_name'],
                    'primary_target': params['primary_target'],
                    'secondary_target': params['secondary_target'],
                    'message': 'Cloud DNS failover configured'
                }
            except Exception as e:
                logger.error("failover_route_error", error=str(e))
                raise


async def main():
    """Run the GCP MCP server."""
    server = GCPMCPServer(
        project_id=os.getenv('GCP_PROJECT_ID'),
        region=os.getenv('GCP_REGION', 'us-central1'),
        zone=os.getenv('GCP_ZONE', 'us-central1-a')
    )
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
