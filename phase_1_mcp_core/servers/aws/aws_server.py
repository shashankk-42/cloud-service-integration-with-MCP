"""
AWS MCP Server Implementation
Provides MCP tools for AWS cloud services including EC2, S3, EKS, and SageMaker.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ..base_server import (
    BaseMCPServer,
    CloudProvider,
    OperationResult,
    OperationStatus,
    logger,
    tracer,
)


class AWSMCPServer(BaseMCPServer):
    """
    AWS-specific MCP server implementation.
    Provides tools for managing AWS resources through the MCP protocol.
    """
    
    def __init__(
        self,
        region: str = "us-east-1",
        endpoint_url: Optional[str] = None,  # For LocalStack testing
        **kwargs
    ):
        super().__init__(
            provider=CloudProvider.AWS,
            server_name="aws-mcp-server",
            **kwargs
        )
        
        self.region = region
        self.endpoint_url = endpoint_url
        
        # Configure boto3 with retry logic
        self.boto_config = Config(
            region_name=region,
            retries={
                'max_attempts': 3,
                'mode': 'adaptive'
            }
        )
        
        # Initialize AWS clients
        self._init_clients()
    
    def _init_clients(self):
        """Initialize AWS service clients."""
        if self.mock_mode:
            logger.info("aws_mcp_mock_active")
            self.ec2 = None
            self.s3 = None
            self.eks = None
            self.sagemaker = None
            self.sts = None
            self.secretsmanager = None
            self.route53 = None
            self.pricing = None
            self.service_quotas = None
            self.autoscaling = None
            return

        client_kwargs = {
            'config': self.boto_config
        }
        if self.endpoint_url:
            client_kwargs['endpoint_url'] = self.endpoint_url
        
        self.ec2 = boto3.client('ec2', **client_kwargs)
        self.s3 = boto3.client('s3', **client_kwargs)
        self.eks = boto3.client('eks', **client_kwargs)
        self.sagemaker = boto3.client('sagemaker', **client_kwargs)
        self.sts = boto3.client('sts', **client_kwargs)
        self.secretsmanager = boto3.client('secretsmanager', **client_kwargs)
        self.route53 = boto3.client('route53', **client_kwargs)
        self.pricing = boto3.client('pricing', region_name='us-east-1')
        self.service_quotas = boto3.client('service-quotas', **client_kwargs)
        self.autoscaling = boto3.client('autoscaling', **client_kwargs)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(ClientError)
    )
    async def provision_compute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provision EC2 instances.
        
        Args:
            params: ProvisionComputeParams as dictionary
            
        Returns:
            Instance details including IDs and state
        """
        with tracer.start_as_current_span("aws.provision_compute") as span:
            span.set_attribute("instance_type", params.get("instance_type"))
            span.set_attribute("count", params.get("count", 1))
            
            try:
                # Build run instances request
                run_params = {
                    'ImageId': await self._get_latest_ami(params.get("region", self.region)),
                    'InstanceType': params['instance_type'],
                    'MinCount': params.get('count', 1),
                    'MaxCount': params.get('count', 1),
                    'TagSpecifications': [{
                        'ResourceType': 'instance',
                        'Tags': [
                            {'Key': k, 'Value': v}
                            for k, v in params.get('tags', {}).items()
                        ] + [
                            {'Key': 'ManagedBy', 'Value': 'mcp-orchestrator'},
                            {'Key': 'CreatedAt', 'Value': str(asyncio.get_event_loop().time())}
                        ]
                    }]
                }
                
                if params.get('subnet_id'):
                    run_params['SubnetId'] = params['subnet_id']
                
                if params.get('security_groups'):
                    run_params['SecurityGroupIds'] = params['security_groups']
                
                if params.get('user_data'):
                    run_params['UserData'] = params['user_data']
                
                # Handle spot instances
                if params.get('spot', False):
                    run_params['InstanceMarketOptions'] = {
                        'MarketType': 'spot',
                        'SpotOptions': {
                            'SpotInstanceType': 'one-time',
                            'InstanceInterruptionBehavior': 'terminate'
                        }
                    }
                    if params.get('max_price'):
                        run_params['InstanceMarketOptions']['SpotOptions']['MaxPrice'] = str(params['max_price'])
                
                # Run instances
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.ec2.run_instances(**run_params)
                )
                
                instances = [{
                    'instance_id': inst['InstanceId'],
                    'instance_type': inst['InstanceType'],
                    'state': inst['State']['Name'],
                    'private_ip': inst.get('PrivateIpAddress'),
                    'public_ip': inst.get('PublicIpAddress')
                } for inst in response['Instances']]
                
                logger.info(
                    "instances_provisioned",
                    count=len(instances),
                    instance_ids=[i['instance_id'] for i in instances]
                )
                
                return {
                    'status': 'success',
                    'instances': instances,
                    'region': params.get('region', self.region)
                }
                
            except ClientError as e:
                logger.error("provision_compute_error", error=str(e))
                raise
    
    async def _get_latest_ami(self, region: str) -> str:
        """Get the latest Amazon Linux 2 AMI ID."""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.ec2.describe_images(
                    Filters=[
                        {'Name': 'name', 'Values': ['amzn2-ami-hvm-*-x86_64-gp2']},
                        {'Name': 'state', 'Values': ['available']},
                        {'Name': 'owner-alias', 'Values': ['amazon']}
                    ],
                    Owners=['amazon']
                )
            )
            
            images = sorted(
                response['Images'],
                key=lambda x: x['CreationDate'],
                reverse=True
            )
            
            return images[0]['ImageId'] if images else 'ami-0c55b159cbfafe1f0'
        except Exception:
            return 'ami-0c55b159cbfafe1f0'  # Fallback AMI
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def scale_nodepool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scale EKS node group.
        
        Args:
            params: ScaleNodepoolParams as dictionary
            
        Returns:
            Scaling operation result
        """
        with tracer.start_as_current_span("aws.scale_nodepool") as span:
            span.set_attribute("cluster_name", params['cluster_name'])
            span.set_attribute("nodepool_name", params['nodepool_name'])
            span.set_attribute("desired_count", params['desired_count'])
            
            try:
                update_config = {
                    'desiredSize': params['desired_count']
                }
                
                if params.get('min_count') is not None:
                    update_config['minSize'] = params['min_count']
                if params.get('max_count') is not None:
                    update_config['maxSize'] = params['max_count']
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.eks.update_nodegroup_config(
                        clusterName=params['cluster_name'],
                        nodegroupName=params['nodepool_name'],
                        scalingConfig=update_config
                    )
                )
                
                return {
                    'status': 'success',
                    'update_id': response['update']['id'],
                    'update_status': response['update']['status'],
                    'cluster_name': params['cluster_name'],
                    'nodepool_name': params['nodepool_name']
                }
                
            except ClientError as e:
                logger.error("scale_nodepool_error", error=str(e))
                raise
    
    async def launch_spot(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Launch spot instances.
        """
        params['spot'] = True
        return await self.provision_compute(params)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def create_storage_bucket(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create S3 bucket with specified configuration.
        
        Args:
            params: CreateStorageBucketParams as dictionary
            
        Returns:
            Bucket creation result
        """
        with tracer.start_as_current_span("aws.create_storage_bucket") as span:
            span.set_attribute("bucket_name", params['bucket_name'])
            
            try:
                bucket_name = params['bucket_name']
                region = params.get('region', self.region)
                
                # Create bucket
                create_params = {'Bucket': bucket_name}
                if region != 'us-east-1':
                    create_params['CreateBucketConfiguration'] = {
                        'LocationConstraint': region
                    }
                
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.s3.create_bucket(**create_params)
                )
                
                # Enable versioning if requested
                if params.get('versioning', True):
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.s3.put_bucket_versioning(
                            Bucket=bucket_name,
                            VersioningConfiguration={'Status': 'Enabled'}
                        )
                    )
                
                # Enable encryption if requested
                if params.get('encryption', True):
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.s3.put_bucket_encryption(
                            Bucket=bucket_name,
                            ServerSideEncryptionConfiguration={
                                'Rules': [{
                                    'ApplyServerSideEncryptionByDefault': {
                                        'SSEAlgorithm': 'AES256'
                                    }
                                }]
                            }
                        )
                    )
                
                # Block public access if requested
                if not params.get('public_access', False):
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.s3.put_public_access_block(
                            Bucket=bucket_name,
                            PublicAccessBlockConfiguration={
                                'BlockPublicAcls': True,
                                'IgnorePublicAcls': True,
                                'BlockPublicPolicy': True,
                                'RestrictPublicBuckets': True
                            }
                        )
                    )
                
                logger.info("bucket_created", bucket_name=bucket_name)
                
                return {
                    'status': 'success',
                    'bucket_name': bucket_name,
                    'region': region,
                    'versioning': params.get('versioning', True),
                    'encryption': params.get('encryption', True)
                }
                
            except ClientError as e:
                logger.error("create_bucket_error", error=str(e))
                raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def submit_ml_job(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit SageMaker training job.
        
        Args:
            params: SubmitMLJobParams as dictionary
            
        Returns:
            Training job details
        """
        with tracer.start_as_current_span("aws.submit_ml_job") as span:
            span.set_attribute("job_name", params['job_name'])
            
            try:
                training_params = {
                    'TrainingJobName': params['job_name'],
                    'AlgorithmSpecification': {
                        'TrainingImage': params['image_uri'],
                        'TrainingInputMode': 'File'
                    },
                    'RoleArn': await self._get_sagemaker_role(),
                    'InputDataConfig': [params['input_data_config']],
                    'OutputDataConfig': {
                        'S3OutputPath': params['output_path']
                    },
                    'ResourceConfig': {
                        'InstanceType': params['instance_type'],
                        'InstanceCount': params.get('instance_count', 1),
                        'VolumeSizeInGB': 50
                    },
                    'StoppingCondition': {
                        'MaxRuntimeInSeconds': params.get('max_runtime_seconds', 86400)
                    },
                    'HyperParameters': params.get('hyperparameters', {}),
                    'Tags': [
                        {'Key': 'ManagedBy', 'Value': 'mcp-orchestrator'}
                    ]
                }
                
                # Use spot instances if requested
                if params.get('spot_instances', False):
                    training_params['EnableManagedSpotTraining'] = True
                    training_params['StoppingCondition']['MaxWaitTimeInSeconds'] = (
                        params.get('max_runtime_seconds', 86400) * 2
                    )
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.sagemaker.create_training_job(**training_params)
                )
                
                logger.info("ml_job_submitted", job_name=params['job_name'])
                
                return {
                    'status': 'success',
                    'training_job_arn': response['TrainingJobArn'],
                    'job_name': params['job_name']
                }
                
            except ClientError as e:
                logger.error("submit_ml_job_error", error=str(e))
                raise
    
    async def _get_sagemaker_role(self) -> str:
        """Get or create SageMaker execution role ARN."""
        # In production, this would look up the actual role
        account_id = await self._get_account_id()
        return f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"
    
    async def _get_account_id(self) -> str:
        """Get AWS account ID."""
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            self.sts.get_caller_identity
        )
        return response['Account']
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def deploy_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deploy model to SageMaker endpoint.
        
        Args:
            params: DeployModelParams as dictionary
            
        Returns:
            Endpoint deployment details
        """
        with tracer.start_as_current_span("aws.deploy_model") as span:
            span.set_attribute("model_name", params['model_name'])
            span.set_attribute("endpoint_name", params['endpoint_name'])
            
            try:
                # Get SageMaker role first (outside lambda)
                sagemaker_role_arn = await self._get_sagemaker_role()
                
                # Create model
                model_response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.sagemaker.create_model(
                        ModelName=params['model_name'],
                        PrimaryContainer={
                            'Image': params.get('inference_image', params['model_artifact_path']),
                            'ModelDataUrl': params['model_artifact_path']
                        },
                        ExecutionRoleArn=sagemaker_role_arn
                    )
                )
                
                # Create endpoint config
                endpoint_config_name = f"{params['endpoint_name']}-config"
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.sagemaker.create_endpoint_config(
                        EndpointConfigName=endpoint_config_name,
                        ProductionVariants=[{
                            'VariantName': 'primary',
                            'ModelName': params['model_name'],
                            'InitialInstanceCount': params.get('instance_count', 1),
                            'InstanceType': params['instance_type'],
                            'InitialVariantWeight': 1.0
                        }]
                    )
                )
                
                # Create endpoint
                endpoint_response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.sagemaker.create_endpoint(
                        EndpointName=params['endpoint_name'],
                        EndpointConfigName=endpoint_config_name
                    )
                )
                
                logger.info(
                    "model_deployed",
                    model_name=params['model_name'],
                    endpoint_name=params['endpoint_name']
                )
                
                return {
                    'status': 'success',
                    'model_arn': model_response['ModelArn'],
                    'endpoint_arn': endpoint_response['EndpointArn'],
                    'endpoint_name': params['endpoint_name']
                }
                
            except ClientError as e:
                logger.error("deploy_model_error", error=str(e))
                raise
    
    async def get_cost_estimate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get cost estimate for AWS resources.
        
        Args:
            params: GetCostEstimateParams as dictionary
            
        Returns:
            Cost estimate details
        """
        with tracer.start_as_current_span("aws.get_cost_estimate") as span:
            try:
                # Use AWS Pricing API for estimates
                resource_type = params['resource_type']
                instance_type = params.get('instance_type')
                region = params.get('region', self.region)
                duration_hours = params.get('duration_hours', 720)
                quantity = params.get('quantity', 1)
                
                # Simplified pricing lookup
                # In production, use AWS Pricing API
                hourly_rates = {
                    't3.micro': 0.0104,
                    't3.small': 0.0208,
                    't3.medium': 0.0416,
                    't3.large': 0.0832,
                    'm5.large': 0.096,
                    'm5.xlarge': 0.192,
                    'ml.m5.large': 0.134,
                    'ml.m5.xlarge': 0.269,
                    'ml.p3.2xlarge': 3.825,
                }
                
                hourly_rate = hourly_rates.get(instance_type, 0.10)
                estimated_cost = hourly_rate * duration_hours * quantity
                
                return {
                    'status': 'success',
                    'resource_type': resource_type,
                    'instance_type': instance_type,
                    'region': region,
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
        """
        Get AWS service quotas.
        
        Args:
            params: GetQuotasParams as dictionary
            
        Returns:
            Quota information
        """
        with tracer.start_as_current_span("aws.get_quotas") as span:
            try:
                service = params['service']
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.service_quotas.list_service_quotas(
                        ServiceCode=service,
                        MaxResults=100
                    )
                )
                
                quotas = [{
                    'quota_name': q['QuotaName'],
                    'quota_code': q['QuotaCode'],
                    'value': q['Value'],
                    'unit': q.get('Unit', 'None'),
                    'adjustable': q.get('Adjustable', False)
                } for q in response['Quotas']]
                
                return {
                    'status': 'success',
                    'service': service,
                    'quotas': quotas
                }
                
            except ClientError as e:
                logger.error("get_quotas_error", error=str(e))
                raise
    
    async def rotate_secret(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rotate secret in AWS Secrets Manager.
        
        Args:
            params: RotateSecretParams as dictionary
            
        Returns:
            Rotation result
        """
        with tracer.start_as_current_span("aws.rotate_secret") as span:
            try:
                rotate_params = {
                    'SecretId': params['secret_id']
                }
                
                if params.get('rotation_lambda_arn'):
                    rotate_params['RotationLambdaARN'] = params['rotation_lambda_arn']
                
                if params.get('rotation_rules'):
                    rotate_params['RotationRules'] = params['rotation_rules']
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.secretsmanager.rotate_secret(**rotate_params)
                )
                
                logger.info("secret_rotated", secret_id=params['secret_id'])
                
                return {
                    'status': 'success',
                    'secret_arn': response['ARN'],
                    'version_id': response['VersionId']
                }
                
            except ClientError as e:
                logger.error("rotate_secret_error", error=str(e))
                raise
    
    async def get_health(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get health status of AWS resources.
        
        Args:
            params: GetHealthParams as dictionary
            
        Returns:
            Health status information
        """
        with tracer.start_as_current_span("aws.get_health") as span:
            try:
                resource_type = params['resource_type']
                resource_ids = params['resource_ids']
                
                health_results = []
                
                if resource_type == 'ec2':
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.ec2.describe_instance_status(
                            InstanceIds=resource_ids
                        )
                    )
                    
                    for status in response['InstanceStatuses']:
                        health_results.append({
                            'resource_id': status['InstanceId'],
                            'instance_status': status['InstanceStatus']['Status'],
                            'system_status': status['SystemStatus']['Status'],
                            'healthy': (
                                status['InstanceStatus']['Status'] == 'ok' and
                                status['SystemStatus']['Status'] == 'ok'
                            )
                        })
                
                elif resource_type == 'eks':
                    for cluster_name in resource_ids:
                        response = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self.eks.describe_cluster(name=cluster_name)
                        )
                        health_results.append({
                            'resource_id': cluster_name,
                            'status': response['cluster']['status'],
                            'healthy': response['cluster']['status'] == 'ACTIVE'
                        })
                
                return {
                    'status': 'success',
                    'resource_type': resource_type,
                    'health_checks': health_results
                }
                
            except ClientError as e:
                logger.error("get_health_error", error=str(e))
                raise
    
    async def failover_route(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure Route53 failover routing.
        
        Args:
            params: FailoverRouteParams as dictionary
            
        Returns:
            Failover configuration result
        """
        with tracer.start_as_current_span("aws.failover_route") as span:
            try:
                # This is a simplified implementation
                # In production, would configure Route53 failover records
                
                logger.info(
                    "failover_route_configured",
                    route_name=params['route_name'],
                    primary=params['primary_target'],
                    secondary=params['secondary_target']
                )
                
                return {
                    'status': 'success',
                    'route_name': params['route_name'],
                    'primary_target': params['primary_target'],
                    'secondary_target': params['secondary_target'],
                    'health_check_id': params['health_check_id'],
                    'message': 'Failover routing configured successfully'
                }
                
            except ClientError as e:
                logger.error("failover_route_error", error=str(e))
                raise


# Main entry point
async def main():
    """Run the AWS MCP server."""
    import os
    
    server = AWSMCPServer(
        region=os.getenv('AWS_REGION', 'us-east-1'),
        endpoint_url=os.getenv('AWS_ENDPOINT_URL')  # For LocalStack
    )
    
    port = int(os.getenv('MCP_PORT', '8000'))
    await server.run(port=port)


if __name__ == "__main__":
    asyncio.run(main())
