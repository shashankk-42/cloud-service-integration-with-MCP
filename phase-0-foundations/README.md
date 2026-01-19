# Phase 0: Foundations

## Objective
Establish the foundational infrastructure including cloud landing zones, CI/CD pipelines, observability stack, secrets management, and service mesh with mTLS.

## Duration: 1-2 weeks

## Deliverables

1. **Cloud Landing Zones (Sandbox)**
   - Terraform/Bicep configurations for AWS, Azure, GCP sandboxes
   - Network isolation and VPC/VNet setup
   - IAM baseline roles and policies

2. **CI/CD Skeleton**
   - GitHub Actions workflow templates
   - Pipeline for lint, test, build, deploy
   - Environment promotion (dev → staging → prod)

3. **Observability Bootstrap**
   - OpenTelemetry Collector deployment
   - Log aggregation sink (ELK/CloudWatch/Stackdriver)
   - Basic metrics and dashboards

4. **Secrets Backbone**
   - External Secrets Operator setup
   - Integration with cloud KMS/Secret Manager
   - Rotation policies

5. **Service Mesh & mTLS**
   - Istio/Linkerd deployment
   - PKI infrastructure
   - Certificate management

## Exit Criteria
- [ ] mTLS working between two sample services
- [ ] CI runs fmt/lint/tests successfully
- [ ] Cost/budget alerts firing in sandbox
- [ ] Secrets accessible via External Secrets Operator
