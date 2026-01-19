# Phase 5: Security, Compliance & IAM Hardening

## Objective
Implement comprehensive security controls including IAM, audit logging, data residency enforcement, and compliance features.

## Duration: 2-3 weeks

## Components

### 1. IAM & Access Control
- Least-privilege role sets per cloud
- Just-in-time (JIT) elevation with approval
- Workload identity (OIDC/SPIFFE)
- Short-lived credentials

### 2. Audit & Compliance
- Comprehensive audit trail (who/what/when/where)
- Immutable log storage
- SIEM integration
- SOC2/GDPR compliance controls

### 3. Data Protection
- TLS everywhere (mutual auth)
- Envelope encryption
- Customer-managed keys
- DLP scanning on outputs

### 4. Policy Enforcement
- Data residency in planner
- OPA policy engine integration
- Retention policies
- Access reviews

## RBAC Matrix

| Role | MCP Tools | Approval Required |
|------|-----------|-------------------|
| Developer | get_*, submit_ml_job | No |
| Operator | provision_*, scale_* | No |
| Admin | rotate_secret, failover_route | Yes |
| Security | All read-only | No |

## Exit Criteria
- [ ] Access review audit passed
- [ ] Audit trail completeness verified
- [ ] Data residency policy blocks non-compliant placement
- [ ] Secrets rotation working via MCP
