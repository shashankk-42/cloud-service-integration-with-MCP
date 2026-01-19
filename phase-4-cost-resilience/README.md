# Phase 4: Cost & Resilience Enhancements

## Objective
Implement cost optimization strategies including cross-cloud arbitrage, spot/preemptible management, and resilience patterns for failover.

## Duration: 2-3 weeks

## Components

### 1. Cost Optimization
- Cross-cloud arbitrage in allocator
- Spot/preemptible instance preference with on-demand fallback
- TTL garbage collection for idle resources
- Budget guardrails and cost alerts

### 2. Spot Instance Management
- Interruption-aware checkpointing
- Automatic fallback to on-demand
- Spot capacity tracking
- Instance diversification

### 3. Cost Attribution
- Per-project budgets
- Real-time cost tracking
- Cost anomaly detection
- Chargeback/showback reporting

### 4. Resilience
- Multi-region failover
- Circuit breakers
- Health-based routing
- Graceful degradation

## Exit Criteria
- [ ] Batch jobs shift to cheapest compliant region/time window
- [ ] Enforced deny on projected budget overrun
- [ ] Idle resource cleanup reduces waste
- [ ] Spot interruption handling works correctly
