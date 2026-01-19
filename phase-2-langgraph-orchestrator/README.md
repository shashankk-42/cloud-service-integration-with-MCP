# Phase 2: LangGraph Orchestrator MVP

## Objective
Build the LangGraph agent runtime with Planner, Allocator, Executor, Verifier, and Finisher workflow graphs.

## Duration: 2-3 weeks

## Agent Workflow Architecture

```
┌─────────────┐     ┌───────────────┐     ┌────────────┐     ┌────────────┐     ┌───────────┐
│   Planner   │ ──▶ │   Allocator   │ ──▶ │  Executor  │ ──▶ │  Verifier  │ ──▶ │  Finisher │
└─────────────┘     └───────────────┘     └────────────┘     └────────────┘     └───────────┘
      │                    │                    │                   │                  │
      │ classify           │ cost/quota         │ provision         │ health           │ cleanup
      │ workload           │ analysis           │ resources         │ checks           │ attribution
      │ select clouds      │ spot/ondemand      │ submit jobs       │ SLO verify       │ teardown
      └────────────────────┴────────────────────┴───────────────────┴──────────────────┘
                                        │
                                   State Store
                                (Postgres/Redis)
```

## Components

### 1. Planner Agent
- Interprets task requirements
- Classifies workload (latency vs batch, GPU vs CPU)
- Selects clouds/regions based on SLO, cost, data residency

### 2. Allocator Agent
- Queries MCP `get_cost_estimate` and `get_quotas`
- Picks mix of on-demand vs spot/preemptible
- Chooses cluster/zone
- Schedules start times for batch

### 3. Executor Agent
- Invokes MCP provisioning tools
- Submits jobs or deploys models
- Attaches storage
- Configures network policies

### 4. Verifier Agent
- Health checks via `get_health`
- Compares against SLOs
- Triggers rollbacks or failover routes

### 5. Finisher Agent
- Teardown/cleanup
- Idle timers
- Cost attribution tagging
- Emits run records

## Reference Workflows

1. **Latency Service Deploy** - Deploy a model serving endpoint with auto-scaling
2. **Batch Job Submit** - Submit an ML training job with checkpointing

## Exit Criteria
- [ ] End-to-end workflow runs in sandbox
- [ ] Decision traces are replayable
- [ ] Failure-path tests pass (spot loss, quota denial)
- [ ] Dashboard shows agent decisions
