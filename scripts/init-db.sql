-- Database Initialization Script
-- Cloud Service Integration with MCP

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ===================
-- Schema: orchestrator
-- ===================
CREATE SCHEMA IF NOT EXISTS orchestrator;

-- Task Requests Table
CREATE TABLE IF NOT EXISTS orchestrator.task_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    workload_type VARCHAR(50) NOT NULL,
    resource_requirements JSONB DEFAULT '{}',
    slo_requirements JSONB DEFAULT '{}',
    budget_limit DECIMAL(10, 2) DEFAULT 0.00,
    preferred_clouds VARCHAR(50)[] DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Workflow State Table
CREATE TABLE IF NOT EXISTS orchestrator.workflow_states (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id VARCHAR(255) NOT NULL REFERENCES orchestrator.task_requests(task_id),
    current_phase VARCHAR(50) NOT NULL,
    plan JSONB,
    allocation JSONB,
    execution_results JSONB DEFAULT '[]',
    verification_results JSONB DEFAULT '[]',
    final_output JSONB,
    error TEXT,
    retry_count INTEGER DEFAULT 0,
    should_rollback BOOLEAN DEFAULT FALSE,
    checkpoint_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Execution Logs Table
CREATE TABLE IF NOT EXISTS orchestrator.execution_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id VARCHAR(255) NOT NULL REFERENCES orchestrator.task_requests(task_id),
    step_id VARCHAR(255),
    agent_name VARCHAR(100),
    action VARCHAR(255),
    cloud_provider VARCHAR(50),
    input_params JSONB,
    output_result JSONB,
    duration_ms INTEGER,
    status VARCHAR(50),
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Cost Tracking Table
CREATE TABLE IF NOT EXISTS orchestrator.cost_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id VARCHAR(255) REFERENCES orchestrator.task_requests(task_id),
    project_id VARCHAR(255),
    cloud_provider VARCHAR(50) NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    cost_usd DECIMAL(10, 4) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    usage_quantity DECIMAL(10, 4),
    usage_unit VARCHAR(50),
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    tags JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Policy Decisions Table
CREATE TABLE IF NOT EXISTS orchestrator.policy_decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id VARCHAR(255) REFERENCES orchestrator.task_requests(task_id),
    request_id VARCHAR(255),
    policy_name VARCHAR(255),
    decision VARCHAR(50) NOT NULL,  -- 'allow', 'deny', 'require_approval'
    reason TEXT,
    user_id VARCHAR(255),
    user_role VARCHAR(100),
    resource_type VARCHAR(100),
    action VARCHAR(255),
    context JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Approval Requests Table
CREATE TABLE IF NOT EXISTS orchestrator.approval_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id VARCHAR(255) REFERENCES orchestrator.task_requests(task_id),
    requester_id VARCHAR(255) NOT NULL,
    approval_type VARCHAR(100) NOT NULL,  -- 'budget', 'security', 'compliance'
    requested_action TEXT NOT NULL,
    estimated_cost DECIMAL(10, 2),
    status VARCHAR(50) DEFAULT 'pending',  -- 'pending', 'approved', 'rejected'
    approver_id VARCHAR(255),
    approved_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ===================
-- Schema: audit
-- ===================
CREATE SCHEMA IF NOT EXISTS audit;

-- Audit Log Table
CREATE TABLE IF NOT EXISTS audit.logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    actor_id VARCHAR(255),
    actor_role VARCHAR(100),
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    action VARCHAR(255) NOT NULL,
    cloud_provider VARCHAR(50),
    region VARCHAR(100),
    request_data JSONB,
    response_data JSONB,
    ip_address INET,
    user_agent TEXT,
    status VARCHAR(50),
    error_message TEXT,
    duration_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ===================
-- Indexes
-- ===================

-- Task Requests Indexes
CREATE INDEX IF NOT EXISTS idx_task_requests_status ON orchestrator.task_requests(status);
CREATE INDEX IF NOT EXISTS idx_task_requests_created_at ON orchestrator.task_requests(created_at);
CREATE INDEX IF NOT EXISTS idx_task_requests_workload_type ON orchestrator.task_requests(workload_type);

-- Workflow States Indexes
CREATE INDEX IF NOT EXISTS idx_workflow_states_task_id ON orchestrator.workflow_states(task_id);
CREATE INDEX IF NOT EXISTS idx_workflow_states_phase ON orchestrator.workflow_states(current_phase);

-- Execution Logs Indexes
CREATE INDEX IF NOT EXISTS idx_execution_logs_task_id ON orchestrator.execution_logs(task_id);
CREATE INDEX IF NOT EXISTS idx_execution_logs_created_at ON orchestrator.execution_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_execution_logs_cloud_provider ON orchestrator.execution_logs(cloud_provider);

-- Cost Records Indexes
CREATE INDEX IF NOT EXISTS idx_cost_records_task_id ON orchestrator.cost_records(task_id);
CREATE INDEX IF NOT EXISTS idx_cost_records_project_id ON orchestrator.cost_records(project_id);
CREATE INDEX IF NOT EXISTS idx_cost_records_cloud_provider ON orchestrator.cost_records(cloud_provider);
CREATE INDEX IF NOT EXISTS idx_cost_records_created_at ON orchestrator.cost_records(created_at);

-- Audit Logs Indexes
CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON audit.logs(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_actor_id ON audit.logs(actor_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit.logs(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit.logs(action);

-- ===================
-- Functions
-- ===================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers
CREATE TRIGGER update_task_requests_updated_at
    BEFORE UPDATE ON orchestrator.task_requests
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_workflow_states_updated_at
    BEFORE UPDATE ON orchestrator.workflow_states
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ===================
-- Views
-- ===================

-- Task Summary View
CREATE OR REPLACE VIEW orchestrator.task_summary AS
SELECT 
    tr.task_id,
    tr.name,
    tr.workload_type,
    tr.status,
    tr.budget_limit,
    ws.current_phase,
    ws.retry_count,
    COALESCE(SUM(cr.cost_usd), 0) as total_cost,
    COUNT(DISTINCT el.id) as execution_steps,
    tr.created_at,
    tr.completed_at,
    EXTRACT(EPOCH FROM (COALESCE(tr.completed_at, CURRENT_TIMESTAMP) - tr.created_at)) as duration_seconds
FROM orchestrator.task_requests tr
LEFT JOIN orchestrator.workflow_states ws ON tr.task_id = ws.task_id
LEFT JOIN orchestrator.cost_records cr ON tr.task_id = cr.task_id
LEFT JOIN orchestrator.execution_logs el ON tr.task_id = el.task_id
GROUP BY tr.task_id, tr.name, tr.workload_type, tr.status, tr.budget_limit,
         ws.current_phase, ws.retry_count, tr.created_at, tr.completed_at;

-- Daily Cost Summary View
CREATE OR REPLACE VIEW orchestrator.daily_cost_summary AS
SELECT 
    DATE(created_at) as date,
    cloud_provider,
    project_id,
    resource_type,
    SUM(cost_usd) as total_cost,
    COUNT(*) as resource_count
FROM orchestrator.cost_records
GROUP BY DATE(created_at), cloud_provider, project_id, resource_type;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA orchestrator TO postgres;
GRANT ALL PRIVILEGES ON SCHEMA audit TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA orchestrator TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA audit TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA orchestrator TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA audit TO postgres;

-- Insert sample data for testing (optional)
-- INSERT INTO orchestrator.task_requests (task_id, name, description, workload_type, status)
-- VALUES ('test-001', 'Sample Task', 'A sample task for testing', 'batch', 'pending');
