-- Initialize Lifegence VoiceID Database

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Speakers table
CREATE TABLE IF NOT EXISTS speakers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id VARCHAR(255) UNIQUE NOT NULL,
    metadata JSONB,
    consent_info JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);

-- Embeddings table
CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    speaker_id UUID REFERENCES speakers(id) ON DELETE CASCADE,
    embedding vector(192) NOT NULL,
    audio_hash VARCHAR(64) NOT NULL,
    quality_score FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Audit logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    action VARCHAR(50) NOT NULL,
    speaker_id UUID,
    client_id VARCHAR(255) NOT NULL,
    result VARCHAR(20) NOT NULL,
    metadata JSONB,
    ip_address INET,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_speakers_external_id ON speakers(external_id);
CREATE INDEX IF NOT EXISTS idx_speakers_deleted_at ON speakers(deleted_at);
CREATE INDEX IF NOT EXISTS idx_embeddings_speaker_id ON embeddings(speaker_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_logs_speaker_id ON audit_logs(speaker_id);

-- IVFFlat index for vector similarity search
-- Note: This should be created after inserting initial data for better performance
-- CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings
-- USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create test database if not exists (for testing)
SELECT 'CREATE DATABASE voiceid_test'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'voiceid_test')\gexec

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO voiceid;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO voiceid;
