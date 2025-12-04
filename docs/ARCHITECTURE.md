# Lifegence Voiceprint API - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Clients                                  │
│         (curl, Python, JavaScript, Mobile Apps, etc.)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTPS (REST API)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Voiceprint API (FastAPI)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   REST API  │  │    Auth     │  │     Rate Limiter        │  │
│  │  Endpoints  │  │  Middleware │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Voice Processing Service                    │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌─────────────┐  │    │
│  │  │ Audio Preproc │  │   Embedding   │  │  Similarity │  │    │
│  │  │  (librosa)    │  │  (SpeechBrain)│  │   Search    │  │    │
│  │  └───────────────┘  └───────────────┘  └─────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Data Layer                            │    │
│  │  ┌───────────────────────┐  ┌─────────────────────────┐ │    │
│  │  │  PostgreSQL + pgvector│  │    Redis (Cache)        │ │    │
│  │  │  - speakers           │  │    - rate limits        │ │    │
│  │  │  - embeddings         │  │    - sessions           │ │    │
│  │  └───────────────────────┘  └─────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. REST API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/speakers/enroll` | Register speaker voiceprint |
| POST | `/v1/speakers/verify` | Verify speaker (1:1) |
| POST | `/v1/speakers/identify` | Identify speaker (1:N) |
| GET | `/v1/speakers/{id}` | Get speaker info |
| DELETE | `/v1/speakers/{id}` | Delete speaker |
| PUT | `/v1/speakers/{id}/embedding` | Update voiceprint |
| GET | `/v1/health` | Health check |

### 2. Authentication

API Key authentication via `X-API-Key` header.

```python
async def verify_api_key(x_api_key: str = Header(...)) -> str:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key
```

### 3. Voice Processing Service

#### Audio Preprocessor

```python
class AudioProcessor:
    SAMPLE_RATE = 16000
    MIN_DURATION = 3.0   # seconds
    MAX_DURATION = 30.0  # seconds

    def preprocess(self, audio_bytes: bytes) -> np.ndarray:
        """
        1. Decode (WAV/MP3/WebM/OGG)
        2. Resample to 16kHz
        3. Convert to mono
        4. Normalize amplitude
        5. Validate duration
        """
```

#### Embedding Extractor

```python
class EmbeddingExtractor:
    MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"
    EMBEDDING_DIM = 192

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract 192-dimensional embedding from audio"""
        embedding = self.model.encode_batch(audio_tensor)
        return embedding / np.linalg.norm(embedding)  # L2 normalize
```

#### Similarity Search

```python
def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Cosine similarity for L2-normalized vectors: dot(a, b)"""
    return float(np.dot(embedding1, embedding2))

# Thresholds
VERIFICATION_THRESHOLD = 0.7   # For 1:1 verification
IDENTIFICATION_THRESHOLD = 0.5  # For 1:N search
```

### 4. Data Layer

#### PostgreSQL Schema

```sql
-- Speakers table
CREATE TABLE speakers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id VARCHAR(255) UNIQUE NOT NULL,
    metadata JSONB,
    consent_info JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Embeddings table (with pgvector)
CREATE TABLE embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    speaker_id UUID REFERENCES speakers(id) ON DELETE CASCADE,
    embedding vector(192) NOT NULL,
    quality_score FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Fast similarity search index
CREATE INDEX ON embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

#### Redis Cache

```
# Rate limiting
rate_limit:{client_id}:{endpoint} -> count (TTL: 1 minute)

# Embedding cache (optional)
embedding_cache:{audio_hash} -> embedding (TTL: 5 minutes)
```

## Processing Flows

### Enrollment Flow

```
1. Client sends audio + speaker_id + consent_info
2. API validates request and authentication
3. AudioProcessor validates and preprocesses audio
4. EmbeddingExtractor generates 192-dim vector
5. Store speaker and embedding in PostgreSQL
6. Return enrollment_id and quality_score
```

### Verification Flow (1:1)

```
1. Client sends audio + speaker_id
2. API retrieves speaker's stored embedding
3. Extract embedding from input audio
4. Calculate cosine similarity
5. Return score and is_verified (score >= threshold)
```

### Identification Flow (1:N)

```
1. Client sends audio
2. Extract embedding from input audio
3. pgvector similarity search across all embeddings
4. Return ranked matches above threshold
```

## Security

### Rate Limiting

| Endpoint | Limit |
|----------|-------|
| `/v1/speakers/enroll` | 10 req/min |
| `/v1/speakers/verify` | 60 req/min |
| `/v1/speakers/identify` | 30 req/min |

### Data Protection

| Layer | Protection |
|-------|------------|
| In Transit | TLS 1.3 |
| At Rest | PostgreSQL encryption |
| API Access | API Key authentication |

## Performance Targets

| Metric | Target |
|--------|--------|
| Embedding extraction | < 500ms (CPU) |
| Verification (1:1) | < 200ms |
| Identification (1:10000) | < 500ms |
| Concurrent requests | 100+ |

## Deployment Options

### Docker Compose (Default)

```yaml
services:
  api:       # FastAPI application
  db:        # PostgreSQL + pgvector
  redis:     # Cache and rate limiting
```

### Kubernetes (Production)

```
┌─────────────┐
│   Ingress   │
└──────┬──────┘
       │
┌──────┴──────┐
│   Service   │
└──────┬──────┘
       │
┌──────┴──────────────────┐
│    Deployment (3+ pods) │
└─────────────────────────┘
       │
┌──────┴──────────────────┐
│  PostgreSQL (StatefulSet)│
│  Redis (StatefulSet)     │
└─────────────────────────┘
```

### GPU Acceleration (Optional)

For high-throughput deployments:

```yaml
services:
  api:
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```
