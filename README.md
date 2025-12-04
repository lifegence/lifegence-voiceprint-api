# Lifegence Voiceprint API

A simple, self-hosted voiceprint authentication REST API. Provides speaker verification (1:1) and identification (1:N) capabilities.

## Features

| Feature | Description | Endpoint |
|---------|-------------|----------|
| **Enroll** | Register a speaker's voiceprint | `POST /v1/speakers/enroll` |
| **Verify** | 1:1 authentication against enrolled speaker | `POST /v1/speakers/verify` |
| **Identify** | 1:N search across all speakers | `POST /v1/speakers/identify` |
| **Delete** | Remove speaker data (GDPR compliance) | `DELETE /v1/speakers/{id}` |

## Quick Start

### 1. Start the API Server

```bash
# Clone the repository
git clone https://github.com/lifegence/voiceprint-api.git
cd voiceprint-api

# Start with Docker Compose
docker-compose up -d

# Health check
curl http://localhost:8000/v1/health
```

### 2. Enroll a Speaker

```bash
curl -X POST http://localhost:8000/v1/speakers/enroll \
  -H "X-API-Key: your-api-key" \
  -F "audio=@voice_sample.wav" \
  -F "speaker_id=user-123" \
  -F "consent_granted=true" \
  -F "consent_timestamp=2024-01-15T10:00:00Z" \
  -F "consent_purpose=voice_authentication"
```

### 3. Verify a Speaker

```bash
curl -X POST http://localhost:8000/v1/speakers/verify \
  -H "X-API-Key: your-api-key" \
  -F "audio=@voice_sample.wav" \
  -F "speaker_id=user-123"
```

Response:
```json
{
  "speaker_id": "user-123",
  "is_verified": true,
  "score": 0.89,
  "threshold": 0.7,
  "confidence": "high"
}
```

### 4. Identify a Speaker

```bash
curl -X POST http://localhost:8000/v1/speakers/identify \
  -H "X-API-Key: your-api-key" \
  -F "audio=@voice_sample.wav" \
  -F "max_results=5"
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | `your-secure-api-key` | API authentication key |
| `DATABASE_URL` | (docker default) | PostgreSQL connection string |
| `REDIS_URL` | (docker default) | Redis connection string |
| `LOG_LEVEL` | `INFO` | Logging level |

### Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Voiceprint Engine | SpeechBrain ECAPA-TDNN (192-dim embeddings) |
| API Framework | FastAPI (Python 3.11) |
| Database | PostgreSQL 16 + pgvector |
| Cache | Redis 7 |
| Container | Docker + docker-compose |

## Audio Requirements

| Parameter | Requirement |
|-----------|-------------|
| Format | WAV, MP3, WebM, OGG |
| Duration | 3-30 seconds (5-10 recommended) |
| Sample Rate | Any (resampled to 16kHz) |
| Channels | Any (converted to mono) |

## Score Interpretation

| Score | Confidence | Meaning |
|-------|------------|---------|
| 0.90+ | `very_high` | Very likely same speaker |
| 0.80-0.90 | `high` | Likely same speaker |
| 0.70-0.80 | `medium` | Possibly same speaker |
| 0.50-0.70 | `low` | Uncertain |
| <0.50 | `very_low` | Likely different speaker |

## Development

### Local Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest -v --cov=app

# Start development server
uvicorn app.main:app --reload
```

### Project Structure

```
voiceprint-api/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── core/
│   │   │   ├── audio_processor.py
│   │   │   └── embedding_extractor.py
│   │   └── services/
│   │       └── speaker_service.py
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── docs/
│   ├── API.md                   # API specification
│   ├── ARCHITECTURE.md          # System architecture
│   └── TECHNICAL.md             # Voice embedding details
├── docker-compose.yml
└── README.md
```

## Documentation

- [API Specification](docs/API.md) - Endpoints, request/response formats
- [Architecture](docs/ARCHITECTURE.md) - System design and components
- [Technical Details](docs/TECHNICAL.md) - ECAPA-TDNN, embeddings, similarity

## Why Self-Hosted?

Major cloud vendors are discontinuing voice recognition services:

| Service | Status |
|---------|--------|
| AWS Amazon Connect Voice ID | End of support: May 2026 |
| Azure Speaker Recognition | Deprecation: September 2025 |

**Self-hosted benefits:**
- Data stays on your servers
- Full control over compliance (GDPR, BIPA)
- No vendor lock-in

## License

MIT License - See [LICENSE](LICENSE)

## Disclaimer

- Compliance with applicable laws (GDPR, BIPA, etc.) is the user's responsibility
- Voice recognition does not guarantee 100% accuracy
- Implement additional security measures for high-security applications

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security

Report vulnerabilities to security@lifegence.com. See [SECURITY.md](SECURITY.md).
