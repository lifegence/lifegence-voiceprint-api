# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- WebSocket API for real-time streaming verification
- Batch enrollment API
- Liveness detection

## [0.1.0] - 2024-12-04

### Added

#### REST API Endpoints
- `POST /v1/speakers/enroll` - Register speaker voiceprint
- `POST /v1/speakers/verify` - 1:1 speaker verification
- `POST /v1/speakers/identify` - 1:N speaker identification
- `GET /v1/speakers/{id}` - Get speaker information
- `DELETE /v1/speakers/{id}` - Delete speaker (GDPR compliance)
- `PUT /v1/speakers/{id}/embedding` - Update voiceprint
- `GET /v1/health` - Service health check

#### Core Components
- **EmbeddingExtractor**: SpeechBrain ECAPA-TDNN integration
  - 192-dimensional speaker embeddings
  - L2 normalization for cosine similarity
  - CPU/CUDA device support

- **AudioProcessor**: Audio validation and preprocessing
  - Format support: WAV, MP3, WebM, OGG
  - Automatic resampling to 16kHz mono
  - Duration validation (3-30 seconds)
  - Quality scoring

- **SpeakerService**: Business logic layer
  - PostgreSQL + pgvector support
  - Redis caching integration

#### Infrastructure
- Docker Compose setup (API, PostgreSQL, Redis)
- GitHub Actions CI/CD pipeline
- Pre-configured linting (Ruff, Black, mypy)
- Test coverage requirements (80% minimum)

#### Documentation
- API specification (docs/API.md)
- Architecture overview (docs/ARCHITECTURE.md)
- Technical details (docs/TECHNICAL.md)

### Security
- API key authentication
- Consent validation enforcement
- Biometric data encryption at rest

### Compliance
- GDPR Article 9 considerations
- Illinois BIPA awareness
- Data deletion API for right to erasure

---

[Unreleased]: https://github.com/lifegence/voiceprint-api/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/lifegence/voiceprint-api/releases/tag/v0.1.0
