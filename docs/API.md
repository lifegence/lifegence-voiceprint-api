# Lifegence Voiceprint API - API Specification

## Overview

| Item | Value |
|------|-------|
| Base URL | `https://api.example.com/v1` |
| Authentication | API Key (Header: `X-API-Key`) |
| Content-Type | `multipart/form-data` (audio), `application/json` (others) |
| Response | `application/json` |

## Authentication

All endpoints require API Key authentication.

```http
X-API-Key: your-api-key-here
```

### Error Response (Authentication Failure)

```json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or missing API key"
  }
}
```

## Endpoints

---

### POST /speakers/enroll

Enrolls a speaker and generates a voiceprint (embedding).

#### Request

```http
POST /v1/speakers/enroll
Content-Type: multipart/form-data
X-API-Key: your-api-key
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | file | Yes | Audio file (WAV/WebM/OGG, 3-30 seconds) |
| `speaker_id` | string | Yes | Client-side speaker ID |
| `consent_granted` | boolean | Yes | Consent obtained flag |
| `consent_timestamp` | string | Yes | Consent timestamp (ISO 8601) |
| `consent_purpose` | string | Yes | Purpose of use |
| `metadata` | json | No | Additional metadata |

#### Example Request

```bash
curl -X POST https://api.example.com/v1/speakers/enroll \
  -H "X-API-Key: your-api-key" \
  -F "audio=@voice_sample.wav" \
  -F "speaker_id=user-123" \
  -F "consent_granted=true" \
  -F "consent_timestamp=2024-01-15T10:30:00Z" \
  -F "consent_purpose=voice_authentication" \
  -F 'metadata={"name": "John Doe"}'
```

#### Response (Success: 201)

```json
{
  "enrollment_id": "550e8400-e29b-41d4-a716-446655440000",
  "speaker_id": "user-123",
  "status": "enrolled",
  "quality_score": 0.85,
  "audio_duration": 5.2,
  "created_at": "2024-01-15T10:30:15Z"
}
```

#### Response (Error: 400)

```json
{
  "error": {
    "code": "INVALID_AUDIO",
    "message": "Audio duration must be between 3 and 30 seconds",
    "details": {
      "actual_duration": 2.1,
      "min_duration": 3.0,
      "max_duration": 30.0
    }
  }
}
```

---

### POST /speakers/verify

Compares audio against a registered speaker for identity verification (1:1 authentication).

#### Request

```http
POST /v1/speakers/verify
Content-Type: multipart/form-data
X-API-Key: your-api-key
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | file | Yes | Audio file (WAV/WebM/OGG, 3-30 seconds) |
| `speaker_id` | string | Yes | Target speaker ID for verification |
| `threshold` | float | No | Verification threshold (default: 0.7) |

#### Example Request

```bash
curl -X POST https://api.example.com/v1/speakers/verify \
  -H "X-API-Key: your-api-key" \
  -F "audio=@voice_sample.wav" \
  -F "speaker_id=user-123" \
  -F "threshold=0.7"
```

#### Response (Success: 200)

```json
{
  "speaker_id": "user-123",
  "is_verified": true,
  "score": 0.89,
  "threshold": 0.7,
  "confidence": "high",
  "processing_time_ms": 156
}
```

#### Score Interpretation

| Score | Confidence | Description |
|-------|------------|-------------|
| 0.9+ | very_high | Very high confidence |
| 0.8-0.9 | high | High confidence |
| 0.7-0.8 | medium | Medium confidence |
| 0.5-0.7 | low | Low confidence (additional authentication recommended) |
| < 0.5 | very_low | Very low (likely different person) |

---

### POST /speakers/identify

Identifies a speaker from audio (1:N search).

#### Request

```http
POST /v1/speakers/identify
Content-Type: multipart/form-data
X-API-Key: your-api-key
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | file | Yes | Audio file (WAV/WebM/OGG, 3-30 seconds) |
| `max_results` | int | No | Maximum number of results (default: 5, max: 20) |
| `threshold` | float | No | Minimum similarity (default: 0.5) |
| `group_id` | string | No | Target search group |

#### Example Request

```bash
curl -X POST https://api.example.com/v1/speakers/identify \
  -H "X-API-Key: your-api-key" \
  -F "audio=@voice_sample.wav" \
  -F "max_results=5" \
  -F "threshold=0.5"
```

#### Response (Success: 200)

```json
{
  "matches": [
    {
      "speaker_id": "user-123",
      "score": 0.92,
      "confidence": "very_high",
      "metadata": {"name": "John Doe"}
    },
    {
      "speaker_id": "user-456",
      "score": 0.67,
      "confidence": "low",
      "metadata": {"name": "Jane Smith"}
    }
  ],
  "total_searched": 1523,
  "processing_time_ms": 234
}
```

#### Response (No Match: 200)

```json
{
  "matches": [],
  "total_searched": 1523,
  "processing_time_ms": 198
}
```

---

### GET /speakers/{speaker_id}

Retrieves speaker information.

#### Request

```http
GET /v1/speakers/{speaker_id}
X-API-Key: your-api-key
```

#### Response (Success: 200)

```json
{
  "speaker_id": "user-123",
  "enrollment_id": "550e8400-e29b-41d4-a716-446655440000",
  "metadata": {"name": "John Doe"},
  "enrollment_count": 3,
  "last_verified_at": "2024-01-15T14:20:00Z",
  "created_at": "2024-01-15T10:30:15Z",
  "updated_at": "2024-01-15T14:20:00Z"
}
```

#### Response (Not Found: 404)

```json
{
  "error": {
    "code": "SPEAKER_NOT_FOUND",
    "message": "Speaker with ID 'user-123' not found"
  }
}
```

---

### DELETE /speakers/{speaker_id}

Deletes a speaker and all associated voiceprint data (GDPR "Right to be Forgotten" compliance).

#### Request

```http
DELETE /v1/speakers/{speaker_id}
X-API-Key: your-api-key
```

#### Response (Success: 200)

```json
{
  "speaker_id": "user-123",
  "status": "deleted",
  "deleted_embeddings": 3,
  "deleted_at": "2024-01-15T16:00:00Z"
}
```

---

### PUT /speakers/{speaker_id}/embedding

Updates an existing speaker's voiceprint (additional enrollment).

#### Request

```http
PUT /v1/speakers/{speaker_id}/embedding
Content-Type: multipart/form-data
X-API-Key: your-api-key
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | file | Yes | Audio file |
| `replace` | boolean | No | Replace existing (default: false = add) |

#### Response (Success: 200)

```json
{
  "speaker_id": "user-123",
  "embedding_id": "660e8400-e29b-41d4-a716-446655440001",
  "status": "updated",
  "total_embeddings": 4,
  "quality_score": 0.88
}
```

---

### GET /health

Service health check.

#### Request

```http
GET /v1/health
```

#### Response (Success: 200)

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "database": "healthy",
    "redis": "healthy",
    "model": "loaded"
  },
  "timestamp": "2024-01-15T10:00:00Z"
}
```

---

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | Invalid or missing API Key |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `SPEAKER_NOT_FOUND` | 404 | Speaker does not exist |
| `SPEAKER_ALREADY_EXISTS` | 409 | Speaker ID already registered |
| `INVALID_AUDIO` | 400 | Invalid audio file |
| `AUDIO_TOO_SHORT` | 400 | Audio too short (< 3 seconds) |
| `AUDIO_TOO_LONG` | 400 | Audio too long (> 30 seconds) |
| `LOW_AUDIO_QUALITY` | 400 | Low audio quality |
| `CONSENT_REQUIRED` | 400 | Missing consent information |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded |
| `INTERNAL_ERROR` | 500 | Internal server error |

## Rate Limiting

| Endpoint | Limit |
|----------|-------|
| POST /speakers/enroll | 10 req/min |
| POST /speakers/verify | 60 req/min |
| POST /speakers/identify | 30 req/min |
| Others | 100 req/min |

Response when exceeded:

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again in 45 seconds.",
    "retry_after": 45
  }
}
```

## Audio File Requirements

| Item | Requirement |
|------|-------------|
| Format | WAV, WebM, OGG, MP3 |
| Sample Rate | 16kHz recommended (auto-resampled) |
| Channels | Mono recommended (auto-converted) |
| Duration | 3 to 30 seconds |
| Recommended Duration | 5 to 10 seconds |

## WebSocket API (Real-time)

WebSocket API for real-time audio streaming.

### Connection

```
wss://api.example.com/v1/ws/stream
```

### Message Format

#### Client → Server

```json
// Start
{"type": "start", "mode": "verify", "speaker_id": "user-123"}

// Audio chunk
{"type": "audio", "data": "<base64-encoded-audio-chunk>"}

// End
{"type": "end"}
```

#### Server → Client

```json
// Progress
{"type": "progress", "audio_duration": 3.5, "status": "processing"}

// Result
{"type": "result", "is_verified": true, "score": 0.89}

// Error
{"type": "error", "code": "AUDIO_TOO_SHORT", "message": "..."}
```
