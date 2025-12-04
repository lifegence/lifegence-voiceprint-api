"""
TDD Tests for API Endpoints.

These tests define the expected API behavior BEFORE implementation.
Run with: pytest tests/test_api_endpoints.py -v
"""

import pytest
from unittest.mock import patch, AsyncMock


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_check_returns_200(self, client, api_key_header):
        """Test health endpoint returns 200."""
        response = client.get("/v1/health")

        assert response.status_code == 200

    def test_health_check_returns_status(self, client):
        """Test health endpoint returns status information."""
        response = client.get("/v1/health")
        data = response.json()

        assert "status" in data
        assert data["status"] == "healthy"
        assert "version" in data


class TestAuthMiddleware:
    """Test API authentication."""

    def test_missing_api_key_returns_401(self, client):
        """Test request without API key returns 401."""
        response = client.post("/v1/speakers/enroll")

        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHORIZED"

    def test_invalid_api_key_returns_401(self, client):
        """Test request with invalid API key returns 401."""
        response = client.post(
            "/v1/speakers/enroll",
            headers={"X-API-Key": "invalid-key"},
        )

        assert response.status_code == 401

    def test_valid_api_key_passes(self, client, api_key_header, sample_audio_bytes):
        """Test request with valid API key passes authentication."""
        response = client.post(
            "/v1/speakers/enroll",
            headers=api_key_header,
            files={"audio": ("test.wav", sample_audio_bytes, "audio/wav")},
            data={
                "speaker_id": "test-speaker",
                "consent_granted": "true",
                "consent_timestamp": "2024-01-15T10:30:00Z",
                "consent_purpose": "voice_authentication",
            },
        )

        # Should not be 401
        assert response.status_code != 401


class TestEnrollEndpoint:
    """Test POST /speakers/enroll endpoint."""

    def test_enroll_success(self, client, api_key_header, sample_audio_bytes):
        """Test successful speaker enrollment."""
        response = client.post(
            "/v1/speakers/enroll",
            headers=api_key_header,
            files={"audio": ("test.wav", sample_audio_bytes, "audio/wav")},
            data={
                "speaker_id": "test-speaker-001",
                "consent_granted": "true",
                "consent_timestamp": "2024-01-15T10:30:00Z",
                "consent_purpose": "voice_authentication",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert "enrollment_id" in data
        assert data["speaker_id"] == "test-speaker-001"
        assert data["status"] == "enrolled"
        assert "quality_score" in data

    def test_enroll_missing_audio(self, client, api_key_header):
        """Test enrollment without audio file returns 400."""
        response = client.post(
            "/v1/speakers/enroll",
            headers=api_key_header,
            data={
                "speaker_id": "test-speaker",
                "consent_granted": "true",
                "consent_timestamp": "2024-01-15T10:30:00Z",
                "consent_purpose": "voice_authentication",
            },
        )

        assert response.status_code == 400

    def test_enroll_missing_consent(self, client, api_key_header, sample_audio_bytes):
        """Test enrollment without consent returns 400."""
        response = client.post(
            "/v1/speakers/enroll",
            headers=api_key_header,
            files={"audio": ("test.wav", sample_audio_bytes, "audio/wav")},
            data={
                "speaker_id": "test-speaker",
            },
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "CONSENT_REQUIRED"

    def test_enroll_short_audio(self, client, api_key_header, short_audio_bytes):
        """Test enrollment with short audio returns 400."""
        response = client.post(
            "/v1/speakers/enroll",
            headers=api_key_header,
            files={"audio": ("test.wav", short_audio_bytes, "audio/wav")},
            data={
                "speaker_id": "test-speaker",
                "consent_granted": "true",
                "consent_timestamp": "2024-01-15T10:30:00Z",
                "consent_purpose": "voice_authentication",
            },
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "AUDIO_TOO_SHORT"

    def test_enroll_duplicate_speaker(self, client, api_key_header, sample_audio_bytes):
        """Test enrolling duplicate speaker returns 409."""
        # First enrollment
        client.post(
            "/v1/speakers/enroll",
            headers=api_key_header,
            files={"audio": ("test.wav", sample_audio_bytes, "audio/wav")},
            data={
                "speaker_id": "duplicate-speaker",
                "consent_granted": "true",
                "consent_timestamp": "2024-01-15T10:30:00Z",
                "consent_purpose": "voice_authentication",
            },
        )

        # Second enrollment with same ID
        response = client.post(
            "/v1/speakers/enroll",
            headers=api_key_header,
            files={"audio": ("test.wav", sample_audio_bytes, "audio/wav")},
            data={
                "speaker_id": "duplicate-speaker",
                "consent_granted": "true",
                "consent_timestamp": "2024-01-15T10:30:00Z",
                "consent_purpose": "voice_authentication",
            },
        )

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "SPEAKER_ALREADY_EXISTS"


class TestVerifyEndpoint:
    """Test POST /speakers/verify endpoint."""

    def test_verify_success(self, client, api_key_header, sample_audio_bytes):
        """Test successful verification."""
        # First enroll a speaker
        client.post(
            "/v1/speakers/enroll",
            headers=api_key_header,
            files={"audio": ("test.wav", sample_audio_bytes, "audio/wav")},
            data={
                "speaker_id": "verify-test-speaker",
                "consent_granted": "true",
                "consent_timestamp": "2024-01-15T10:30:00Z",
                "consent_purpose": "voice_authentication",
            },
        )

        # Then verify
        response = client.post(
            "/v1/speakers/verify",
            headers=api_key_header,
            files={"audio": ("test.wav", sample_audio_bytes, "audio/wav")},
            data={
                "speaker_id": "verify-test-speaker",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["speaker_id"] == "verify-test-speaker"
        assert "is_verified" in data
        assert "score" in data
        assert 0.0 <= data["score"] <= 1.0
        assert "confidence" in data

    def test_verify_speaker_not_found(self, client, api_key_header, sample_audio_bytes):
        """Test verification of non-existent speaker returns 404."""
        response = client.post(
            "/v1/speakers/verify",
            headers=api_key_header,
            files={"audio": ("test.wav", sample_audio_bytes, "audio/wav")},
            data={
                "speaker_id": "non-existent-speaker",
            },
        )

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "SPEAKER_NOT_FOUND"

    def test_verify_custom_threshold(self, client, api_key_header, sample_audio_bytes):
        """Test verification with custom threshold."""
        # First enroll
        client.post(
            "/v1/speakers/enroll",
            headers=api_key_header,
            files={"audio": ("test.wav", sample_audio_bytes, "audio/wav")},
            data={
                "speaker_id": "threshold-test-speaker",
                "consent_granted": "true",
                "consent_timestamp": "2024-01-15T10:30:00Z",
                "consent_purpose": "voice_authentication",
            },
        )

        # Verify with high threshold
        response = client.post(
            "/v1/speakers/verify",
            headers=api_key_header,
            files={"audio": ("test.wav", sample_audio_bytes, "audio/wav")},
            data={
                "speaker_id": "threshold-test-speaker",
                "threshold": "0.9",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["threshold"] == 0.9


class TestIdentifyEndpoint:
    """Test POST /speakers/identify endpoint."""

    def test_identify_success(self, client, api_key_header, sample_audio_bytes):
        """Test successful identification."""
        # Enroll a speaker first
        client.post(
            "/v1/speakers/enroll",
            headers=api_key_header,
            files={"audio": ("test.wav", sample_audio_bytes, "audio/wav")},
            data={
                "speaker_id": "identify-test-speaker",
                "consent_granted": "true",
                "consent_timestamp": "2024-01-15T10:30:00Z",
                "consent_purpose": "voice_authentication",
            },
        )

        # Identify
        response = client.post(
            "/v1/speakers/identify",
            headers=api_key_header,
            files={"audio": ("test.wav", sample_audio_bytes, "audio/wav")},
            data={
                "max_results": "5",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "matches" in data
        assert isinstance(data["matches"], list)
        assert "total_searched" in data
        assert "processing_time_ms" in data

    def test_identify_returns_sorted_matches(
        self, client, api_key_header, sample_audio_bytes
    ):
        """Test that matches are sorted by score descending."""
        # This test requires multiple enrolled speakers
        response = client.post(
            "/v1/speakers/identify",
            headers=api_key_header,
            files={"audio": ("test.wav", sample_audio_bytes, "audio/wav")},
            data={
                "max_results": "10",
            },
        )

        assert response.status_code == 200
        data = response.json()
        matches = data["matches"]

        if len(matches) > 1:
            scores = [m["score"] for m in matches]
            assert scores == sorted(scores, reverse=True)

    def test_identify_respects_max_results(
        self, client, api_key_header, sample_audio_bytes
    ):
        """Test that max_results limits the number of matches."""
        response = client.post(
            "/v1/speakers/identify",
            headers=api_key_header,
            files={"audio": ("test.wav", sample_audio_bytes, "audio/wav")},
            data={
                "max_results": "3",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["matches"]) <= 3


class TestDeleteEndpoint:
    """Test DELETE /speakers/{speaker_id} endpoint."""

    def test_delete_success(self, client, api_key_header, sample_audio_bytes):
        """Test successful speaker deletion."""
        # First enroll
        client.post(
            "/v1/speakers/enroll",
            headers=api_key_header,
            files={"audio": ("test.wav", sample_audio_bytes, "audio/wav")},
            data={
                "speaker_id": "delete-test-speaker",
                "consent_granted": "true",
                "consent_timestamp": "2024-01-15T10:30:00Z",
                "consent_purpose": "voice_authentication",
            },
        )

        # Delete
        response = client.delete(
            "/v1/speakers/delete-test-speaker",
            headers=api_key_header,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["speaker_id"] == "delete-test-speaker"
        assert data["status"] == "deleted"
        assert "deleted_at" in data

    def test_delete_not_found(self, client, api_key_header):
        """Test deleting non-existent speaker returns 404."""
        response = client.delete(
            "/v1/speakers/non-existent-speaker",
            headers=api_key_header,
        )

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "SPEAKER_NOT_FOUND"

    def test_deleted_speaker_cannot_verify(
        self, client, api_key_header, sample_audio_bytes
    ):
        """Test that deleted speaker cannot be verified."""
        # Enroll
        client.post(
            "/v1/speakers/enroll",
            headers=api_key_header,
            files={"audio": ("test.wav", sample_audio_bytes, "audio/wav")},
            data={
                "speaker_id": "deleted-verify-speaker",
                "consent_granted": "true",
                "consent_timestamp": "2024-01-15T10:30:00Z",
                "consent_purpose": "voice_authentication",
            },
        )

        # Delete
        client.delete(
            "/v1/speakers/deleted-verify-speaker",
            headers=api_key_header,
        )

        # Try to verify
        response = client.post(
            "/v1/speakers/verify",
            headers=api_key_header,
            files={"audio": ("test.wav", sample_audio_bytes, "audio/wav")},
            data={
                "speaker_id": "deleted-verify-speaker",
            },
        )

        assert response.status_code == 404


class TestGetSpeakerEndpoint:
    """Test GET /speakers/{speaker_id} endpoint."""

    def test_get_speaker_success(self, client, api_key_header, sample_audio_bytes):
        """Test getting speaker information."""
        # First enroll
        client.post(
            "/v1/speakers/enroll",
            headers=api_key_header,
            files={"audio": ("test.wav", sample_audio_bytes, "audio/wav")},
            data={
                "speaker_id": "get-test-speaker",
                "consent_granted": "true",
                "consent_timestamp": "2024-01-15T10:30:00Z",
                "consent_purpose": "voice_authentication",
            },
        )

        # Get speaker
        response = client.get(
            "/v1/speakers/get-test-speaker",
            headers=api_key_header,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["speaker_id"] == "get-test-speaker"
        assert "enrollment_id" in data
        assert "created_at" in data

    def test_get_speaker_not_found(self, client, api_key_header):
        """Test getting non-existent speaker returns 404."""
        response = client.get(
            "/v1/speakers/non-existent-speaker",
            headers=api_key_header,
        )

        assert response.status_code == 404


class TestRateLimiting:
    """Test rate limiting."""

    def test_rate_limit_exceeded(self, client, api_key_header, sample_audio_bytes):
        """Test that rate limit returns 429."""
        # This test would need proper rate limiting setup
        # For now, we just verify the endpoint structure
        pass  # Placeholder for rate limit testing


class TestAuditLogging:
    """Test audit logging functionality."""

    def test_enroll_creates_audit_log(
        self, client, api_key_header, sample_audio_bytes
    ):
        """Test that enrollment creates an audit log entry."""
        # This would be tested with database inspection
        # For now, we verify the operation succeeds
        response = client.post(
            "/v1/speakers/enroll",
            headers=api_key_header,
            files={"audio": ("test.wav", sample_audio_bytes, "audio/wav")},
            data={
                "speaker_id": "audit-test-speaker",
                "consent_granted": "true",
                "consent_timestamp": "2024-01-15T10:30:00Z",
                "consent_purpose": "voice_authentication",
            },
        )

        assert response.status_code == 201
