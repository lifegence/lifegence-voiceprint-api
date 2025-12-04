"""
TDD Tests for Speaker Service.

These tests define the expected behavior BEFORE implementation.
Run with: pytest tests/test_speaker_service.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
import numpy as np
from datetime import datetime


class TestSpeakerService:
    """Test suite for SpeakerService class."""

    @pytest.fixture
    def speaker_service(self, mock_database, mock_embedding_extractor):
        """Create SpeakerService with mocked dependencies."""
        from app.services.speaker_service import SpeakerService

        service = SpeakerService(
            db=mock_database,
            embedding_extractor=mock_embedding_extractor,
        )
        return service

    @pytest.mark.asyncio
    async def test_enroll_speaker_success(
        self, speaker_service, sample_audio_bytes, consent_info
    ):
        """Test successful speaker enrollment."""
        from app.services.speaker_service import EnrollmentResult

        result = await speaker_service.enroll(
            speaker_id="test-speaker-001",
            audio_data=sample_audio_bytes,
            consent_info=consent_info,
        )

        assert isinstance(result, EnrollmentResult)
        assert result.speaker_id == "test-speaker-001"
        assert result.status == "enrolled"
        assert result.enrollment_id is not None
        assert 0.0 <= result.quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_enroll_speaker_generates_embedding(
        self, speaker_service, sample_audio_bytes, consent_info, mock_embedding_extractor
    ):
        """Test that enrollment generates and stores embedding."""
        await speaker_service.enroll(
            speaker_id="test-speaker-002",
            audio_data=sample_audio_bytes,
            consent_info=consent_info,
        )

        # Verify embedding extractor was called
        mock_embedding_extractor.extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_enroll_speaker_stores_consent(
        self, speaker_service, sample_audio_bytes, consent_info, mock_database
    ):
        """Test that enrollment stores consent information."""
        await speaker_service.enroll(
            speaker_id="test-speaker-003",
            audio_data=sample_audio_bytes,
            consent_info=consent_info,
        )

        # Verify database was called with consent info
        # This checks the service stores consent properly
        assert mock_database.execute.called or mock_database.add.called

    @pytest.mark.asyncio
    async def test_verify_speaker_success(
        self, speaker_service, sample_audio_bytes, sample_embedding
    ):
        """Test successful speaker verification."""
        from app.services.speaker_service import VerificationResult

        # Setup: pretend speaker exists
        speaker_service._get_speaker_embedding = AsyncMock(return_value=sample_embedding)

        result = await speaker_service.verify(
            speaker_id="existing-speaker",
            audio_data=sample_audio_bytes,
        )

        assert isinstance(result, VerificationResult)
        assert result.speaker_id == "existing-speaker"
        assert isinstance(result.is_verified, bool)
        assert 0.0 <= result.score <= 1.0
        assert result.confidence in ["very_high", "high", "medium", "low", "very_low"]

    @pytest.mark.asyncio
    async def test_verify_speaker_not_found(self, speaker_service, sample_audio_bytes):
        """Test verification of non-existent speaker raises error."""
        from app.services.speaker_service import SpeakerNotFoundError

        speaker_service._get_speaker_embedding = AsyncMock(return_value=None)

        with pytest.raises(SpeakerNotFoundError):
            await speaker_service.verify(
                speaker_id="non-existent",
                audio_data=sample_audio_bytes,
            )

    @pytest.mark.asyncio
    async def test_verify_uses_threshold(
        self, speaker_service, sample_audio_bytes, sample_embedding
    ):
        """Test that verification uses specified threshold."""
        speaker_service._get_speaker_embedding = AsyncMock(return_value=sample_embedding)

        # With low threshold
        result_low = await speaker_service.verify(
            speaker_id="test-speaker",
            audio_data=sample_audio_bytes,
            threshold=0.3,
        )

        # With high threshold
        result_high = await speaker_service.verify(
            speaker_id="test-speaker",
            audio_data=sample_audio_bytes,
            threshold=0.99,
        )

        assert result_low.threshold == 0.3
        assert result_high.threshold == 0.99

    @pytest.mark.asyncio
    async def test_identify_speaker_returns_matches(
        self, speaker_service, sample_audio_bytes
    ):
        """Test speaker identification returns matches."""
        from app.services.speaker_service import IdentificationResult

        result = await speaker_service.identify(
            audio_data=sample_audio_bytes,
            max_results=5,
        )

        assert isinstance(result, IdentificationResult)
        assert isinstance(result.matches, list)
        assert result.total_searched >= 0
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_identify_matches_sorted_by_score(
        self, speaker_service, sample_audio_bytes
    ):
        """Test that identification matches are sorted by score."""
        result = await speaker_service.identify(
            audio_data=sample_audio_bytes,
            max_results=10,
        )

        if len(result.matches) > 1:
            scores = [m.score for m in result.matches]
            assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_identify_respects_threshold(
        self, speaker_service, sample_audio_bytes
    ):
        """Test that identification respects minimum threshold."""
        result = await speaker_service.identify(
            audio_data=sample_audio_bytes,
            max_results=10,
            threshold=0.8,
        )

        for match in result.matches:
            assert match.score >= 0.8

    @pytest.mark.asyncio
    async def test_delete_speaker_success(self, speaker_service, mock_database):
        """Test successful speaker deletion."""
        from app.services.speaker_service import DeletionResult

        mock_database.get_speaker = AsyncMock(
            return_value=MagicMock(id="uuid", external_id="test-speaker")
        )

        result = await speaker_service.delete(speaker_id="test-speaker")

        assert isinstance(result, DeletionResult)
        assert result.speaker_id == "test-speaker"
        assert result.status == "deleted"
        assert result.deleted_at is not None

    @pytest.mark.asyncio
    async def test_delete_speaker_not_found(self, speaker_service, mock_database):
        """Test deleting non-existent speaker raises error."""
        from app.services.speaker_service import SpeakerNotFoundError

        mock_database.get_speaker = AsyncMock(return_value=None)

        with pytest.raises(SpeakerNotFoundError):
            await speaker_service.delete(speaker_id="non-existent")

    @pytest.mark.asyncio
    async def test_delete_removes_all_embeddings(self, speaker_service, mock_database):
        """Test that deletion removes all speaker embeddings."""
        mock_database.get_speaker = AsyncMock(
            return_value=MagicMock(id="uuid", external_id="test-speaker")
        )
        mock_database.delete_embeddings = AsyncMock(return_value=3)

        result = await speaker_service.delete(speaker_id="test-speaker")

        assert result.deleted_embeddings == 3

    @pytest.mark.asyncio
    async def test_get_speaker_success(self, speaker_service, mock_database):
        """Test getting speaker information."""
        from app.services.speaker_service import SpeakerInfo

        mock_speaker = MagicMock(
            id="uuid-123",
            external_id="test-speaker",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_database.get_speaker = AsyncMock(return_value=mock_speaker)
        mock_database.count_embeddings = AsyncMock(return_value=2)

        result = await speaker_service.get_speaker(speaker_id="test-speaker")

        assert isinstance(result, SpeakerInfo)
        assert result.speaker_id == "test-speaker"
        assert result.enrollment_count == 2

    @pytest.mark.asyncio
    async def test_update_embedding_adds_new(
        self, speaker_service, sample_audio_bytes, mock_database
    ):
        """Test updating embedding adds to existing."""
        mock_database.get_speaker = AsyncMock(
            return_value=MagicMock(id="uuid", external_id="test-speaker")
        )

        result = await speaker_service.update_embedding(
            speaker_id="test-speaker",
            audio_data=sample_audio_bytes,
            replace=False,
        )

        assert result.status == "updated"
        assert result.total_embeddings > 0


class TestConsentValidation:
    """Test consent validation logic."""

    def test_valid_consent(self):
        """Test valid consent passes validation."""
        from app.services.speaker_service import validate_consent

        consent = {
            "granted": True,
            "timestamp": "2024-01-15T10:30:00Z",
            "purpose": "voice_authentication",
        }

        result = validate_consent(consent)
        assert result is True

    def test_consent_not_granted(self):
        """Test consent not granted fails validation."""
        from app.services.speaker_service import validate_consent, ConsentError

        consent = {
            "granted": False,
            "timestamp": "2024-01-15T10:30:00Z",
            "purpose": "voice_authentication",
        }

        with pytest.raises(ConsentError) as exc_info:
            validate_consent(consent)

        assert "not granted" in str(exc_info.value).lower()

    def test_consent_missing_timestamp(self):
        """Test consent missing timestamp fails validation."""
        from app.services.speaker_service import validate_consent, ConsentError

        consent = {
            "granted": True,
            "purpose": "voice_authentication",
        }

        with pytest.raises(ConsentError) as exc_info:
            validate_consent(consent)

        assert "timestamp" in str(exc_info.value).lower()

    def test_consent_missing_purpose(self):
        """Test consent missing purpose fails validation."""
        from app.services.speaker_service import validate_consent, ConsentError

        consent = {
            "granted": True,
            "timestamp": "2024-01-15T10:30:00Z",
        }

        with pytest.raises(ConsentError) as exc_info:
            validate_consent(consent)

        assert "purpose" in str(exc_info.value).lower()


class TestQualityScoreCalculation:
    """Test quality score calculation."""

    def test_quality_score_good_audio(self, sample_audio_bytes):
        """Test quality score for good audio."""
        from app.services.speaker_service import calculate_quality_score

        score = calculate_quality_score(sample_audio_bytes)

        assert 0.0 <= score <= 1.0

    def test_quality_score_noisy_audio(self):
        """Test quality score is lower for noisy audio."""
        from app.services.speaker_service import calculate_quality_score
        import io
        import wave

        # Create noisy audio
        sample_rate = 16000
        duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        # Mostly noise, little signal
        audio = np.random.normal(0, 0.5, len(t)).astype(np.float32)
        audio_int16 = (audio * 32767).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio_int16.tobytes())

        score = calculate_quality_score(buffer.getvalue())

        # Noisy audio should have lower score
        assert score < 0.7
