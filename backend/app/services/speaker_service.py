"""
Speaker service for voice recognition operations.

Handles enrollment, verification, identification, and management of speakers.
"""

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.audio_processor import AudioProcessor, AudioValidationError
from app.core.embedding_extractor import (
    EmbeddingExtractor,
    cosine_similarity,
    interpret_similarity_score,
)


class SpeakerNotFoundError(Exception):
    """Raised when speaker is not found."""

    pass


class SpeakerAlreadyExistsError(Exception):
    """Raised when speaker already exists."""

    pass


class ConsentError(Exception):
    """Raised when consent validation fails."""

    pass


@dataclass
class EnrollmentResult:
    """Result of speaker enrollment."""

    enrollment_id: str
    speaker_id: str
    status: str
    quality_score: float
    audio_duration: float
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class VerificationResult:
    """Result of speaker verification."""

    speaker_id: str
    is_verified: bool
    score: float
    threshold: float
    confidence: str
    processing_time_ms: int


@dataclass
class IdentificationMatch:
    """A single identification match."""

    speaker_id: str
    score: float
    confidence: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class IdentificationResult:
    """Result of speaker identification."""

    matches: List[IdentificationMatch]
    total_searched: int
    processing_time_ms: int


@dataclass
class DeletionResult:
    """Result of speaker deletion."""

    speaker_id: str
    status: str
    deleted_embeddings: int
    deleted_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SpeakerInfo:
    """Speaker information."""

    speaker_id: str
    enrollment_id: str
    enrollment_count: int
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    last_verified_at: Optional[datetime] = None


@dataclass
class UpdateEmbeddingResult:
    """Result of embedding update."""

    speaker_id: str
    embedding_id: str
    status: str
    total_embeddings: int
    quality_score: float


def validate_consent(consent_info: Dict[str, Any]) -> bool:
    """
    Validate consent information.

    Args:
        consent_info: Dictionary with consent details

    Returns:
        True if valid

    Raises:
        ConsentError: If validation fails
    """
    if not consent_info.get("granted"):
        raise ConsentError("Consent not granted")

    if "timestamp" not in consent_info:
        raise ConsentError("Consent timestamp is required")

    if "purpose" not in consent_info:
        raise ConsentError("Consent purpose is required")

    return True


def calculate_quality_score(audio_bytes: bytes) -> float:
    """
    Calculate quality score for audio.

    Args:
        audio_bytes: Raw audio bytes

    Returns:
        Quality score (0.0 to 1.0)
    """
    processor = AudioProcessor()
    result = processor.validate(audio_bytes)
    return result.quality_score


class SpeakerService:
    """
    Service for speaker voice recognition operations.

    Provides enrollment, verification, identification, and management.
    """

    DEFAULT_VERIFICATION_THRESHOLD = 0.7
    DEFAULT_IDENTIFICATION_THRESHOLD = 0.5

    def __init__(
        self,
        db: Any = None,
        embedding_extractor: Optional[EmbeddingExtractor] = None,
    ):
        """
        Initialize speaker service.

        Args:
            db: Database session/connection
            embedding_extractor: Embedding extractor instance
        """
        self._db = db
        self._embedding_extractor = embedding_extractor or EmbeddingExtractor()
        self._audio_processor = AudioProcessor()

        # In-memory storage for testing (replace with DB in production)
        self._speakers: Dict[str, Dict] = {}
        self._embeddings: Dict[str, List[np.ndarray]] = {}

    async def enroll(
        self,
        speaker_id: str,
        audio_data: bytes,
        consent_info: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EnrollmentResult:
        """
        Enroll a new speaker.

        Args:
            speaker_id: External speaker identifier
            audio_data: Raw audio bytes
            consent_info: Consent information
            metadata: Optional metadata

        Returns:
            EnrollmentResult

        Raises:
            ConsentError: If consent is invalid
            AudioValidationError: If audio is invalid
            SpeakerAlreadyExistsError: If speaker exists
        """
        # Validate consent
        validate_consent(consent_info)

        # Check if speaker exists
        if speaker_id in self._speakers:
            raise SpeakerAlreadyExistsError(f"Speaker {speaker_id} already exists")

        # Validate and preprocess audio
        validation_result = self._audio_processor.validate(audio_data)
        audio = self._audio_processor.preprocess(audio_data)

        # Extract embedding
        embedding = self._embedding_extractor.extract(audio)

        # Generate enrollment ID
        enrollment_id = str(uuid.uuid4())

        # Store speaker (in-memory for now)
        self._speakers[speaker_id] = {
            "id": enrollment_id,
            "external_id": speaker_id,
            "consent_info": consent_info,
            "metadata": metadata,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        self._embeddings[speaker_id] = [embedding]

        return EnrollmentResult(
            enrollment_id=enrollment_id,
            speaker_id=speaker_id,
            status="enrolled",
            quality_score=validation_result.quality_score,
            audio_duration=validation_result.duration,
        )

    async def verify(
        self,
        speaker_id: str,
        audio_data: bytes,
        threshold: Optional[float] = None,
    ) -> VerificationResult:
        """
        Verify a speaker against enrolled voiceprint.

        Args:
            speaker_id: Speaker to verify against
            audio_data: Raw audio bytes
            threshold: Verification threshold (default: 0.7)

        Returns:
            VerificationResult

        Raises:
            SpeakerNotFoundError: If speaker not found
            AudioValidationError: If audio is invalid
        """
        start_time = time.time()

        threshold = threshold or self.DEFAULT_VERIFICATION_THRESHOLD

        # Get enrolled embedding
        enrolled_embedding = await self._get_speaker_embedding(speaker_id)
        if enrolled_embedding is None:
            raise SpeakerNotFoundError(f"Speaker {speaker_id} not found")

        # Validate and preprocess audio
        self._audio_processor.validate(audio_data)
        audio = self._audio_processor.preprocess(audio_data)

        # Extract embedding
        test_embedding = self._embedding_extractor.extract(audio)

        # Calculate similarity
        score = cosine_similarity(test_embedding, enrolled_embedding)

        # Determine verification result
        is_verified = score >= threshold
        confidence = interpret_similarity_score(score)

        processing_time_ms = int((time.time() - start_time) * 1000)

        # Update last verified timestamp
        if speaker_id in self._speakers:
            self._speakers[speaker_id]["last_verified_at"] = datetime.utcnow()

        return VerificationResult(
            speaker_id=speaker_id,
            is_verified=is_verified,
            score=float(score),
            threshold=threshold,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
        )

    async def identify(
        self,
        audio_data: bytes,
        max_results: int = 5,
        threshold: Optional[float] = None,
        group_id: Optional[str] = None,
    ) -> IdentificationResult:
        """
        Identify speaker from audio.

        Args:
            audio_data: Raw audio bytes
            max_results: Maximum matches to return
            threshold: Minimum similarity threshold
            group_id: Optional group to search within

        Returns:
            IdentificationResult
        """
        start_time = time.time()

        threshold = threshold or self.DEFAULT_IDENTIFICATION_THRESHOLD

        # Validate and preprocess audio
        self._audio_processor.validate(audio_data)
        audio = self._audio_processor.preprocess(audio_data)

        # Extract embedding
        test_embedding = self._embedding_extractor.extract(audio)

        # Search all enrolled speakers
        matches = []
        for speaker_id, embeddings in self._embeddings.items():
            # Use average of all embeddings for speaker
            if embeddings:
                avg_embedding = np.mean(embeddings, axis=0)
                score = cosine_similarity(test_embedding, avg_embedding)

                if score >= threshold:
                    metadata = self._speakers.get(speaker_id, {}).get("metadata")
                    matches.append(
                        IdentificationMatch(
                            speaker_id=speaker_id,
                            score=float(score),
                            confidence=interpret_similarity_score(score),
                            metadata=metadata,
                        )
                    )

        # Sort by score descending
        matches.sort(key=lambda m: m.score, reverse=True)

        processing_time_ms = int((time.time() - start_time) * 1000)

        return IdentificationResult(
            matches=matches[:max_results],
            total_searched=len(self._embeddings),
            processing_time_ms=processing_time_ms,
        )

    async def delete(self, speaker_id: str) -> DeletionResult:
        """
        Delete a speaker and all associated data.

        Args:
            speaker_id: Speaker to delete

        Returns:
            DeletionResult

        Raises:
            SpeakerNotFoundError: If speaker not found
        """
        if speaker_id not in self._speakers:
            raise SpeakerNotFoundError(f"Speaker {speaker_id} not found")

        # Count embeddings
        embedding_count = len(self._embeddings.get(speaker_id, []))

        # Delete
        del self._speakers[speaker_id]
        if speaker_id in self._embeddings:
            del self._embeddings[speaker_id]

        return DeletionResult(
            speaker_id=speaker_id,
            status="deleted",
            deleted_embeddings=embedding_count,
        )

    async def get_speaker(self, speaker_id: str) -> SpeakerInfo:
        """
        Get speaker information.

        Args:
            speaker_id: Speaker ID

        Returns:
            SpeakerInfo

        Raises:
            SpeakerNotFoundError: If speaker not found
        """
        if speaker_id not in self._speakers:
            raise SpeakerNotFoundError(f"Speaker {speaker_id} not found")

        speaker = self._speakers[speaker_id]
        embedding_count = len(self._embeddings.get(speaker_id, []))

        return SpeakerInfo(
            speaker_id=speaker_id,
            enrollment_id=speaker["id"],
            enrollment_count=embedding_count,
            metadata=speaker.get("metadata"),
            created_at=speaker["created_at"],
            updated_at=speaker["updated_at"],
            last_verified_at=speaker.get("last_verified_at"),
        )

    async def update_embedding(
        self,
        speaker_id: str,
        audio_data: bytes,
        replace: bool = False,
    ) -> UpdateEmbeddingResult:
        """
        Update speaker embedding.

        Args:
            speaker_id: Speaker ID
            audio_data: Raw audio bytes
            replace: Replace existing embeddings if True

        Returns:
            UpdateEmbeddingResult

        Raises:
            SpeakerNotFoundError: If speaker not found
        """
        if speaker_id not in self._speakers:
            raise SpeakerNotFoundError(f"Speaker {speaker_id} not found")

        # Validate and preprocess audio
        validation_result = self._audio_processor.validate(audio_data)
        audio = self._audio_processor.preprocess(audio_data)

        # Extract embedding
        embedding = self._embedding_extractor.extract(audio)
        embedding_id = str(uuid.uuid4())

        if replace:
            self._embeddings[speaker_id] = [embedding]
        else:
            if speaker_id not in self._embeddings:
                self._embeddings[speaker_id] = []
            self._embeddings[speaker_id].append(embedding)

        # Update timestamp
        self._speakers[speaker_id]["updated_at"] = datetime.utcnow()

        return UpdateEmbeddingResult(
            speaker_id=speaker_id,
            embedding_id=embedding_id,
            status="updated",
            total_embeddings=len(self._embeddings[speaker_id]),
            quality_score=validation_result.quality_score,
        )

    async def _get_speaker_embedding(
        self, speaker_id: str
    ) -> Optional[np.ndarray]:
        """
        Get speaker's embedding (average of all enrollments).

        Args:
            speaker_id: Speaker ID

        Returns:
            Average embedding or None if not found
        """
        embeddings = self._embeddings.get(speaker_id)
        if not embeddings:
            return None

        return np.mean(embeddings, axis=0)
