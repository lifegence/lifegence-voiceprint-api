"""
Main FastAPI application for Lifegence VoiceID.
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import __version__
from app.core.audio_processor import AudioValidationError
from app.services.speaker_service import (
    ConsentError,
    SpeakerAlreadyExistsError,
    SpeakerNotFoundError,
    SpeakerService,
)


# Global speaker service instance
_speaker_service: Optional[SpeakerService] = None


def get_speaker_service() -> SpeakerService:
    """Get speaker service instance."""
    global _speaker_service
    if _speaker_service is None:
        _speaker_service = SpeakerService()
    return _speaker_service


# API Key validation
API_KEY = os.environ.get("API_KEY", "test-api-key-12345")


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """Verify API key from header."""
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail={"error": {"code": "UNAUTHORIZED", "message": "Missing API key"}},
        )
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail={"error": {"code": "UNAUTHORIZED", "message": "Invalid API key"}},
        )
    return x_api_key


# Response models
class ErrorResponse(BaseModel):
    error: dict


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


class EnrollmentResponse(BaseModel):
    enrollment_id: str
    speaker_id: str
    status: str
    quality_score: float
    audio_duration: float
    created_at: str


class VerificationResponse(BaseModel):
    speaker_id: str
    is_verified: bool
    score: float
    threshold: float
    confidence: str
    processing_time_ms: int


class IdentificationMatchResponse(BaseModel):
    speaker_id: str
    score: float
    confidence: str
    metadata: Optional[dict] = None


class IdentificationResponse(BaseModel):
    matches: list[IdentificationMatchResponse]
    total_searched: int
    processing_time_ms: int


class SpeakerResponse(BaseModel):
    speaker_id: str
    enrollment_id: str
    enrollment_count: int
    metadata: Optional[dict] = None
    created_at: str
    updated_at: str
    last_verified_at: Optional[str] = None


class DeletionResponse(BaseModel):
    speaker_id: str
    status: str
    deleted_embeddings: int
    deleted_at: str


class UpdateEmbeddingResponse(BaseModel):
    speaker_id: str
    embedding_id: str
    status: str
    total_embeddings: int
    quality_score: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    global _speaker_service
    _speaker_service = SpeakerService()
    yield
    # Shutdown
    _speaker_service = None


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Lifegence VoiceID API",
        description="Voice recognition and speaker identification API",
        version=__version__,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health endpoint (no auth required)
    @app.get("/v1/health", response_model=HealthResponse)
    async def health_check():
        """Check service health."""
        return HealthResponse(
            status="healthy",
            version=__version__,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

    # Enrollment endpoint
    @app.post(
        "/v1/speakers/enroll",
        response_model=EnrollmentResponse,
        status_code=201,
    )
    async def enroll_speaker(
        audio: UploadFile = File(...),
        speaker_id: str = Form(...),
        consent_granted: str = Form(...),
        consent_timestamp: str = Form(...),
        consent_purpose: str = Form(...),
        metadata: Optional[str] = Form(None),
        api_key: str = Depends(verify_api_key),
        service: SpeakerService = Depends(get_speaker_service),
    ):
        """Enroll a new speaker."""
        try:
            # Parse consent
            consent_info = {
                "granted": consent_granted.lower() == "true",
                "timestamp": consent_timestamp,
                "purpose": consent_purpose,
            }

            # Parse metadata
            parsed_metadata = None
            if metadata:
                import json

                parsed_metadata = json.loads(metadata)

            # Read audio
            audio_data = await audio.read()

            # Enroll
            result = await service.enroll(
                speaker_id=speaker_id,
                audio_data=audio_data,
                consent_info=consent_info,
                metadata=parsed_metadata,
            )

            return EnrollmentResponse(
                enrollment_id=result.enrollment_id,
                speaker_id=result.speaker_id,
                status=result.status,
                quality_score=result.quality_score,
                audio_duration=result.audio_duration,
                created_at=result.created_at.isoformat() + "Z",
            )

        except ConsentError as e:
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": "CONSENT_REQUIRED", "message": str(e)}},
            )
        except AudioValidationError as e:
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": e.code, "message": e.message}},
            )
        except SpeakerAlreadyExistsError as e:
            raise HTTPException(
                status_code=409,
                detail={"error": {"code": "SPEAKER_ALREADY_EXISTS", "message": str(e)}},
            )

    # Verification endpoint
    @app.post("/v1/speakers/verify", response_model=VerificationResponse)
    async def verify_speaker(
        audio: UploadFile = File(...),
        speaker_id: str = Form(...),
        threshold: Optional[str] = Form(None),
        api_key: str = Depends(verify_api_key),
        service: SpeakerService = Depends(get_speaker_service),
    ):
        """Verify a speaker."""
        try:
            audio_data = await audio.read()
            thresh = float(threshold) if threshold else None

            result = await service.verify(
                speaker_id=speaker_id,
                audio_data=audio_data,
                threshold=thresh,
            )

            return VerificationResponse(
                speaker_id=result.speaker_id,
                is_verified=result.is_verified,
                score=result.score,
                threshold=result.threshold,
                confidence=result.confidence,
                processing_time_ms=result.processing_time_ms,
            )

        except SpeakerNotFoundError as e:
            raise HTTPException(
                status_code=404,
                detail={"error": {"code": "SPEAKER_NOT_FOUND", "message": str(e)}},
            )
        except AudioValidationError as e:
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": e.code, "message": e.message}},
            )

    # Identification endpoint
    @app.post("/v1/speakers/identify", response_model=IdentificationResponse)
    async def identify_speaker(
        audio: UploadFile = File(...),
        max_results: Optional[str] = Form(None),
        threshold: Optional[str] = Form(None),
        group_id: Optional[str] = Form(None),
        api_key: str = Depends(verify_api_key),
        service: SpeakerService = Depends(get_speaker_service),
    ):
        """Identify a speaker from audio."""
        try:
            audio_data = await audio.read()
            max_res = int(max_results) if max_results else 5
            thresh = float(threshold) if threshold else None

            result = await service.identify(
                audio_data=audio_data,
                max_results=max_res,
                threshold=thresh,
                group_id=group_id,
            )

            return IdentificationResponse(
                matches=[
                    IdentificationMatchResponse(
                        speaker_id=m.speaker_id,
                        score=m.score,
                        confidence=m.confidence,
                        metadata=m.metadata,
                    )
                    for m in result.matches
                ],
                total_searched=result.total_searched,
                processing_time_ms=result.processing_time_ms,
            )

        except AudioValidationError as e:
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": e.code, "message": e.message}},
            )

    # Get speaker endpoint
    @app.get("/v1/speakers/{speaker_id}", response_model=SpeakerResponse)
    async def get_speaker(
        speaker_id: str,
        api_key: str = Depends(verify_api_key),
        service: SpeakerService = Depends(get_speaker_service),
    ):
        """Get speaker information."""
        try:
            result = await service.get_speaker(speaker_id)

            return SpeakerResponse(
                speaker_id=result.speaker_id,
                enrollment_id=result.enrollment_id,
                enrollment_count=result.enrollment_count,
                metadata=result.metadata,
                created_at=result.created_at.isoformat() + "Z",
                updated_at=result.updated_at.isoformat() + "Z",
                last_verified_at=(
                    result.last_verified_at.isoformat() + "Z"
                    if result.last_verified_at
                    else None
                ),
            )

        except SpeakerNotFoundError as e:
            raise HTTPException(
                status_code=404,
                detail={"error": {"code": "SPEAKER_NOT_FOUND", "message": str(e)}},
            )

    # Delete speaker endpoint
    @app.delete("/v1/speakers/{speaker_id}", response_model=DeletionResponse)
    async def delete_speaker(
        speaker_id: str,
        api_key: str = Depends(verify_api_key),
        service: SpeakerService = Depends(get_speaker_service),
    ):
        """Delete a speaker and all associated data."""
        try:
            result = await service.delete(speaker_id)

            return DeletionResponse(
                speaker_id=result.speaker_id,
                status=result.status,
                deleted_embeddings=result.deleted_embeddings,
                deleted_at=result.deleted_at.isoformat() + "Z",
            )

        except SpeakerNotFoundError as e:
            raise HTTPException(
                status_code=404,
                detail={"error": {"code": "SPEAKER_NOT_FOUND", "message": str(e)}},
            )

    # Update embedding endpoint
    @app.put(
        "/v1/speakers/{speaker_id}/embedding",
        response_model=UpdateEmbeddingResponse,
    )
    async def update_embedding(
        speaker_id: str,
        audio: UploadFile = File(...),
        replace: Optional[str] = Form(None),
        api_key: str = Depends(verify_api_key),
        service: SpeakerService = Depends(get_speaker_service),
    ):
        """Update speaker embedding."""
        try:
            audio_data = await audio.read()
            do_replace = replace and replace.lower() == "true"

            result = await service.update_embedding(
                speaker_id=speaker_id,
                audio_data=audio_data,
                replace=do_replace,
            )

            return UpdateEmbeddingResponse(
                speaker_id=result.speaker_id,
                embedding_id=result.embedding_id,
                status=result.status,
                total_embeddings=result.total_embeddings,
                quality_score=result.quality_score,
            )

        except SpeakerNotFoundError as e:
            raise HTTPException(
                status_code=404,
                detail={"error": {"code": "SPEAKER_NOT_FOUND", "message": str(e)}},
            )
        except AudioValidationError as e:
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": e.code, "message": e.message}},
            )

    return app


# Application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
