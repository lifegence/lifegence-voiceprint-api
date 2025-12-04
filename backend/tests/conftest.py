"""
Pytest configuration and fixtures for VoiceID tests.
"""

import asyncio
import os
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

# Test configuration
os.environ["TESTING"] = "1"
os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost:5432/test_voiceid"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"
os.environ["API_KEY"] = "test-api-key-12345"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def app() -> FastAPI:
    """Create test FastAPI application."""
    from app.main import create_app

    return create_app()


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create synchronous test client."""
    return TestClient(app)


@pytest.fixture
async def async_client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Create asynchronous test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


@pytest.fixture
def api_key_header() -> dict[str, str]:
    """Return valid API key header."""
    return {"X-API-Key": "test-api-key-12345"}


@pytest.fixture
def sample_audio_bytes() -> bytes:
    """Generate sample audio data (16kHz, mono, 5 seconds)."""
    sample_rate = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # Generate simple sine wave with some variation
    audio = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz tone
    audio += np.sin(2 * np.pi * 880 * t) * 0.25  # harmonic
    audio += np.random.normal(0, 0.05, len(t))  # noise

    # Convert to 16-bit PCM WAV format
    import io
    import wave

    audio_int16 = (audio * 32767).astype(np.int16)
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    return buffer.getvalue()


@pytest.fixture
def short_audio_bytes() -> bytes:
    """Generate short audio (< 3 seconds) for validation tests."""
    sample_rate = 16000
    duration = 1.5  # Too short
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t) * 0.5

    import io
    import wave

    audio_int16 = (audio * 32767).astype(np.int16)
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    return buffer.getvalue()


@pytest.fixture
def sample_embedding() -> np.ndarray:
    """Generate sample embedding vector (192 dimensions)."""
    np.random.seed(42)
    embedding = np.random.randn(192).astype(np.float32)
    # Normalize
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


@pytest.fixture
def mock_embedding_extractor() -> MagicMock:
    """Mock embedding extractor."""
    mock = MagicMock()
    np.random.seed(42)
    embedding = np.random.randn(192).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    mock.extract.return_value = embedding
    return mock


@pytest.fixture
def mock_database() -> AsyncMock:
    """Mock database session."""
    mock = AsyncMock()
    return mock


@pytest.fixture
def consent_info() -> dict:
    """Sample consent information."""
    return {
        "granted": True,
        "timestamp": "2024-01-15T10:30:00Z",
        "purpose": "voice_authentication",
    }
