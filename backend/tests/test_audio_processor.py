"""
TDD Tests for Audio Processor.

These tests define the expected behavior BEFORE implementation.
Run with: pytest tests/test_audio_processor.py -v
"""

import io
import wave

import numpy as np
import pytest


class TestAudioProcessor:
    """Test suite for AudioProcessor class."""

    def test_preprocess_valid_audio(self, sample_audio_bytes: bytes):
        """Test preprocessing valid audio returns numpy array."""
        from app.core.audio_processor import AudioProcessor

        processor = AudioProcessor()
        result = processor.preprocess(sample_audio_bytes)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result.shape) == 1  # 1D array

    def test_preprocess_resamples_to_16khz(self):
        """Test that audio is resampled to 16kHz."""
        from app.core.audio_processor import AudioProcessor

        # Create 44.1kHz audio
        sample_rate = 44100
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t)
        audio_int16 = (audio * 32767).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        processor = AudioProcessor()
        result = processor.preprocess(buffer.getvalue())

        # Should be close to 16000 * 3 = 48000 samples
        expected_samples = 16000 * duration
        assert abs(len(result) - expected_samples) < 100  # Allow small tolerance

    def test_preprocess_converts_stereo_to_mono(self):
        """Test that stereo audio is converted to mono."""
        from app.core.audio_processor import AudioProcessor

        # Create stereo audio
        sample_rate = 16000
        duration = 3.0
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, dtype=np.float32)
        left = np.sin(2 * np.pi * 440 * t)
        right = np.sin(2 * np.pi * 880 * t)
        stereo = np.column_stack([left, right])
        stereo_int16 = (stereo * 32767).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(2)  # Stereo
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(stereo_int16.tobytes())

        processor = AudioProcessor()
        result = processor.preprocess(buffer.getvalue())

        # Result should be 1D (mono)
        assert len(result.shape) == 1

    def test_preprocess_normalizes_amplitude(self, sample_audio_bytes: bytes):
        """Test that audio amplitude is normalized."""
        from app.core.audio_processor import AudioProcessor

        processor = AudioProcessor()
        result = processor.preprocess(sample_audio_bytes)

        # Check that values are in reasonable range
        assert np.max(np.abs(result)) <= 1.0

    def test_validate_audio_too_short(self, short_audio_bytes: bytes):
        """Test validation rejects audio shorter than 3 seconds."""
        from app.core.audio_processor import AudioProcessor, AudioValidationError

        processor = AudioProcessor()

        with pytest.raises(AudioValidationError) as exc_info:
            processor.validate(short_audio_bytes)

        assert "too short" in str(exc_info.value).lower()
        assert exc_info.value.code == "AUDIO_TOO_SHORT"

    def test_validate_audio_too_long(self):
        """Test validation rejects audio longer than 30 seconds."""
        from app.core.audio_processor import AudioProcessor, AudioValidationError

        # Create 35 second audio
        sample_rate = 16000
        duration = 35.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t)
        audio_int16 = (audio * 32767).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        processor = AudioProcessor()

        with pytest.raises(AudioValidationError) as exc_info:
            processor.validate(buffer.getvalue())

        assert "too long" in str(exc_info.value).lower()
        assert exc_info.value.code == "AUDIO_TOO_LONG"

    def test_validate_audio_valid_duration(self, sample_audio_bytes: bytes):
        """Test validation accepts audio with valid duration."""
        from app.core.audio_processor import AudioProcessor, AudioValidationResult

        processor = AudioProcessor()
        result = processor.validate(sample_audio_bytes)

        assert isinstance(result, AudioValidationResult)
        assert result.is_valid is True
        assert 3.0 <= result.duration <= 30.0

    def test_validate_returns_duration(self, sample_audio_bytes: bytes):
        """Test validation result includes audio duration."""
        from app.core.audio_processor import AudioProcessor

        processor = AudioProcessor()
        result = processor.validate(sample_audio_bytes)

        assert hasattr(result, "duration")
        assert isinstance(result.duration, float)
        assert result.duration > 0

    def test_validate_returns_quality_score(self, sample_audio_bytes: bytes):
        """Test validation result includes quality score."""
        from app.core.audio_processor import AudioProcessor

        processor = AudioProcessor()
        result = processor.validate(sample_audio_bytes)

        assert hasattr(result, "quality_score")
        assert 0.0 <= result.quality_score <= 1.0

    def test_invalid_audio_format(self):
        """Test that invalid audio format raises error."""
        from app.core.audio_processor import AudioProcessor, AudioValidationError

        processor = AudioProcessor()
        invalid_data = b"not audio data"

        with pytest.raises(AudioValidationError) as exc_info:
            processor.preprocess(invalid_data)

        assert exc_info.value.code == "INVALID_AUDIO"

    def test_empty_audio(self):
        """Test that empty audio raises error."""
        from app.core.audio_processor import AudioProcessor, AudioValidationError

        processor = AudioProcessor()

        with pytest.raises(AudioValidationError) as exc_info:
            processor.preprocess(b"")

        assert exc_info.value.code == "INVALID_AUDIO"

    def test_get_audio_info(self, sample_audio_bytes: bytes):
        """Test getting audio information without full processing."""
        from app.core.audio_processor import AudioProcessor

        processor = AudioProcessor()
        info = processor.get_info(sample_audio_bytes)

        assert hasattr(info, "duration")
        assert hasattr(info, "sample_rate")
        assert hasattr(info, "channels")
        assert info.sample_rate == 16000
        assert info.channels == 1
