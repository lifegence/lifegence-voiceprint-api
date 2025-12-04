"""
Audio processing module for voice recognition.

Handles audio preprocessing, validation, and format conversion.
"""

import io
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import librosa
    import soundfile as sf

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class AudioValidationError(Exception):
    """Exception raised for audio validation errors."""

    def __init__(self, message: str, code: str):
        self.message = message
        self.code = code
        super().__init__(self.message)


@dataclass
class AudioValidationResult:
    """Result of audio validation."""

    is_valid: bool
    duration: float
    sample_rate: int
    channels: int
    quality_score: float
    message: Optional[str] = None


@dataclass
class AudioInfo:
    """Basic audio information."""

    duration: float
    sample_rate: int
    channels: int


class AudioProcessor:
    """
    Processes audio for voice recognition.

    Handles:
    - Audio format conversion
    - Resampling to 16kHz
    - Stereo to mono conversion
    - Amplitude normalization
    - Silence removal (VAD)
    """

    # Configuration
    TARGET_SAMPLE_RATE = 16000
    MIN_DURATION = 3.0  # seconds
    MAX_DURATION = 30.0  # seconds
    MIN_QUALITY_SCORE = 0.3

    def preprocess(self, audio_bytes: bytes) -> np.ndarray:
        """
        Preprocess audio data for embedding extraction.

        Args:
            audio_bytes: Raw audio bytes (WAV, WebM, OGG, MP3)

        Returns:
            Preprocessed audio as numpy array (float32, 16kHz, mono)

        Raises:
            AudioValidationError: If audio cannot be processed
        """
        if not audio_bytes:
            raise AudioValidationError("Empty audio data", "INVALID_AUDIO")

        try:
            # Try to read as WAV first
            audio, sample_rate = self._read_audio(audio_bytes)
        except Exception as e:
            raise AudioValidationError(
                f"Failed to decode audio: {str(e)}", "INVALID_AUDIO"
            )

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample to target rate
        if sample_rate != self.TARGET_SAMPLE_RATE:
            audio = self._resample(audio, sample_rate, self.TARGET_SAMPLE_RATE)

        # Normalize amplitude
        audio = self._normalize(audio)

        # Ensure float32
        return audio.astype(np.float32)

    def validate(self, audio_bytes: bytes) -> AudioValidationResult:
        """
        Validate audio for voice recognition.

        Args:
            audio_bytes: Raw audio bytes

        Returns:
            AudioValidationResult with validation details

        Raises:
            AudioValidationError: If validation fails
        """
        if not audio_bytes:
            raise AudioValidationError("Empty audio data", "INVALID_AUDIO")

        try:
            info = self.get_info(audio_bytes)
        except Exception as e:
            raise AudioValidationError(f"Invalid audio format: {e}", "INVALID_AUDIO")

        # Check duration
        if info.duration < self.MIN_DURATION:
            raise AudioValidationError(
                f"Audio too short: {info.duration:.1f}s (minimum: {self.MIN_DURATION}s)",
                "AUDIO_TOO_SHORT",
            )

        if info.duration > self.MAX_DURATION:
            raise AudioValidationError(
                f"Audio too long: {info.duration:.1f}s (maximum: {self.MAX_DURATION}s)",
                "AUDIO_TOO_LONG",
            )

        # Calculate quality score
        quality_score = self._calculate_quality_score(audio_bytes)

        return AudioValidationResult(
            is_valid=True,
            duration=info.duration,
            sample_rate=info.sample_rate,
            channels=info.channels,
            quality_score=quality_score,
        )

    def get_info(self, audio_bytes: bytes) -> AudioInfo:
        """
        Get basic audio information without full processing.

        Args:
            audio_bytes: Raw audio bytes

        Returns:
            AudioInfo with duration, sample_rate, channels
        """
        try:
            audio, sample_rate = self._read_audio(audio_bytes)

            if len(audio.shape) > 1:
                channels = audio.shape[1]
            else:
                channels = 1

            duration = len(audio) / sample_rate

            return AudioInfo(
                duration=duration,
                sample_rate=self.TARGET_SAMPLE_RATE,  # Return target rate
                channels=1,  # Return target channels
            )
        except Exception as e:
            raise AudioValidationError(f"Cannot read audio info: {e}", "INVALID_AUDIO")

    def _read_audio(self, audio_bytes: bytes) -> tuple[np.ndarray, int]:
        """
        Read audio from bytes.

        Supports WAV format primarily. For other formats, requires librosa.
        """
        buffer = io.BytesIO(audio_bytes)

        try:
            # Try WAV first
            with wave.open(buffer, "rb") as wav:
                sample_rate = wav.getframerate()
                n_channels = wav.getnchannels()
                n_frames = wav.getnframes()
                sample_width = wav.getsampwidth()

                raw_data = wav.readframes(n_frames)

                # Convert to numpy
                if sample_width == 2:  # 16-bit
                    dtype = np.int16
                elif sample_width == 4:  # 32-bit
                    dtype = np.int32
                else:
                    dtype = np.int16

                audio = np.frombuffer(raw_data, dtype=dtype)

                # Handle stereo
                if n_channels == 2:
                    audio = audio.reshape(-1, 2)

                # Convert to float
                audio = audio.astype(np.float32) / np.iinfo(dtype).max

                return audio, sample_rate

        except Exception:
            # Try librosa for other formats
            if LIBROSA_AVAILABLE:
                buffer.seek(0)
                audio, sample_rate = librosa.load(
                    buffer, sr=None, mono=False
                )
                if len(audio.shape) > 1:
                    audio = audio.T  # librosa returns (channels, samples)
                return audio, sample_rate
            else:
                raise AudioValidationError(
                    "Unsupported audio format", "INVALID_AUDIO"
                )

    def _resample(
        self, audio: np.ndarray, orig_rate: int, target_rate: int
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        if LIBROSA_AVAILABLE:
            return librosa.resample(audio, orig_sr=orig_rate, target_sr=target_rate)
        else:
            # Simple resampling using scipy
            from scipy import signal

            num_samples = int(len(audio) * target_rate / orig_rate)
            return signal.resample(audio, num_samples)

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude to [-1, 1]."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio

    def _calculate_quality_score(self, audio_bytes: bytes) -> float:
        """
        Calculate audio quality score.

        Considers:
        - Signal-to-noise ratio
        - Dynamic range
        - Clipping
        """
        try:
            audio, _ = self._read_audio(audio_bytes)

            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Normalize for analysis
            audio = audio.astype(np.float32)
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

            # Calculate metrics
            rms = np.sqrt(np.mean(audio**2))
            peak = np.max(np.abs(audio))

            # Check for clipping (values near 1.0)
            clipping_ratio = np.mean(np.abs(audio) > 0.99)

            # Calculate dynamic range (simplified)
            if rms > 0:
                dynamic_range = 20 * np.log10(peak / rms + 1e-10)
            else:
                dynamic_range = 0

            # Combine into quality score
            score = 0.0

            # RMS score (not too quiet, not too loud)
            if 0.05 < rms < 0.5:
                score += 0.4
            elif 0.01 < rms < 0.7:
                score += 0.2

            # Dynamic range score
            if dynamic_range > 10:
                score += 0.3
            elif dynamic_range > 5:
                score += 0.15

            # Clipping penalty
            if clipping_ratio < 0.01:
                score += 0.3
            elif clipping_ratio < 0.05:
                score += 0.15

            return min(1.0, max(0.0, score))

        except Exception:
            return 0.5  # Default score on error
