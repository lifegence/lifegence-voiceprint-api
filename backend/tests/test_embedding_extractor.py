"""
TDD Tests for Embedding Extractor.

These tests define the expected behavior BEFORE implementation.
Run with: pytest tests/test_embedding_extractor.py -v
"""

import numpy as np
import pytest


class TestEmbeddingExtractor:
    """Test suite for EmbeddingExtractor class."""

    def test_extract_returns_correct_dimensions(self, sample_audio_bytes: bytes):
        """Test that embedding has 192 dimensions (ECAPA-TDNN)."""
        from app.core.embedding_extractor import EmbeddingExtractor

        extractor = EmbeddingExtractor()

        # Preprocess audio first
        from app.core.audio_processor import AudioProcessor

        processor = AudioProcessor()
        audio = processor.preprocess(sample_audio_bytes)

        embedding = extractor.extract(audio)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (192,)

    def test_extract_returns_normalized_vector(self, sample_audio_bytes: bytes):
        """Test that embedding is L2 normalized."""
        from app.core.embedding_extractor import EmbeddingExtractor
        from app.core.audio_processor import AudioProcessor

        extractor = EmbeddingExtractor()
        processor = AudioProcessor()
        audio = processor.preprocess(sample_audio_bytes)

        embedding = extractor.extract(audio)

        # Check L2 norm is approximately 1
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01

    def test_extract_same_audio_same_embedding(self, sample_audio_bytes: bytes):
        """Test that same audio produces same embedding."""
        from app.core.embedding_extractor import EmbeddingExtractor
        from app.core.audio_processor import AudioProcessor

        extractor = EmbeddingExtractor()
        processor = AudioProcessor()
        audio = processor.preprocess(sample_audio_bytes)

        embedding1 = extractor.extract(audio)
        embedding2 = extractor.extract(audio)

        # Should be identical
        np.testing.assert_array_almost_equal(embedding1, embedding2)

    def test_extract_different_audio_different_embedding(self):
        """Test that different audio produces different embeddings."""
        from app.core.embedding_extractor import EmbeddingExtractor
        from app.core.audio_processor import AudioProcessor
        import io
        import wave

        extractor = EmbeddingExtractor()
        processor = AudioProcessor()

        # Create two different audio samples
        sample_rate = 16000
        duration = 5.0

        # Audio 1: 440Hz
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio1 = np.sin(2 * np.pi * 440 * t) * 0.5
        audio1_int16 = (audio1 * 32767).astype(np.int16)
        buffer1 = io.BytesIO()
        with wave.open(buffer1, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio1_int16.tobytes())

        # Audio 2: 880Hz with different pattern
        audio2 = np.sin(2 * np.pi * 880 * t) * 0.5
        audio2 += np.sin(2 * np.pi * 220 * t) * 0.3
        audio2_int16 = (audio2 * 32767).astype(np.int16)
        buffer2 = io.BytesIO()
        with wave.open(buffer2, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio2_int16.tobytes())

        processed1 = processor.preprocess(buffer1.getvalue())
        processed2 = processor.preprocess(buffer2.getvalue())

        embedding1 = extractor.extract(processed1)
        embedding2 = extractor.extract(processed2)

        # Should not be identical
        assert not np.allclose(embedding1, embedding2, atol=0.1)

    def test_extract_batch(self, sample_audio_bytes: bytes):
        """Test batch extraction for multiple audio samples."""
        from app.core.embedding_extractor import EmbeddingExtractor
        from app.core.audio_processor import AudioProcessor

        extractor = EmbeddingExtractor()
        processor = AudioProcessor()
        audio = processor.preprocess(sample_audio_bytes)

        # Extract batch of 3
        audios = [audio, audio, audio]
        embeddings = extractor.extract_batch(audios)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 192)

    def test_extract_float32_dtype(self, sample_audio_bytes: bytes):
        """Test that embedding has float32 dtype."""
        from app.core.embedding_extractor import EmbeddingExtractor
        from app.core.audio_processor import AudioProcessor

        extractor = EmbeddingExtractor()
        processor = AudioProcessor()
        audio = processor.preprocess(sample_audio_bytes)

        embedding = extractor.extract(audio)

        assert embedding.dtype == np.float32


class TestSimilarityCalculation:
    """Test suite for similarity calculations."""

    def test_cosine_similarity_identical(self, sample_embedding: np.ndarray):
        """Test cosine similarity of identical vectors is 1.0."""
        from app.core.embedding_extractor import cosine_similarity

        score = cosine_similarity(sample_embedding, sample_embedding)

        assert abs(score - 1.0) < 0.001

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors is 0.0."""
        from app.core.embedding_extractor import cosine_similarity

        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        score = cosine_similarity(vec1, vec2)

        assert abs(score) < 0.001

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors is -1.0."""
        from app.core.embedding_extractor import cosine_similarity

        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

        score = cosine_similarity(vec1, vec2)

        assert abs(score + 1.0) < 0.001

    def test_cosine_similarity_range(self, sample_embedding: np.ndarray):
        """Test cosine similarity is always between -1 and 1."""
        from app.core.embedding_extractor import cosine_similarity

        # Generate random embeddings
        np.random.seed(123)
        for _ in range(100):
            vec1 = np.random.randn(192).astype(np.float32)
            vec2 = np.random.randn(192).astype(np.float32)
            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = vec2 / np.linalg.norm(vec2)

            score = cosine_similarity(vec1, vec2)

            assert -1.0 <= score <= 1.0

    def test_similarity_score_interpretation(self):
        """Test similarity score interpretation."""
        from app.core.embedding_extractor import interpret_similarity_score

        assert interpret_similarity_score(0.95) == "very_high"
        assert interpret_similarity_score(0.85) == "high"
        assert interpret_similarity_score(0.75) == "medium"
        assert interpret_similarity_score(0.60) == "low"
        assert interpret_similarity_score(0.40) == "very_low"
