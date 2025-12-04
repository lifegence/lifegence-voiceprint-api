"""
Voice embedding extraction module.

Uses SpeechBrain ECAPA-TDNN model for speaker recognition.
"""

from typing import List, Optional

import numpy as np

# Try to import SpeechBrain, but allow fallback for testing
try:
    import torch
    import torchaudio
    from speechbrain.inference import EncoderClassifier

    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False


# Embedding dimension for ECAPA-TDNN
EMBEDDING_DIM = 192


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score (-1.0 to 1.0)
    """
    # Ensure 1D arrays
    e1 = embedding1.flatten()
    e2 = embedding2.flatten()

    # Calculate cosine similarity
    dot_product = np.dot(e1, e2)
    norm1 = np.linalg.norm(e1)
    norm2 = np.linalg.norm(e2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def interpret_similarity_score(score: float) -> str:
    """
    Interpret similarity score into confidence level.

    Args:
        score: Cosine similarity score (0.0 to 1.0 for voice matching)

    Returns:
        Confidence level string
    """
    if score >= 0.9:
        return "very_high"
    elif score >= 0.8:
        return "high"
    elif score >= 0.7:
        return "medium"
    elif score >= 0.5:
        return "low"
    else:
        return "very_low"


class EmbeddingExtractor:
    """
    Extracts speaker embeddings from audio.

    Uses SpeechBrain's ECAPA-TDNN model trained on VoxCeleb.
    Produces 192-dimensional normalized embeddings.
    """

    MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"
    EMBEDDING_DIM = 192

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the embedding extractor.

        Args:
            device: Device to run model on ('cpu', 'cuda', or None for auto)
        """
        self._model: Optional[EncoderClassifier] = None
        self._device = device

        if SPEECHBRAIN_AVAILABLE:
            self._init_model()
        else:
            # For testing without SpeechBrain
            self._use_mock = True

    def _init_model(self):
        """Initialize the SpeechBrain model."""
        if not SPEECHBRAIN_AVAILABLE:
            return

        device = self._device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = EncoderClassifier.from_hparams(
            source=self.MODEL_NAME,
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        self._use_mock = False

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract embedding from preprocessed audio.

        Args:
            audio: Preprocessed audio (float32, 16kHz, mono)

        Returns:
            192-dimensional normalized embedding (float32)
        """
        if hasattr(self, "_use_mock") and self._use_mock:
            return self._mock_extract(audio)

        if self._model is None:
            self._init_model()

        # Convert to tensor
        audio_tensor = torch.tensor(audio).unsqueeze(0)

        # Extract embedding
        with torch.no_grad():
            embedding = self._model.encode_batch(audio_tensor)
            embedding = embedding.squeeze().cpu().numpy()

        # Normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.astype(np.float32)

    def extract_batch(self, audios: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings from multiple audio samples.

        Args:
            audios: List of preprocessed audio arrays

        Returns:
            Array of shape (n_samples, 192)
        """
        if hasattr(self, "_use_mock") and self._use_mock:
            return np.array([self._mock_extract(audio) for audio in audios])

        if self._model is None:
            self._init_model()

        # Pad to same length
        max_len = max(len(audio) for audio in audios)
        padded = np.zeros((len(audios), max_len), dtype=np.float32)
        for i, audio in enumerate(audios):
            padded[i, : len(audio)] = audio

        # Convert to tensor
        audio_tensor = torch.tensor(padded)

        # Extract embeddings
        with torch.no_grad():
            embeddings = self._model.encode_batch(audio_tensor)
            embeddings = embeddings.squeeze().cpu().numpy()

        # Handle single sample case
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        # Normalize each embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        return embeddings.astype(np.float32)

    def _mock_extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Generate mock embedding for testing without SpeechBrain.

        Uses audio characteristics to create deterministic pseudo-embedding.
        """
        # Create deterministic embedding based on audio features
        np.random.seed(int(np.sum(np.abs(audio[:1000])) * 1000) % 2**31)
        embedding = np.random.randn(self.EMBEDDING_DIM).astype(np.float32)

        # Normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding


class SimilaritySearch:
    """
    Performs similarity search on speaker embeddings.

    For production use with pgvector, this class provides
    the interface for vector similarity operations.
    """

    # Thresholds
    VERIFICATION_THRESHOLD = 0.7
    IDENTIFICATION_THRESHOLD = 0.5

    @staticmethod
    def find_most_similar(
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        candidate_ids: List[str],
        threshold: float = 0.5,
        limit: int = 10,
    ) -> List[dict]:
        """
        Find most similar embeddings to query.

        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embeddings
            candidate_ids: List of candidate IDs
            threshold: Minimum similarity threshold
            limit: Maximum results to return

        Returns:
            List of matches with speaker_id and score
        """
        results = []

        for emb, speaker_id in zip(candidate_embeddings, candidate_ids):
            score = cosine_similarity(query_embedding, emb)
            if score >= threshold:
                results.append({
                    "speaker_id": speaker_id,
                    "score": score,
                    "confidence": interpret_similarity_score(score),
                })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:limit]
