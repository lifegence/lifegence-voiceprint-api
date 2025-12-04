# Lifegence Voiceprint API - Technical Documentation

## Overview

This document explains the technical foundations of the voice recognition system used in Lifegence VoiceID SDK, including the neural network architecture, embedding extraction process, and similarity computation.

## Table of Contents

1. [Speaker Recognition Fundamentals](#speaker-recognition-fundamentals)
2. [ECAPA-TDNN Architecture](#ecapa-tdnn-architecture)
3. [Voice Embedding Extraction](#voice-embedding-extraction)
4. [Similarity Computation](#similarity-computation)
5. [Audio Preprocessing](#audio-preprocessing)
6. [Performance Characteristics](#performance-characteristics)
7. [Limitations and Considerations](#limitations-and-considerations)

---

## Speaker Recognition Fundamentals

### What is Speaker Recognition?

Speaker recognition is the task of identifying or verifying a person based on their voice characteristics. It differs from speech recognition (what is said) by focusing on who is speaking.

```
┌─────────────────────────────────────────────────────────────┐
│                    Speaker Recognition                       │
├─────────────────────────────┬───────────────────────────────┤
│       Verification (1:1)    │     Identification (1:N)      │
│                             │                               │
│  "Is this person X?"        │  "Who is this person?"        │
│                             │                               │
│  Input: Audio + Claimed ID  │  Input: Audio only            │
│  Output: Yes/No + Score     │  Output: Ranked candidates    │
└─────────────────────────────┴───────────────────────────────┘
```

### Voice Characteristics

Human voice contains multiple discriminative features:

| Category | Features | Stability |
|----------|----------|-----------|
| **Physiological** | Vocal tract shape, nasal cavity, vocal cord tension | Highly stable |
| **Behavioral** | Speaking rate, pitch patterns, accent | Moderately stable |
| **Linguistic** | Word choice, phrase patterns | Variable |

This SDK focuses on physiological and behavioral features encoded by the neural network.

---

## ECAPA-TDNN Architecture

### Model Selection

We use **ECAPA-TDNN** (Emphasized Channel Attention, Propagation and Aggregation in TDNN), a state-of-the-art speaker recognition model.

**Why ECAPA-TDNN?**

| Criterion | ECAPA-TDNN | x-vector | d-vector |
|-----------|------------|----------|----------|
| EER on VoxCeleb1 | **0.87%** | 3.30% | 4.19% |
| Embedding size | 192 | 512 | 256 |
| Robustness to noise | Excellent | Good | Moderate |
| Short utterance performance | Excellent | Moderate | Good |

### Architecture Details

```
┌─────────────────────────────────────────────────────────────────┐
│                      ECAPA-TDNN Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Mel-Filterbank Features (80-dim × T frames)             │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Conv1D Layer (1024 channels)                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │         SE-Res2Net Block × 3 (with skip connections)     │    │
│  │                                                          │    │
│  │   ┌──────────────┐                                       │    │
│  │   │   Res2Net    │  Multi-scale feature extraction      │    │
│  │   │   Module     │  (scales = 8)                         │    │
│  │   └──────────────┘                                       │    │
│  │          │                                               │    │
│  │          ▼                                               │    │
│  │   ┌──────────────┐                                       │    │
│  │   │  SE Block    │  Channel attention (r = 4)           │    │
│  │   │  (Squeeze &  │                                       │    │
│  │   │   Excite)    │                                       │    │
│  │   └──────────────┘                                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │         Multi-layer Feature Aggregation (MFA)            │    │
│  │         Concatenates outputs from all SE-Res2Net blocks  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │         Attentive Statistical Pooling (ASP)              │    │
│  │         Self-attention weighted mean + std               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Fully Connected (192-dim output)            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│              Output: 192-dimensional embedding                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. SE-Res2Net Blocks
- **Res2Net**: Processes features at multiple scales within a single block
- **Squeeze-and-Excitation (SE)**: Learns channel-wise attention weights
- Enables the network to focus on speaker-discriminative frequency bands

#### 2. Multi-layer Feature Aggregation (MFA)
- Concatenates features from all SE-Res2Net blocks
- Preserves both low-level and high-level speaker characteristics

#### 3. Attentive Statistical Pooling (ASP)
- Computes weighted mean and standard deviation across time
- Attention mechanism emphasizes speaker-relevant frames
- Robust to utterance length variations

---

## Voice Embedding Extraction

### Embedding Properties

| Property | Value | Description |
|----------|-------|-------------|
| **Dimensions** | 192 | Compact yet discriminative |
| **Normalization** | L2 (unit sphere) | Enables cosine similarity |
| **Data type** | float32 | 768 bytes per embedding |
| **Training data** | VoxCeleb1 + VoxCeleb2 | ~1M utterances, 7,000+ speakers |

### Extraction Process

```python
# Simplified extraction flow
def extract_embedding(audio: np.ndarray) -> np.ndarray:
    # 1. Convert to tensor
    audio_tensor = torch.tensor(audio).unsqueeze(0)  # Shape: (1, samples)

    # 2. Extract embedding via model forward pass
    with torch.no_grad():
        embedding = model.encode_batch(audio_tensor)  # Shape: (1, 192)

    # 3. L2 normalize to unit sphere
    embedding = embedding / np.linalg.norm(embedding)  # ||e|| = 1

    return embedding  # Shape: (192,)
```

### Why 192 Dimensions?

The 192-dimensional embedding space provides:

1. **Sufficient capacity**: Encodes ~7,000 distinct speakers in training
2. **Compact representation**: Only 768 bytes per speaker
3. **Fast similarity computation**: O(192) operations per comparison
4. **Scalable search**: Compatible with vector databases (pgvector)

```
┌──────────────────────────────────────────────────────────────┐
│            Embedding Space Visualization (2D projection)      │
│                                                               │
│              Speaker A (multiple enrollments)                 │
│                    ●  ●                                       │
│                   ●  ●  ●                                     │
│                                                               │
│                                    Speaker B                  │
│                                      ●  ●                     │
│                                        ●                      │
│                                                               │
│       Speaker C                                               │
│          ●                                                    │
│        ●  ●                      Speaker D                    │
│                                    ●  ●  ●                    │
│                                                               │
│  Note: Same speaker embeddings cluster together               │
│        Different speakers are well-separated                  │
└──────────────────────────────────────────────────────────────┘
```

---

## Similarity Computation

### Cosine Similarity

We use cosine similarity to compare embeddings:

```
                    A · B           Σᵢ(Aᵢ × Bᵢ)
cos(θ) = ────────────────── = ─────────────────────
              ||A|| ||B||      √(Σᵢ Aᵢ²) × √(Σᵢ Bᵢ²)
```

Since embeddings are L2-normalized (||A|| = ||B|| = 1):

```
cos(θ) = A · B = Σᵢ(Aᵢ × Bᵢ)
```

### Implementation

```python
def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two L2-normalized embeddings.

    For normalized vectors: cos(θ) = dot(A, B)

    Returns:
        Score in range [-1.0, 1.0]
        Typical same-speaker scores: 0.7 - 1.0
        Typical different-speaker scores: -0.2 - 0.5
    """
    return float(np.dot(embedding1.flatten(), embedding2.flatten()))
```

### Score Interpretation

| Score Range | Confidence | Typical Scenario |
|-------------|------------|------------------|
| 0.90 - 1.00 | `very_high` | Same recording or very similar conditions |
| 0.80 - 0.90 | `high` | Same speaker, different session |
| 0.70 - 0.80 | `medium` | Same speaker, different conditions (noise, emotion) |
| 0.50 - 0.70 | `low` | Possibly same speaker, recommend additional verification |
| < 0.50 | `very_low` | Likely different speakers |

### Threshold Selection

```
┌─────────────────────────────────────────────────────────────────┐
│           Score Distribution (Same vs Different Speakers)        │
│                                                                  │
│  Frequency                                                       │
│      │                                                           │
│      │        Different                Same                      │
│      │        Speakers                 Speaker                   │
│      │           ██                      ██                      │
│      │          ████                    ████                     │
│      │         ██████                  ██████                    │
│      │        ████████                ████████                   │
│      │       ██████████              ██████████                  │
│      │      ████████████            ████████████                 │
│      └──────────────────────────────────────────────── Score    │
│           0.0    0.3    0.5    0.7    0.8    0.9    1.0         │
│                              ↑                                   │
│                     Default Threshold (0.7)                      │
│                                                                  │
│  FAR (False Accept Rate): ~1% at threshold 0.7                  │
│  FRR (False Reject Rate): ~5% at threshold 0.7                  │
└─────────────────────────────────────────────────────────────────┘
```

**Threshold recommendations:**

| Use Case | Threshold | Priority |
|----------|-----------|----------|
| High security (banking) | 0.80 - 0.85 | Low FAR |
| Standard verification | 0.70 (default) | Balanced |
| Convenience-focused | 0.60 - 0.65 | Low FRR |

---

## Audio Preprocessing

### Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                   Audio Preprocessing Pipeline                │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Raw Audio (WAV/MP3/WebM/OGG)                                │
│            │                                                  │
│            ▼                                                  │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  1. Format Decoding (librosa/soundfile)              │     │
│  │     - Decode compressed formats                       │     │
│  │     - Handle various bit depths                       │     │
│  └─────────────────────────────────────────────────────┘     │
│            │                                                  │
│            ▼                                                  │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  2. Resampling → 16,000 Hz                           │     │
│  │     - Standard rate for speech processing            │     │
│  │     - Preserves frequencies up to 8kHz (Nyquist)     │     │
│  └─────────────────────────────────────────────────────┘     │
│            │                                                  │
│            ▼                                                  │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  3. Channel Conversion → Mono                        │     │
│  │     - Average stereo channels                         │     │
│  │     - Consistent input format                         │     │
│  └─────────────────────────────────────────────────────┘     │
│            │                                                  │
│            ▼                                                  │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  4. Normalization → float32 [-1.0, 1.0]              │     │
│  │     - Peak normalization                              │     │
│  │     - Prevents clipping                               │     │
│  └─────────────────────────────────────────────────────┘     │
│            │                                                  │
│            ▼                                                  │
│  Preprocessed Audio: float32, 16kHz, mono                    │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Audio Requirements

| Parameter | Requirement | Reason |
|-----------|-------------|--------|
| Duration | 3 - 30 seconds | Minimum for reliable embedding, maximum for efficiency |
| Optimal duration | 5 - 10 seconds | Best quality-to-processing ratio |
| Sample rate | Any (resampled to 16kHz) | Standard for speech |
| Channels | Any (converted to mono) | Model expects single channel |
| Formats | WAV, MP3, WebM, OGG | Common audio formats |

### Quality Scoring

```python
def calculate_quality_score(audio: np.ndarray, sample_rate: int) -> float:
    """
    Estimate audio quality for speaker recognition.

    Factors considered:
    - Signal-to-noise ratio (SNR)
    - Speech activity ratio
    - Clipping detection
    - Duration appropriateness
    """
    score = 1.0

    # Penalize very short/long recordings
    duration = len(audio) / sample_rate
    if duration < 5:
        score *= 0.8
    elif duration > 20:
        score *= 0.9

    # Penalize low energy (possible silence/noise)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 0.01:
        score *= 0.7

    # Penalize clipping
    clipping_ratio = np.mean(np.abs(audio) > 0.99)
    if clipping_ratio > 0.01:
        score *= 0.8

    return min(1.0, max(0.0, score))
```

---

## Performance Characteristics

### Accuracy Metrics

Evaluated on VoxCeleb1 test set:

| Metric | Value | Description |
|--------|-------|-------------|
| EER | 0.87% | Equal Error Rate |
| minDCF (p=0.01) | 0.0941 | Minimum Detection Cost Function |
| minDCF (p=0.001) | 0.1734 | At lower prior probability |

### Processing Speed

Benchmarks on typical hardware:

| Hardware | Enrollment | Verification | Identification (1000 speakers) |
|----------|------------|--------------|--------------------------------|
| CPU (Intel i7) | ~500ms | ~300ms | ~350ms |
| GPU (NVIDIA T4) | ~100ms | ~80ms | ~120ms |
| GPU (NVIDIA A100) | ~50ms | ~40ms | ~60ms |

### Memory Requirements

| Component | Memory |
|-----------|--------|
| Model weights | ~85 MB |
| Per embedding | 768 bytes |
| 1M speakers | ~750 MB |

---

## Limitations and Considerations

### Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **No liveness detection** | Vulnerable to replay attacks | Implement challenge-response |
| **Language dependency** | Trained primarily on English | Test with target languages |
| **Noise sensitivity** | Degraded performance in noisy environments | Apply noise reduction |
| **Short utterances** | Lower accuracy under 3 seconds | Enforce minimum duration |

### Environmental Factors

```
┌─────────────────────────────────────────────────────────────────┐
│           Factors Affecting Recognition Accuracy                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  High Impact (> 10% accuracy change):                           │
│  ├── Background noise level                                      │
│  ├── Microphone quality difference (enrollment vs test)          │
│  └── Health conditions (cold, sore throat)                       │
│                                                                  │
│  Medium Impact (5-10% accuracy change):                          │
│  ├── Emotional state                                             │
│  ├── Speaking rate variation                                     │
│  └── Room acoustics (reverb)                                     │
│                                                                  │
│  Low Impact (< 5% accuracy change):                              │
│  ├── Time of day                                                 │
│  ├── Minor background sounds                                     │
│  └── Slight volume differences                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Best Practices

1. **Enrollment**
   - Use high-quality microphone
   - Record in quiet environment
   - Capture 5-10 seconds of natural speech
   - Consider multiple enrollment samples

2. **Verification**
   - Match recording conditions to enrollment
   - Use same or similar microphone
   - Ensure clear speech without interruptions

3. **Threshold Tuning**
   - Collect same-speaker and different-speaker pairs
   - Plot score distributions
   - Select threshold based on FAR/FRR requirements

---

## References

1. Desplanques, B., Thienpondt, J., & Demuynck, K. (2020). ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification. *Interspeech 2020*.

2. Nagrani, A., Chung, J. S., & Zisserman, A. (2017). VoxCeleb: A Large-scale Speaker Identification Dataset. *Interspeech 2017*.

3. SpeechBrain: A PyTorch-based Speech Toolkit. https://speechbrain.github.io/

4. Snyder, D., Garcia-Romero, D., Sell, G., Povey, D., & Khudanpur, S. (2018). X-vectors: Robust DNN Embeddings for Speaker Recognition. *ICASSP 2018*.

---

## Appendix: Embedding Vector Example

```python
# Example 192-dimensional embedding (truncated for display)
embedding = np.array([
    -0.0234,  0.0891,  0.0456, -0.0123,  0.0678,  # dims 0-4
     0.0345, -0.0567,  0.0234,  0.0891, -0.0345,  # dims 5-9
    # ... 182 more dimensions ...
], dtype=np.float32)

# Properties
print(f"Shape: {embedding.shape}")        # (192,)
print(f"Norm: {np.linalg.norm(embedding)}")  # 1.0 (normalized)
print(f"Size: {embedding.nbytes} bytes")  # 768 bytes
```
