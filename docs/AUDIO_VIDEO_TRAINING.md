# Audio & Video Training Guide for Igris

## Overview

Igris now supports audio and video generation tasks through a framework implementation that uses precomputed embeddings. This guide explains how to train and use the model for audio and video generation tasks.

## Architecture

### Task-Aware Tokenization

The multi-modal system now includes two additional task types:
- `<TASK_AUDIO>` (Token ID: 3) for audio generation
- `<TASK_VIDEO>` (Token ID: 4) for video generation

### Loss Functions

- **Text/Code/Image**: Cross-entropy loss (token prediction)
- **Audio/Video**: MSE (Mean Squared Error) loss (embedding prediction)

### Embedding-Based Training

Audio and video tasks use precomputed embeddings instead of token sequences:
- **Audio embeddings**: 128-dimensional vectors (configurable)
- **Video embeddings**: 256-dimensional vectors (configurable)

## Dataset Format

### JSONL Format with Embeddings

Each line should be a JSON object with the following structure:

```json
{
  "task": "audio|video",
  "instruction": "Description of what to generate",
  "input": "Optional context",
  "output": "Text description (for logging/display)",
  "embedding": [0.1, 0.2, 0.3, ...] // Precomputed embedding vector
}
```

### Example Entries

**Audio Generation:**
```json
{"task": "audio", "instruction": "Generate speech for 'Hello world'", "input": "", "output": "Audio: Hello world", "embedding": [0.1, 0.2, 0.3, ..., 0.5]}
```

**Video Generation:**
```json
{"task": "video", "instruction": "Generate a 5-second video of a flying drone", "input": "", "output": "Video: Flying drone", "embedding": [0.3, 0.4, 0.5, ..., 0.7]}
```

## Training

### Multi-Modal Training (All Tasks)

Train on a mixed dataset containing text, code, image, audio, and video examples:

```bash
# CPU Training
cargo run --release -- train --dataset datasets/mixed_multimodal_av.jsonl --dataset-format jsonl

# GPU Training (faster)
cargo run --release --features gpu -- train --dataset datasets/mixed_multimodal_av.jsonl --dataset-format jsonl
```

### Audio-Only Training

```bash
cargo run --release -- train --dataset datasets/sample_audio.jsonl --dataset-format jsonl
```

### Video-Only Training

```bash
cargo run --release -- train --dataset datasets/sample_video.jsonl --dataset-format jsonl
```

## Inference

### Audio Generation

```bash
cargo run --release -- infer --task audio --prompt "Generate speech for 'Hello world'" --count 100 --temperature 0.7
```

### Video Generation

```bash
cargo run --release -- infer --task video --prompt "Generate a video of a sunset" --count 100 --temperature 0.7
```

## Embedding Extraction

This is a framework implementation. To use it with real audio/video data, you need to:

1. **Extract embeddings from your audio/video files**
   - Use pre-trained models like Wav2Vec2 for audio
   - Use pre-trained models like VideoMAE or CLIP for video

2. **Create your dataset**
   - Store embeddings as arrays in JSONL format
   - Keep embedding dimensions consistent within each task type

3. **Train the model**
   - The model learns to predict embeddings given text instructions
   - Loss is computed using MSE between predicted and target embeddings

## Technical Details

### Embedding Dimensions

Default dimensions (configurable in `src/gpt.rs`):
- Audio: 128 dimensions
- Video: 256 dimensions

### Model Output

For audio/video tasks, the model outputs:
- During training: Learns to predict embedding vectors
- During inference: Generates token sequences (framework mode)

### Converting to Real Generation

To convert this framework into a full audio/video generation system:

1. Add encoder/decoder models for audio/video
2. Modify inference to output actual embedding vectors
3. Pass embeddings to decoder models (vocoder for audio, video decoder for video)
4. Generate actual audio/video files

## Sample Datasets

### Provided Examples

- `datasets/sample_audio.jsonl`: Audio generation examples with dummy embeddings
- `datasets/sample_video.jsonl`: Video generation examples with dummy embeddings  
- `datasets/mixed_multimodal_av.jsonl`: Combined multi-modal dataset

### Creating Your Own Datasets

```python
import json
import numpy as np

# Example: Create audio dataset
audio_data = {
    "task": "audio",
    "instruction": "Generate speech for 'Custom text'",
    "input": "",
    "output": "Audio: Custom text",
    "embedding": your_audio_embedding.tolist()  # 128-dim vector
}

with open('custom_audio.jsonl', 'a') as f:
    f.write(json.dumps(audio_data) + '\n')
```

## Performance Considerations

### Memory Usage

Embedding-based tasks require additional memory for:
- Storing embedding vectors in dataset
- MSE loss computation
- Embedding target tensors

### Training Speed

- Audio/video training is similar to text/code/image training
- GPU acceleration significantly improves performance
- Batch size may need to be reduced for large embedding dimensions

## Hyperparameters

### Recommended Settings

For audio/video training:
- **Learning Rate**: 2e-5 to 5e-5 (lower than text-only training)
- **Batch Size**: 4-16 (smaller due to embedding memory)
- **Temperature**: 0.7-0.9 for generation
- **Context Window**: 64 tokens (standard)

## Troubleshooting

### Common Issues

**1. Out of Memory**
- Reduce batch size
- Use smaller embedding dimensions
- Enable gradient accumulation

**2. High Loss Values**
- Normalize embedding values to [-1, 1] range
- Adjust learning rate
- Check embedding dimensions are consistent

**3. Poor Generation Quality**
- Increase dataset size
- Use higher quality embeddings from better pre-trained models
- Train for more iterations

## Future Enhancements

Potential improvements for full audio/video generation:

1. **Real Embedding Prediction**: Modify model to output actual embedding vectors
2. **Decoder Integration**: Add vocoder for audio, video decoder for video
3. **Multi-Scale Embeddings**: Support variable-length sequences for longer audio/video
4. **Latent Diffusion**: Integrate diffusion models for better generation quality
5. **Real-Time Generation**: Optimize for streaming audio/video generation

## References

- **Audio**: Wav2Vec2, HuBERT for audio embeddings
- **Video**: VideoMAE, CLIP, TimeSformer for video embeddings
- **Vocoders**: HiFi-GAN, WaveGlow for audio synthesis
- **Video Synthesis**: Video Diffusion Models, Make-A-Video

## Support

For questions or issues related to audio/video training:
- Check the main [README](../README.md) for general setup
- See [MULTIMODAL_TRAINING.md](MULTIMODAL_TRAINING.md) for multi-modal basics
- Review [ADVANCED_TRAINING_GUIDE.md](ADVANCED_TRAINING_GUIDE.md) for optimization tips

