# 🚀 Igris – Quick Start Training Guide

## Overview

This guide provides a streamlined, step-by-step approach to training your Igris LLM model with multi-modal capabilities. Whether you're new to language models or looking for a quick reference, this guide will get you from dataset preparation to trained model in the shortest time possible.

**What you'll learn:**
- How to prepare datasets for text, code, and image prompt generation
- Optimal hyperparameters and training configurations
- GPU optimization techniques for faster training
- Step-by-step training and inference workflows

## 1️⃣ Dataset Preparation

### A. Text Generation

**Sources:** Wikipedia, Books, Stories, News, Blogs

**Format:** JSONL
```json
{"task": "text", "instruction": "Write a short story about a robot learning to dance.", "input": "", "output": "Once upon a time, a robot named Igris learned to dance in a rainbow-colored forest..."}
```

**Tips:**
- Clean data (remove HTML, special chars)
- Split long text into smaller sequences (< context window)
- Maintain diverse writing styles and genres
- Include various text types: narratives, dialogues, descriptions, summaries

### B. Code Generation

**Sources:** GitHub, competitive programming datasets, open-source repos

**Format:** JSONL
```json
{"task": "code", "instruction": "Write Python function to reverse a string.", "input": "", "output": "def reverse_string(s): return s[::-1]"}
```

**Tips:**
- Include multiple languages (Python, C, JavaScript, Rust, etc.)
- Keep examples concise but functional
- Cover different programming paradigms
- Include error handling and edge cases
- Add comments and documentation examples

### C. Image Prompt Generation

**Sources:** LAION, COCO, or custom captions/images

**Format:** JSONL (prompt → description/URL)
```json
{"task": "image", "instruction": "Generate an image of a futuristic city skyline.", "input": "", "output": "https://example.com/futuristic_city.jpg"}
```

**Tip:** For future image generation, consider storing CLIP embeddings instead of URLs

## 2️⃣ Dataset Size Recommendations

| Task | Examples | Notes |
|------|----------|-------|
| Text | 50k–500k | Diverse stories, summaries, quotes |
| Code | 20k–100k | Multi-language snippets |
| Image | 10k–50k | Text-to-image pairs |

**Start smaller for testing (~1k examples per task) before scaling.**

## 3️⃣ Batching & Data Loading

### Task-Aware Batching
- Separate batches per task or mix randomly
- Prepend task prefix token (`<TASK_TEXT>`, `<TASK_CODE>`, `<TASK_IMAGE>`)
- Maintain balanced representation across tasks

### Batch Size Recommendations
- **CPU:** 8–16 sequences per batch
- **GPU (16GB+):** 32–64 sequences per batch
- **GPU (8GB):** 16–32 sequences per batch

### Sequence Length / Context Window
- **Text:** 64–128 tokens
- **Code:** 128–256 tokens
- **Image prompts:** 32–64 tokens (for description strings)

## 4️⃣ Hyperparameters

| Parameter | Suggested Range | Notes |
|-----------|----------------|-------|
| Learning rate | 2e-5 – 5e-5 | Lower for code, higher for creative text |
| Epochs | 3–10 | Scale with dataset size |
| Gradient clipping | 1.0 | Prevents exploding gradients |
| Optimizer | AdamW | With weight decay |
| Weight decay | 0.01 | Regularization |
| Dropout | 0.1 – 0.2 | Higher for larger models |
| Temperature (inference) | 0.7–1.0 | Lower for code, higher for creativity |
| Top-k / Top-p | k=50, p=0.9 | Nucleus sampling |

**Fine-tune learning rate per task if you notice instability (code usually needs lower LR).**

## 5️⃣ Training Commands

### CPU Training (Multimodal)
```bash
cargo run --release -- train --dataset datasets/mixed_multimodal.jsonl --dataset-format jsonl
```

### GPU Training
```bash
cargo run --release --features gpu -- train --dataset datasets/mixed_multimodal.jsonl --dataset-format jsonl
```

### Plain Text Backward-Compatible Training
```bash
cargo run --release -- train --dataset dataset.txt --dataset-format text
```

### Custom Training Parameters
```bash
# Large dataset training
cargo run --release --features gpu -- train \
  --dataset large_multimodal_dataset.jsonl \
  --dataset-format jsonl \
  --model large_model.dat
```

## 6️⃣ Inference Commands

### Text Generation
```bash
cargo run --release -- infer --task text --prompt "Write a story about AI"
```

### Code Generation
```bash
cargo run --release -- infer --task code --prompt "Write Python function to merge two lists"
```

### Image Prompt Generation
```bash
cargo run --release -- infer --task image --prompt "Generate an image of a flying car"
```

### Advanced Inference Options
```bash
# Custom temperature and length
cargo run --release -- infer \
  --task text \
  --prompt "Write a poem about space" \
  --count 200 \
  --temperature 0.8
```

## 7️⃣ GPU Optimization Tips

### OpenCL Backend
- Use `--features gpu` for faster matrix operations
- Ensure OpenCL drivers are properly installed
- Works with both NVIDIA and AMD GPUs

### Memory Management
- **Gradient accumulation:** Combine small batches if GPU memory is limited
- **Mixed precision (f32 → f16):** Reduces memory usage and speeds up training
- **Batch size tuning:** Start with 32, increase until memory limit

### Performance Optimization
- **Checkpointing:** Save model every N steps to avoid loss in long runs
- **Profiling:** Measure tensor allocations and training speed
- **Parallel processing:** Leverage multiple GPU cores effectively

### Memory Usage Guidelines
| GPU Memory | Recommended Batch Size | Context Length |
|------------|----------------------|----------------|
| 8GB | 16-32 | 64-128 |
| 16GB | 32-64 | 128-256 |
| 24GB+ | 64-128 | 256-512 |

## 8️⃣ Scaling Tips

### Dataset Scaling
1. **Start small:** Begin with 1k examples per task for testing
2. **Validate quality:** Ensure good results before scaling
3. **Gradual increase:** Scale dataset size incrementally
4. **Monitor performance:** Watch for overfitting and quality degradation

### Model Scaling
- **Context window:** Adjust for code (longer sequences needed)
- **Vocabulary size:** Expand for multilingual code/text
- **Model depth:** Increase layers for complex patterns

### Infrastructure Scaling
- **Distributed training:** Consider multi-GPU setups for large datasets
- **Data pipeline:** Optimize data loading and preprocessing
- **Storage:** Use fast SSDs for large dataset storage

## 9️⃣ Recommended Workflow

### Phase 1: Setup & Testing
1. **Prepare datasets** → JSONL with task prefix
2. **Test on small sample** (10–50 examples per task)
3. **Validate pipeline** → Ensure data loading works correctly

### Phase 2: Initial Training
1. **Train on CPU/GPU** → Monitor loss and sample outputs
2. **Fine-tune hyperparameters** per task
3. **Evaluate quality** → Check generation quality for each task

### Phase 3: Production Training
1. **Expand dataset size** → Scale to full multimodal dataset
2. **Long-term training** → Train for multiple epochs
3. **Continuous evaluation** → Regular quality checks

### Phase 4: Optimization
1. **Performance tuning** → Optimize for your hardware
2. **Model refinement** → Adjust architecture if needed
3. **Production deployment** → Deploy for inference

## 🔟 Quick Start Example

### Complete Training Session
```bash
# 1. Prepare your dataset in JSONL format
# 2. Start training
cargo run --release -- train --dataset my_multimodal_dataset.jsonl --dataset-format jsonl

# 3. Test inference on different tasks
cargo run --release -- infer --task text --prompt "Write a poem about AI"
cargo run --release -- infer --task code --prompt "Write a Python function to sort a list"
cargo run --release -- infer --task image --prompt "Generate an image of a robot"
```

### GPU Training Example
```bash
# Install OpenCL drivers first
cargo run --release --features gpu -- train --dataset datasets/mixed_multimodal.jsonl --dataset-format jsonl
```

## 🚨 Troubleshooting

### Common Issues

**Out of Memory (OOM)**
- Reduce batch size
- Decrease context window
- Use gradient accumulation
- Enable mixed precision

**Poor Generation Quality**
- Check dataset quality and diversity
- Adjust learning rate
- Increase training time
- Verify task token usage

**Training Instability**
- Lower learning rate
- Increase gradient clipping
- Check data preprocessing
- Monitor loss curves

### Performance Issues
- Profile GPU utilization
- Check OpenCL driver compatibility
- Optimize data loading pipeline
- Consider CPU training for small datasets

## 📊 Monitoring & Logging

### Training Metrics
- Loss per task type
- Generation quality samples
- Training speed (tokens/second)
- Memory usage

### Evaluation Strategy
- Regular inference testing
- Task-specific quality metrics
- Cross-task knowledge transfer
- Long-term stability checks

## 🔮 Next Steps

After completing this quick start guide:

1. **Explore Advanced Features:** See [Advanced Training Guide](ADVANCED_TRAINING_GUIDE.md) for production-level strategies
2. **Multi-Modal Deep Dive:** Check [Multi-Modal Training Guide](MULTIMODAL_TRAINING.md) for detailed setup
3. **Custom Tasks:** Add your own task types by extending the tokenizer
4. **Scale Up:** Move to larger datasets and longer training runs
5. **Deploy:** Set up inference endpoints for production use

## 📚 Additional Resources

- [Multi-Modal Training Guide](MULTIMODAL_TRAINING.md) - Detailed multi-modal setup
- [Advanced Training Guide](ADVANCED_TRAINING_GUIDE.md) - Production-level optimization
- [Igris GitHub Repository](https://github.com/JithinGK51/igris_LLM) - Source code and issues
- [OpenCL Documentation](https://www.khronos.org/opencl/) - GPU acceleration setup

---

*This guide is designed for users who want to get started quickly with Igris training. For comprehensive details, see the [Advanced Training Guide](ADVANCED_TRAINING_GUIDE.md) and [Multi-Modal Training Guide](MULTIMODAL_TRAINING.md).*
