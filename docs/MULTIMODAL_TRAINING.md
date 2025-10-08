# Multi-Modal Training Guide for Igris

## Overview

Igris now supports multi-modal training, allowing you to train a single model on multiple types of tasks:
- **Text Generation**: Stories, poetry, dialogues, descriptions
- **Code Generation**: Python, C, JavaScript, and other programming languages
- **Image Prompt Generation**: Text-to-image descriptions and prompts

## Architecture

### Task-Aware Tokenization
- Special task tokens are prepended to each training example:
  - `<TASK_TEXT>` (Token ID: 0) for text generation
  - `<TASK_CODE>` (Token ID: 1) for code generation  
  - `<TASK_IMAGE>` (Token ID: 2) for image prompt generation
- The model learns to associate these tokens with specific generation patterns

### Unified Model
- Single GPT model trained on all task types simultaneously
- Task prefix guides the model's generation behavior
- Shared parameters allow knowledge transfer between tasks

## Dataset Format

### JSONL Format
Each line in your dataset should be a JSON object with the following structure:

```json
{
  "task": "text|code|image",
  "instruction": "What the model should do",
  "input": "Optional input data or prompt",
  "output": "Expected response"
}
```

### Example Entries

**Text Generation:**
```json
{"task": "text", "instruction": "Write a short story about a robot learning to dance.", "input": "", "output": "Once upon a time, a little robot named Igris learned to dance in a rainbow-colored forest..."}
```

**Code Generation:**
```json
{"task": "code", "instruction": "Write a Python function to reverse a string.", "input": "", "output": "def reverse_string(s):\n    return s[::-1]"}
```

**Image Prompt Generation:**
```json
{"task": "image", "instruction": "Generate an image of a futuristic city skyline.", "input": "", "output": "A breathtaking futuristic city skyline with towering glass and steel buildings..."}
```

## Usage

### Training

#### Train on JSONL Dataset
```bash
cargo run --release -- train --dataset datasets/mixed_multimodal.jsonl --dataset-format jsonl
```

#### Train on Plain Text (Backward Compatibility)
```bash
cargo run --release -- train --dataset dataset.txt --dataset-format text
```

### Inference

#### Text Generation
```bash
cargo run --release -- infer --task text --prompt "Write a story about a space adventure"
```

#### Code Generation
```bash
cargo run --release -- infer --task code --prompt "Write a Python function to calculate fibonacci numbers"
```

#### Image Prompt Generation
```bash
cargo run --release -- infer --task image --prompt "Generate an image of a magical forest"
```

### GPU Training
Add the `--features gpu` flag for GPU acceleration:
```bash
cargo run --release --features gpu -- train --dataset datasets/mixed_multimodal.jsonl --dataset-format jsonl
```

## Sample Datasets

The `datasets/` directory contains sample JSONL files:

- `sample_text.jsonl` - Text generation examples
- `sample_code.jsonl` - Code generation examples  
- `sample_image.jsonl` - Image prompt generation examples
- `mixed_multimodal.jsonl` - Mixed task examples

## Training Process

### 1. Dataset Loading
- JSONL files are parsed into `MultiModalDataset` structs
- Each example is converted to a training sequence with task prefix
- Text is tokenized using the task-aware tokenizer

### 2. Training Sequence Format
For each example, the training sequence becomes:
```
<TASK_X> + instruction + input + output
```

### 3. Model Training
- Uses existing GPT training loop with multimodal dataset sampling
- Task tokens are learned as part of the vocabulary
- Model learns to associate task tokens with generation patterns

## Configuration

### Model Parameters
Default configuration in `src/main.rs`:
- Context window: 64 tokens
- Embedding dimension: 64
- Number of layers: 4
- Number of attention heads: 4
- Batch size: 32

### Learning Rate Schedule
- Base learning rate: 0.001
- Minimum learning rate: 0.00001
- Warmup steps: 100
- Decay steps: 50000

## Best Practices

### Dataset Creation
1. **Balanced Tasks**: Include roughly equal numbers of examples for each task type
2. **Quality Examples**: Use high-quality, diverse examples for each task
3. **Clear Instructions**: Write clear, specific instructions for each example
4. **Consistent Format**: Maintain consistent JSONL format across all examples

### Training Tips
1. **Start Small**: Begin with smaller datasets to test the setup
2. **Monitor Loss**: Watch for task-specific loss patterns during training
3. **Validation**: Test inference on each task type during training
4. **Checkpoints**: Save model checkpoints regularly for resumable training

### Inference Tips
1. **Task Selection**: Always specify the correct task type for inference
2. **Temperature**: Adjust temperature based on task (lower for code, higher for creative text)
3. **Prompt Quality**: Provide clear, specific prompts for better results

## Troubleshooting

### Common Issues

**"Invalid task type" Error**
- Ensure task field is exactly "text", "code", or "image"
- Check JSONL format for syntax errors

**Poor Generation Quality**
- Verify dataset quality and diversity
- Check if model has trained long enough
- Ensure task tokens are being used correctly

**Memory Issues**
- Reduce batch size for large datasets
- Use CPU training if GPU memory is insufficient
- Consider smaller context window

### Debugging
- Enable verbose logging to see task token usage
- Check vocabulary size includes task tokens (+3)
- Verify JSONL parsing with sample files

## Advanced Usage

### Custom Task Types
To add new task types:
1. Add new task token in `src/tokenizer/task_aware.rs`
2. Update task mapping in `src/dataset.rs`
3. Add new task handling in training/inference code

### Fine-tuning
- Train on specific task types by filtering datasets
- Use different learning rates for different tasks
- Implement task-specific adapters for specialized behavior

### Evaluation
- Create separate validation sets for each task type
- Measure task-specific metrics (BLEU for text, syntax correctness for code)
- Monitor cross-task knowledge transfer

## Examples

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

## Future Enhancements

- Support for more task types (translation, summarization, etc.)
- Task-specific model heads for specialized behavior
- Automatic dataset validation and quality metrics
- Integration with external datasets (GitHub, Wikipedia, etc.)
- Real image generation (beyond text-to-image prompts)

## Contributing

When contributing to the multi-modal training system:
1. Maintain backward compatibility with plain text training
2. Add tests for new functionality
3. Update documentation for new features
4. Follow existing code patterns and conventions
