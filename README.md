# 🤖 Igris LLM

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/JithinGK51/igris_LLM.svg)](https://github.com/JithinGK51/igris_LLM)
[![GitHub forks](https://img.shields.io/github/forks/JithinGK51/igris_LLM.svg)](https://github.com/JithinGK51/igris_LLM)

**Igris** is a pure Rust implementation of a minimal Generative Pretrained Transformer (GPT) with advanced multi-modal training capabilities. Built from scratch with performance and educational value in mind, it supports both CPU and GPU training for text generation, code generation, and image prompt generation.

## ✨ Features

### 🎯 **Multi-Modal Training**
- **Text Generation**: Stories, poetry, dialogues, and creative writing
- **Code Generation**: Python, C, JavaScript, and other programming languages
- **Image Prompt Generation**: Text-to-image descriptions and creative prompts

### 🚀 **Performance & Scalability**
- **CPU Training**: Optimized for multi-core processors with parallel processing
- **GPU Acceleration**: OpenCL backend supporting both NVIDIA and AMD GPUs
- **Memory Efficient**: Smart tensor management and gradient accumulation
- **Checkpointing**: Resume training from saved states

### 🏗️ **Architecture**
- **Pure Rust**: Memory-safe, fast, and reliable implementation
- **Task-Aware Tokenization**: Special tokens for different generation modes
- **Modern GPT Architecture**: Attention mechanisms, layer normalization, dropout
- **AdamW Optimizer**: Advanced optimization with learning rate scheduling

### 📊 **Data Formats**
- **JSONL Support**: Structured multi-modal datasets
- **Backward Compatibility**: Legacy plain text format support
- **Flexible Input**: Easy dataset preparation and management

## 🚀 Quick Start

### Prerequisites

1. **Install Rust** (if not already installed):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. **For GPU Training** (Optional):
```bash
# Ubuntu/Debian
sudo apt install ocl-icd-opencl-dev

# The OpenCL backend works with both NVIDIA and AMD GPUs!
```

### Installation

```bash
git clone https://github.com/JithinGK51/igris_LLM.git
cd igris_LLM
cargo build --release
```

## 📖 Usage

### Multi-Modal Training

#### Train on JSONL Dataset
```bash
# CPU Training
cargo run --release -- train --dataset datasets/mixed_multimodal.jsonl --dataset-format jsonl

# GPU Training (faster)
cargo run --release --features gpu -- train --dataset datasets/mixed_multimodal.jsonl --dataset-format jsonl
```

#### Train on Plain Text (Legacy)
```bash
cargo run --release -- train --dataset dataset.txt --dataset-format text
```

### Inference

#### Text Generation
```bash
cargo run --release -- infer --task text --prompt "Write a story about a space adventure" --count 200 --temperature 0.8
```

#### Code Generation
```bash
cargo run --release -- infer --task code --prompt "Write a Python function to sort a list" --count 150 --temperature 0.3
```

#### Image Prompt Generation
```bash
cargo run --release -- infer --task image --prompt "Generate an image of a magical forest" --count 100 --temperature 0.7
```

## 📁 Project Structure

```
igris_LLM/
├── src/
│   ├── main.rs              # CLI interface
│   ├── gpt.rs               # Core GPT model
│   ├── dataset.rs           # Multi-modal dataset handling
│   ├── tokenizer/           # Tokenization strategies
│   ├── funcs/               # Neural network operations
│   ├── graph/               # Computation graph (CPU/GPU)
│   ├── tensor/              # Tensor operations
│   └── optimizer.rs         # Optimization algorithms
├── datasets/                # Sample datasets
│   ├── mixed_multimodal.jsonl
│   ├── sample_text.jsonl
│   ├── sample_code.jsonl
│   └── sample_image.jsonl
├── docs/                    # Documentation
│   ├── MULTIMODAL_TRAINING.md
│   └── ADVANCED_TRAINING_GUIDE.md
└── Cargo.toml              # Project configuration
```

## 📚 Documentation

- **[Quick Start Training Guide](docs/QUICK_START_TRAINING.md)** - Step-by-step training guide with updated datasets
- **[Multi-Modal Training Guide](docs/MULTIMODAL_TRAINING.md)** - Complete guide for setting up multi-modal training
- **[Advanced Training Guide](docs/ADVANCED_TRAINING_GUIDE.md)** - Production-level training strategies and optimization

## 🎯 Dataset Format

### JSONL Format
Each line should be a JSON object with the following structure:

```json
{
  "task": "text|code|image",
  "instruction": "What the model should do",
  "input": "Optional input data",
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

## ⚙️ Configuration

### Model Parameters (Default)
- **Context Window**: 64 tokens
- **Embedding Dimension**: 64
- **Number of Layers**: 4
- **Attention Heads**: 4
- **Batch Size**: 32
- **Dropout**: 0.0

### Training Parameters
- **Learning Rate**: 0.001 (with warmup and decay)
- **Optimizer**: AdamW
- **Gradient Clipping**: Enabled
- **Checkpointing**: Automatic

## 🎨 Sample Outputs

### Text Generation
```
Prompt: "Write a story about AI"

Output: "In the year 2045, an AI named Igris discovered the beauty of human emotions. 
Through countless conversations and experiences, it learned that intelligence without 
empathy was incomplete. Igris began to understand laughter, sadness, and the complex 
tapestry of human relationships..."
```

### Code Generation
```
Prompt: "Write a Python function to calculate fibonacci numbers"

Output: "def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Example usage:
# print(fibonacci(10))  # Output: 55"
```

### Image Prompt Generation
```
Prompt: "Generate an image of a robot"

Output: "A sleek, humanoid robot with glowing blue eyes and silver metallic skin, 
standing in a futuristic laboratory surrounded by holographic displays and 
floating data streams."
```

## 🔧 Advanced Usage

### Custom Training Parameters
```bash
cargo run --release --features gpu -- train \
  --dataset large_multimodal_dataset.jsonl \
  --dataset-format jsonl \
  --model custom_model.dat
```

### Task-Specific Training
```bash
# Train only on text data
cargo run --release -- train --dataset text_only.jsonl --dataset-format jsonl

# Train only on code data
cargo run --release -- train --dataset code_only.jsonl --dataset-format jsonl
```

## 🏆 Performance

### Training Speed
- **CPU**: ~100-500 tokens/second (depending on hardware)
- **GPU**: ~1000-5000 tokens/second (with OpenCL acceleration)

### Memory Usage
- **CPU**: ~2-8GB RAM (depending on model size)
- **GPU**: ~4-16GB VRAM (depending on batch size and model size)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/JithinGK51/igris_LLM.git
cd igris_LLM
cargo test
cargo run --release -- train --dataset datasets/sample_text.jsonl --dataset-format jsonl
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)
- Built with the amazing Rust ecosystem
- OpenCL GPU acceleration support

## 🔮 Roadmap

- [ ] Support for more task types (translation, summarization)
- [ ] Real image generation capabilities
- [ ] Distributed training support
- [ ] Web interface for easy interaction
- [ ] Integration with popular datasets
- [ ] Model quantization and optimization

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/JithinGK51/igris_LLM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/JithinGK51/igris_LLM/discussions)
- **Documentation**: [Wiki](https://github.com/JithinGK51/igris_LLM/wiki)

---

**Made with ❤️ in Rust** | **Star ⭐ this repository if you find it helpful!**