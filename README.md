# Nameless Vector - Verb Outcome Generator

A Rust application that generates comprehensive verb outcomes using quantized language models. Built with the Candle ML framework for efficient inference with GGUF models.

## Features

- **Multi-Model Support**: Llama-2, Mistral, DeepSeek, CodeLlama, Phi-3, Qwen, Gemma
- **GPU Acceleration**: CUDA, Metal, and Accelerate framework support
- **Robust Tokenizer Discovery**: Automatic tokenizer loading with fallback mechanisms
- **State Management**: Resume interrupted processing with persistent state
- **Concurrent Processing**: Async/await with optimized batch sizes
- **Progress Tracking**: Real-time progress bars and performance metrics

## Quick Start

### 1. Prerequisites

- Rust 1.70+ with Cargo
- GPU drivers (optional but recommended):
  - NVIDIA: CUDA Toolkit 11.8+
  - Apple Silicon: macOS 12+

### 2. Setup

```bash
git clone <repository-url>
cd nameless_vector
```

### 3. Model Setup

Place your GGUF model file in `./src/models/` and update the `MODEL_PATH` constant in `src/main.rs`:

```rust
const MODEL_PATH: &str = "./src/models/your-model.gguf";
```

### 4. Tokenizer Authentication (Important!)

Most Llama models require Hugging Face authentication. Choose one option:

#### Option A: Environment Variable (Recommended)
```bash
# Get token from https://huggingface.co/settings/tokens
export HF_TOKEN=your_hugging_face_token_here

# Windows PowerShell:
$env:HF_TOKEN="your_hugging_face_token_here"

# Windows CMD:
set HF_TOKEN=your_hugging_face_token_here
```

#### Option B: Manual Tokenizer Download
1. Go to https://huggingface.co/huggyllama/llama-7b
2. Download `tokenizer.json` to your model directory (`./src/models/`)

#### Option C: Use Open Models
Update `MODEL_PATH` to use models that don't require authentication.

### 5. Build and Run

```bash
# CPU version
cargo run --release

# GPU versions
cargo run --release --features cuda    # NVIDIA GPUs
cargo run --release --features metal   # Apple Silicon
```

## Configuration

### Key Constants (in `src/main.rs`)

```rust
const MODEL_PATH: &str = "./src/models/llama-2-7b-chat.Q4_K_S.gguf";
const MAX_VERBS_PER_PREFIX: usize = 600;  // Max verbs per prefix (aa-zz)
const MAX_GENERATION_TOKENS: usize = 512; // Max tokens per generation
const GPU_BATCH_SIZE: usize = 8;          // GPU batch size
const CPU_BATCH_SIZE: usize = 3;          // CPU batch size
```

### Supported Model Types

The application automatically detects model types from filenames:

- **Llama**: `llama-2-7b-chat`, `llama-2-13b`, etc.
- **Mistral**: `mistral-7b-instruct`, `mistral-7b`, etc.
- **CodeLlama**: `codellama-7b`, `code-llama`, etc.
- **DeepSeek**: `deepseek-coder`, `deepseek-r1`, etc.
- **Phi-3**: `phi-3-mini`, etc.
- **Qwen**: `qwen2-7b`, etc.
- **Gemma**: `gemma-7b`, etc.

## Troubleshooting

### Tokenizer Issues

**Error**: `status code 401` or `Failed to download tokenizer`

**Solutions**:
1. Set `HF_TOKEN` environment variable (see Setup section)
2. Download tokenizer manually to model directory
3. Use an open model that doesn't require authentication

**Error**: `Tokenizer vocab size seems unusual`

**Solution**: This is usually a warning, not an error. The application will continue running.

### GPU Issues

**Error**: `CUDA not available` or `Metal not available`

**Solutions**:
1. Install appropriate GPU drivers
2. Rebuild with correct features: `--features cuda` or `--features metal`
3. Use CPU version (slower but works everywhere)

### Memory Issues

**Error**: Out of memory or allocation failures

**Solutions**:
1. Reduce `GPU_BATCH_SIZE` or `CPU_BATCH_SIZE`
2. Use a smaller model (e.g., 7B instead of 13B)
3. Close other applications to free memory

### Performance Optimization

**Slow generation** (< 5 tokens/second on CPU):

**Solutions**:
1. Use GPU acceleration: `--features cuda` or `--features metal`
2. Use quantized models (Q4_K_S, Q4_K_M)
3. Increase batch size for GPU processing

## Output

The application generates:

- **State Files**: `./verb_state/` - Progress tracking and resume capability
- **Output Files**: `./verb_output/` - Final JSON files with verb outcomes
- **Progress Display**: Real-time progress bar with ETA and performance metrics

### Output Format

Each prefix generates a JSON file with verb outcomes:

```json
[
  {
    "verb": "abandon",
    "preconditions": ["something to leave behind", "a reason to leave"],
    "physical_effects": ["object or place is left unattended", "distance increases"],
    "emotional_effects": ["possible feelings of loss or relief", "sense of letting go"],
    "environmental_effects": ["abandoned item may deteriorate", "space becomes unoccupied"]
  }
]
```

## Architecture

### Key Components

- **CandleModel**: ML model wrapper with device management
- **AppState**: Application state with async processing
- **TokenOutputStream**: Tokenizer integration with streaming
- **Progress Tracking**: Real-time progress bars and metrics

### Processing Flow

1. **Initialization**: Load model, tokenizer, and previous state
2. **Prefix Processing**: Process aa-zz prefixes sequentially
3. **Batch Generation**: Generate multiple verbs per batch
4. **Validation**: Verify verb uniqueness and validity
5. **State Persistence**: Save progress after each batch
6. **Output Generation**: Create final JSON files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
1. Check this README's troubleshooting section
2. Review the application logs for specific error messages
3. Open an issue with detailed error information and system specs 