# Nameless Vector - Verb Outcome Generator

A high-performance Rust application that generates comprehensive verb outcomes using large language models. This tool systematically processes English verbs starting with all two-letter combinations (aa-zz) and generates detailed preconditions and effects for each verb.

## Features

- **Systematic Processing**: Generates verbs for all 676 two-letter prefixes (aa through zz)
- **Comprehensive Analysis**: For each verb, generates:
  - Preconditions (what must exist before the action)
  - Physical effects (tangible changes in the world)
  - Emotional effects (psychological impacts)
  - Environmental effects (changes to surroundings)
- **Proper Tokenization**: Automatically downloads the correct tokenizer for your model
- **State Management**: Robust state persistence with automatic recovery
- **Progress Tracking**: Real-time progress bar with detailed statistics
- **Interrupt Handling**: Graceful shutdown with state preservation
- **Batch Processing**: Efficient batch generation to optimize model usage

## Requirements

- Rust 1.70+ with Cargo
- A compatible GGUF model file (Mistral, Llama, or CodeLlama)
- At least 8GB RAM (16GB+ recommended for larger models)
- Optional: CUDA-compatible GPU for acceleration

## Setup

1. **Clone and navigate to the project:**
   ```bash
   git clone <your-repo>
   cd nameless_vector
   ```

2. **Place your model file:**
   - Download a compatible GGUF model (e.g., Mistral-7B-Instruct)
   - You can use direct download URLs like: `https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf`
   - Place it in `src/models/`
   - Update `MODEL_PATH` in `src/main.rs` if using a different model

3. **Build the application:**
   ```bash
   # CPU-only build (default)
   cargo build --release
   
   # Or with CUDA support (requires NVCC and Visual Studio Build Tools)
   cargo build --release --features cuda
   ```

## Usage

Run the application:
```bash
cargo run --release
```

The application will:
1. Initialize the language model
2. Load or create processing state
3. Begin systematic verb generation
4. Save progress continuously
5. Handle interrupts gracefully (Ctrl+C)

## Output Structure

```
verb_state/          # Processing state files
├── global.json      # Overall progress tracking
├── aa.json         # State for 'aa' prefix
├── ab.json         # State for 'ab' prefix
└── ...

verb_output/         # Final output files
├── aa.json         # Completed verbs for 'aa' prefix
├── ab.json         # Completed verbs for 'ab' prefix
└── ...
```

Each output file contains an array of verb outcomes:
```json
[
  {
    "verb": "abandon",
    "preconditions": ["something to leave behind", "a reason to leave"],
    "physical_effects": ["object or place is left unattended", "distance increases from abandoned item"],
    "emotional_effects": ["possible feelings of loss or relief", "sense of letting go"],
    "environmental_effects": ["abandoned item may deteriorate", "space becomes unoccupied"]
  }
]
```

## Configuration

Key constants in `src/main.rs`:
- `MODEL_PATH`: Path to your GGUF model file
- `BATCH_SIZE`: Number of verbs to generate per batch (default: 5)
- `MAX_VERBS_PER_PREFIX`: Maximum verbs per two-letter prefix (default: 200)
- `MAX_GENERATION_TOKENS`: Maximum tokens per model response (default: 512)

## Performance Tips

- **GPU Acceleration**: Use `--features cuda` for significant speedup
- **Memory**: Ensure sufficient RAM for your model size
- **Batch Size**: Adjust `BATCH_SIZE` based on your hardware capabilities
- **Model Selection**: Smaller quantized models (Q4_K_M) offer good performance/quality balance

## Monitoring Progress

The application provides real-time feedback:
- Progress bar showing completion percentage
- Current prefix being processed
- Number of verbs generated per prefix
- Estimated time remaining
- Token generation speed

## Error Handling

The application includes robust error handling:
- Automatic state recovery on restart
- Graceful handling of model errors
- JSON validation and cleanup
- Network timeout handling for model downloads

## Troubleshooting

**Model Loading Issues:**
- Verify model file path and format
- Ensure sufficient disk space and RAM
- Check model compatibility (GGUF format required)

**Tokenizer Download Issues:**
- Ensure internet connectivity for first run
- Check if Hugging Face Hub is accessible
- The correct tokenizer is automatically selected based on model name

**CUDA Compilation Errors:**
- Install Visual Studio Build Tools
- Ensure NVCC is in PATH
- Use CPU-only build as fallback

**Memory Issues:**
- Reduce `BATCH_SIZE` or `MAX_GENERATION_TOKENS`
- Use a smaller quantized model
- Ensure adequate system RAM

## Architecture

The application uses:
- **Candle**: High-performance ML inference framework
- **Tokio**: Async runtime for concurrent operations
- **GGUF**: Efficient model format for quantized models
- **State Persistence**: JSON-based state management
- **Progress Tracking**: Real-time monitoring with Indicatif

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here] 