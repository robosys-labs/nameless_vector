# Axiom - Semantic Grounding Research Framework

A **research-grade semantic validation layer** that advances LLM reliability through formal constraint checking. Routes queries intelligently, validates responses against physical/logical constraints, and catches hallucinations before deployment. Built with Rust using the Candle ML framework.

## Research Focus

**Core Research Question**: Can we reduce LLM hallucinations by 50%+ through deterministic semantic validation without sacrificing generation quality?

### Core Innovation: The Semantic Firewall

**Problem**: Small LLMs (1B parameters) on edge devices hallucinate—generating physically impossible actions and logical contradictions.

**Solution**: A deterministic grounding layer that:
1. **Routes queries** to the right tier (local / small LLM / large LLM)
2. **Validates outputs** against structured semantic frames
3. **Catches contradictions** (e.g., "asleep and awake")

### Novel Contributions
- **Formal validation methodology** - State algebra as a mathematically-grounded approach to constraint checking
- **Cross-domain validation framework** - Works for code, medical, database, legal, and general domains
- **Evaluation benchmarks** - Open-source hallucination detection benchmarks
- **Failure mode taxonomy** - Systematic categorization of when/why validation fails

### Key Capabilities

| Feature | What It Does | Example |
|---------|--------------|---------|
| **Semantic Retrieval** | Embed queries, find relevant frames | Query → "abandon" frame (cosine similarity: 0.87) |
| **Query Routing** | Route to optimal processing tier | Simple query → LocalHandle; Complex → LargeModel |
| **Output Validation** | Check LLM outputs against constraints | Reject: "person is asleep and awake" |
| **State Algebra** | Validate if actions are possible | "Can `open`? Only if object is `closed`" |
| **Intent Extraction** | Map free text to structured frames | "I want to leave" → `abandon` verb frame |
| **Inference Graph** | O(1) lookups for applicable actions | "What actions work on `closed door`?" |
| **Observability** | Structured logging, metrics | Request IDs, validation pass/fail rates |
| **Security** | Rate limiting, input validation | 100 req/s limit, injection detection |

## Architecture

### The Semantic Firewall Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE                            │
├─────────────────────────────────────────────────────────────┤
│  User Query                                                 │
│      ↓                                                      │
│  ┌─────────────────┐                                        │
│  │  QueryRouter    │ ──→ RoutingDecision                     │
│  │  • Embed query  │       • LocalHandle (exact match)      │
│  │  • Classify     │       • SmallModel (Qwen 2.5 0.5B)     │
│  │  • Check constraints │    • LargeModel (cloud)           │
│  └─────────────────┘       • Reject (violation)             │
│                                      ↓                      │
├─────────────────────────────────────────────────────────────┤
│                    VALIDATION PIPELINE                       │
│  (if routed to LLM)                                         │
│      ↓                                                      │
│  LLM Output (text)                                          │
│      ↓                                                      │
│  ┌─────────────────┐                                        │
│  │ GroundingLayer  │ ──→ ValidationOutcome                   │
│  │ • Find frame    │       • Accept (valid)                 │
│  │ • Check preconds│       • Reject (contradiction)           │
│  │ • Detect conflicts     • NeedsClarification               │
│  └─────────────────┘                                        │
│      ↓                                                      │
│  User receives validated (or rejected) response             │
├─────────────────────────────────────────────────────────────┤
│                    DATA LAYER                                │
│  verb_state/*.json → Semantic frames (20,000+ verbs)       │
│  MiniLM-L6-v2 → 384-dim embeddings (22MB)                   │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start (Research Demos)

Try the out-of-the-box demos to see the grounding layer in action:

```bash
# 1. Semantic Grounding Layer (NEW) - Query routing and LLM validation
cargo run --example demo_grounding_layer

# 2. Basic graph loading and querying
cargo run --example demo_basic

# 3. State validation and preconditions
cargo run --example demo_state_validation

# 4. Goal planning and sequencing
cargo run --example demo_planning

# 5. Performance comparison (--release for accurate timing)
cargo run --example demo_performance --release
```

Each demo shows a different aspect of the grounding layer. See `examples/README.md` for research infrastructure details and `GROUNDING_LAYER.md` for architecture documentation.

---

## Quick Start (Full Setup)

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

## Usage Examples

### Building a Connected Inference Graph

```rust
use nameless_vector::edge_inference::{build_connected_graph, EdgeInferenceConfig};
use nameless_vector::inference_graph::InferenceGraph;

// Auto-generate edges from state analysis
let config = EdgeInferenceConfig::default();
let mut graph = build_connected_graph("./verb_state", Some(config))?;

// Now graph has verbs AND edges enabling multi-hop inference
println!("Graph: {} verbs, {} edges", graph.verb_count(), graph.edge_count());
```

### Planning to Achieve a Goal

```rust
use nameless_vector::temporal::{TemporalGraph, TemporalRelation};
use nameless_vector::state_algebra::StateSet;

let temporal = TemporalGraph::new(graph);
let initial = StateSet::new().with_physical(vec!["wood", "nails"]);
let goal = StateSet::new().with_physical(vec!["house"]);

// Find sequence of verbs to achieve goal
let plan = temporal.plan_to_goal(&initial, &goal, &all_verbs, 10)?;
// Result: ["gather", "prepare", "assemble", "finish"]
```

### Checking Preconditions

```rust
use nameless_vector::state_algebra::StateSet;

let current_state = StateSet::new()
    .with_physical(vec!["wet", "crops"])
    .with_emotional(vec!["determined"]);

let destroy_requirements = StateSet::new()
    .with_physical(vec!["active", "intact"]);

// Can we destroy wet crops?
let can_destroy = current_state.satisfies(&destroy_requirements);
// false → crops are wet, not active
```

### Using the Grounding Layer

```rust
use axiom_ai::{
    Embedder, FrameMemory, GroundingLayer, QueryRouter, RouterBuilder,
    RoutingDecision, StateSet, ValidationOutcome
};

// 1. Initialize embedding model (MiniLM-L6-v2, 22MB)
let embedder = Embedder::new("sentence-transformers/all-MiniLM-L6-v2")?;

// 2. Load semantic frames from verb_state
let memory = FrameMemory::load_with_embedder("./verb_state", &embedder)?;

// 3. Create query router
let graph = InferenceGraph::new();
let router = RouterBuilder::new(memory, graph)
    .local_threshold(0.80)
    .small_model_threshold(0.60)
    .build();

// 4. Route user query
let context = StateSet::new();
let decision = router.route("I want to open the door", &context)?;

match decision {
    RoutingDecision::LocalHandle { reason } => {
        // Handle deterministically without LLM
        println!("Local: {}", reason);
    }
    RoutingDecision::SmallModel { model, .. } => {
        // Route to Qwen 2.5 0.5B
        let llm_output = call_small_model(model, query).await?;
        
        // 5. Validate LLM output
        let outcome = router.validate_output(&llm_output, &context)?;
        match outcome {
            ValidationOutcome::Accept { .. } => println!("✅ Valid"),
            ValidationOutcome::Reject { reason, .. } => {
                println!("🚫 Invalid: {}", reason);
                // Retry or escalate to larger model
            }
            _ => {}
        }
    }
    RoutingDecision::Reject { reason } => {
        println!("Reject: {}", reason);
    }
    _ => {}
}
```

### Extracting Intent from Natural Language

```rust
// Extract structured intent from free-form text
if let Some(intent) = grounding.extract_intent("I need to depart from here")? {
    println!("Verb: {}", intent.verb);           // "abandon"
    println!("Confidence: {:.2}", intent.confidence);  // 0.87
    println!("Subjects: {:?}", intent.applicable_subjects);  // ["biological_body"]
}
```

## Module Reference

### Semantic Grounding Layer (NEW)

| Module | File | Purpose |
|--------|------|---------|
| **Retrieval** | `src/retrieval.rs` | MiniLM-L6 embedding engine for semantic similarity |
| **Frame Memory** | `src/frame_memory.rs` | Semantic frame storage with pre-computed embeddings |
| **Grounding** | `src/grounding.rs` | Validation layer—checks constraints, detects contradictions |
| **Router** | `src/router.rs` | Query routing—Local/SmallModel/LargeModel/Reject decisions |

### Core Inference Modules

| Module | File | Purpose |
|--------|------|---------|
| **State Algebra** | `src/state_algebra.rs` | Formal state validation, conflict detection, effect application |
| **Inference Graph** | `src/inference_graph.rs` | Petgraph-based indexing for O(1) verb lookups |
| **Edge Inference** | `src/edge_inference.rs` | Auto-generates verb relationships from state analysis |
| **Temporal** | `src/temporal.rs` | Temporal planning, causal reasoning, sequence validation |
| **Schema** | `src/schema.rs` | Version migration, backward compatibility |

### Production Infrastructure

| Module | File | Purpose |
|--------|------|---------|
| **Observability** | `src/observability.rs` | Structured logging (tracing), metrics, health checks |
| **Security** | `src/security.rs` | Rate limiting, input validation, resource quotas |

## Configuration

### Key Constants (in `src/main.rs`)

```rust
// Recommended: Qwen 2.5 0.5B for edge deployment (~300MB, runs on Raspberry Pi)
const MODEL_PATH: &str = "./src/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

const STATE_DIR: &str = "./verb_state";           // Semantic frames (20,000+ verbs)
const OUTPUT_DIR: &str = "./verb_output";         // Generated outputs
const BATCH_SIZE: usize = 5;
const MAX_VERBS_PER_PREFIX: usize = 600;
```

### Grounding Layer Configuration (in `src/router.rs`)

```rust
// Similarity thresholds for routing decisions
RouterBuilder::new(memory, graph)
    .local_threshold(0.80)        // Route to LocalHandle if similarity > 0.80
    .small_model_threshold(0.60)  // Use SmallModel if similarity > 0.60
    .build()
```

### Security Configuration (in `src/security.rs`)

```rust
// Rate limiting
const REQUESTS_PER_SECOND: u64 = 100;
const BURST_CAPACITY: u64 = 150;

// Resource quotas
const MAX_CONCURRENT: usize = 10;
const MAX_TOKENS_PER_REQUEST: usize = 800;
const MAX_GPU_MEMORY_BYTES: u64 = 4 * 1024 * 1024 * 1024; // 4GB
```

### Edge Inference Configuration (in `src/edge_inference.rs`)

```rust
EdgeInferenceConfig {
    min_confidence: 0.6,           // Minimum edge confidence
    state_overlap_threshold: 0.5,  // State matching threshold
    generate_negative_edges: true, // Also detect "disables" relationships
    max_edges_per_verb: 50,       // Prevent edge explosion
}
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

## Production Readiness

### What's Production-Grade

✅ **Concurrency Safety**: Async-safe RwLock, CAS operations for resource quotas, mutex poisoning recovery  
✅ **Resource Protection**: Token bucket rate limiting, per-IP tracking, GPU memory limits  
✅ **Observability**: Structured JSON logging, request tracing, latency histograms  
✅ **Input Sanitization**: SQL/command/XSS injection detection, regex-based validation  
✅ **Schema Evolution**: Versioned data with migration paths (1.0.0 → 1.1.0)  
✅ **Graceful Degradation**: Timeout handling, emergency state saves, error recovery  

### Bug Fixes Applied

| Issue | Fix |
|-------|-----|
| Rate limiter token leak | Reordered checks + return tokens on global reject |
| Resource quota race condition | CAS loop with `compare_exchange` |
| Mutex poisoning crashes | `unwrap_or_else` with poison recovery |
| Temporal relation gaps | Added all inverse relation variants |

## Architecture

### Semantic Grounding Pipeline
```
User Query
    ↓
[QueryRouter]
• Embed query (MiniLM-L6, 384-dim)
• Retrieve closest frame (cosine similarity)
• Classify complexity
• Check constraints
    ↓
RoutingDecision ──┬──→ LocalHandle (exact match, no LLM)
                   ├──→ SmallModel (Qwen 2.5 0.5B + grounding)
                   ├──→ LargeModel (cloud + pre-validation)
                   └──→ Reject (physical/logical violation)
    
(if SmallModel/LargeModel)
    ↓
LLM generates response
    ↓
[GroundingLayer.validate()]
• Find semantic frame for output
• Check preconditions against current state
• Detect contradictions (hot + cold)
    ↓
ValidationOutcome ──┬──→ Accept (return to user)
                     ├──→ Reject (retry or escalate)
                     └──→ AcceptWithWarning (flag issue)
```

### Key Components

| Component | Responsibility | Lines |
|-----------|---------------|-------|
| `retrieval.rs` | MiniLM embedding engine, semantic similarity | 150 |
| `frame_memory.rs` | Frame storage with pre-computed embeddings | 200 |
| `grounding.rs` | Validation layer, constraint checking | 250 |
| `router.rs` | Query routing, complexity classification | 300 |
| `state_algebra.rs` | State validation, conflict detection | 403 |
| `inference_graph.rs` | Petgraph-based indexing | 495 |
| `observability.rs` | Tracing, metrics, health checks | 406 |
| `security.rs` | Rate limiting, input validation | 509 |

## Integration with Small LLMs (Qwen 2.5 0.5B)

### Recommended Setup

**Model**: Qwen 2.5 0.5B Instruct (GGUF quantized)  
**Size**: ~300MB  
**Speed**: 1-2 tokens/sec on Raspberry Pi 4  
**Memory**: 500MB RAM at runtime

### Download

```bash
# Download Qwen 2.5 0.5B GGUF from Hugging Face
wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf

# Place in models directory
mkdir -p ./src/models
mv qwen2.5-0.5b-instruct-q4_k_m.gguf ./src/models/
```

### Pipeline Integration

```rust
// 1. Route query
let decision = router.route(query, &context)?;

// 2. Process based on routing decision
match decision {
    RoutingDecision::SmallModel { model, grounding_required } => {
        // Generate with small model
        let llm_response = qwen_generate(query, model).await?;
        
        // 3. Validate if grounding required
        if grounding_required {
            let outcome = router.validate_output(&llm_response, &context)?;
            if let ValidationOutcome::Reject { reason, .. } = outcome {
                // Retry with larger model or return error
                return Err(anyhow!("Grounding failed: {}", reason));
            }
        }
        
        Ok(llm_response)
    }
    _ => Err(anyhow!("Unsupported routing decision")),
}
```

## Documentation

| Document | Purpose |
|----------|---------|
| `README.md` | This file—overview and quick start |
| `GROUNDING_LAYER.md` | Detailed architecture and API reference |
| `examples/README.md` | Demo explanations |
| `src/goal.md` | Original design goals (archived) |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License

## Support

For issues and questions:
1. Check this README's troubleshooting section
2. Review `GROUNDING_LAYER.md` for architecture details
3. Run `cargo run --example demo_grounding_layer` to verify setup
4. Open an issue with detailed error information and system specs