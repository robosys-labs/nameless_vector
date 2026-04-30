# Semantic Grounding Layer

**Axiom** is a **semantic validation layer** for LLMs—not a generative model. This document explains the architecture and how to use it.

## Problem Statement

Small LLMs (1B parameters) deployed on edge devices suffer from:
- **Hallucinations** - generating physically impossible actions
- **Lack of grounding** - no connection to real-world constraints
- **Costly validation** - everything goes to large cloud models

## Solution: Semantic Grounding Layer

A deterministic constraint-checking system that:
1. **Routes queries** to appropriate tier (local / small LLM / large LLM)
2. **Validates outputs** against structured semantic frames
3. **Catches contradictions** before deployment

## Architecture

```
User Query
    ↓
[QueryRouter] → RoutingDecision
    ↓
├─ LocalHandle: Return deterministic answer
├─ SmallModel: Route to Qwen 2.5 0.5B + grounding
├─ LargeModel: Route to cloud LLM + pre-validation
└─ Reject: Physical/logical impossibility detected
    ↓
LLM Output (if routed to model)
    ↓
[GroundingLayer.validate()] → ValidationOutcome
    ↓
├─ Accept: Output is valid
├─ AcceptWithWarning: Minor issues
├─ Reject: Critical violation
└─ RequestClarification: Ambiguous
```

## Core Components

### 1. `retrieval.rs` - Semantic Embeddings
- **MiniLM-L6-v2** (22MB, 384-dim embeddings)
- Cosine similarity for frame matching
- Zero-shot query-to-frame alignment

### 2. `frame_memory.rs` - Structured Knowledge
- Loads verb/noun frames from `verb_state/*.json`
- Pre-computes embeddings for O(1) lookup
- Domain-agnostic (works for code, circuits, biology)

### 3. `grounding.rs` - Validation Engine
- Checks preconditions against current state
- Detects physical contradictions (hot + cold)
- Validates state transitions

### 4. `router.rs` - Intelligent Routing
- **LocalHandle**: High-confidence exact matches (>0.85 similarity)
- **SmallModel**: Moderate complexity → Qwen 2.5 0.5B
- **LargeModel**: Complex reasoning → Cloud LLM
- **Reject**: Constraint violations

## Quick Start

### 1. Load Frames and Initialize

```rust
use nameless_vector::{Embedder, FrameMemory, GroundingLayer, InferenceGraph};

// Initialize embedding model
let embedder = Embedder::new("sentence-transformers/all-MiniLM-L6-v2")?;

// Load semantic frames
let memory = FrameMemory::load_with_embedder("./verb_state", &embedder)?;

// Create grounding layer
let graph = InferenceGraph::new();
let grounding = GroundingLayer::new(memory.clone(), graph.clone());
```

### 2. Route Queries

```rust
use nameless_vector::{QueryRouter, RouterBuilder, StateSet};

let router = RouterBuilder::new(memory, graph)
    .local_threshold(0.80)
    .small_model_threshold(0.60)
    .build();

let context = StateSet::new(); // Current world state
let decision = router.route("I want to open the door", &context)?;

match decision {
    RoutingDecision::LocalHandle { reason } => println!("Handle locally: {}", reason),
    RoutingDecision::SmallModel { model, .. } => println!("Use {}", model),
    RoutingDecision::Reject { reason } => println!("Reject: {}", reason),
    _ => {}
}
```

### 3. Validate LLM Outputs

```rust
let llm_output = "The door is now open";
let outcome = router.validate_output(llm_output, &context)?;

match outcome {
    ValidationOutcome::Accept { .. } => println!("✅ Valid"),
    ValidationOutcome::Reject { reason, .. } => println!("🚫 Invalid: {}", reason),
    _ => {}
}
```

## Example: Running the Demo

```bash
# Run the grounding layer demonstration
cargo run --example demo_grounding_layer

# Expected output shows:
# - Query routing decisions
# - Output validation results
# - Intent extraction
```

## Integration with Small LLMs

Recommended model: **Qwen 2.5 0.5B**
- 500M parameters
- GGUF quantized (~300MB)
- Runs on Raspberry Pi 4 at ~1-2 tokens/sec
- Fully supported by Candle

### Pipeline

```rust
// 1. Route query
let decision = router.route(query, &context)?;

// 2. If SmallModel, call Qwen
let llm_output = match decision {
    RoutingDecision::SmallModel { .. } => {
        call_qwen_model(query).await?
    }
    _ => return Ok(())
};

// 3. Validate output
let outcome = router.validate_output(&llm_output, &context)?;
if matches!(outcome, ValidationOutcome::Reject { .. }) {
    // Retry or escalate
}
```

## Extending to New Domains

The grounding layer is domain-agnostic. To add new domains:

1. **Create frame files** in `verb_state/`:
```json
[
  {
    "verb": "compile",
    "applicable_subjects": ["developer", "build_system"],
    "applicable_objects": ["source_code", "project"],
    "required_subject_states": {
      "physical": [],
      "mental": ["has_toolchain"]
    },
    ...
  }
]
```

2. **Load and validate** same as verb frames

## Performance Characteristics

| Operation | Latency | Memory |
|-----------|---------|--------|
| Embedding generation | ~50ms | 22MB model |
| Frame retrieval | ~1ms | Pre-indexed |
| State validation | ~0.1ms | Deterministic |
| Full routing decision | ~60ms | Total |

## Files Created

- `src/retrieval.rs` - Embedding-based semantic retrieval
- `src/frame_memory.rs` - Frame storage with indexing
- `src/grounding.rs` - Validation layer
- `src/router.rs` - Query routing engine
- `examples/demo_grounding_layer.rs` - Usage demonstration

## Next Steps

1. **Test integration** with actual Qwen 2.5 0.5B
2. **Add domain frames** for your specific use case
3. **Deploy as sidecar** to existing LLM pipelines
4. **Tune thresholds** based on your accuracy/latency requirements

## Oxidized-GPT Status

The `Oxidized-GPT/` directory is now a **parts donor**:
- ✅ `retrieval.rs` → Ported to main crate
- ✅ `memory.rs` → Adapted to `frame_memory.rs`
- ❌ `model.rs`, `training.rs` → Archived (generative model not needed)

The custom GPT model with cross-attention is no longer required—we validate external LLM outputs instead of generating our own.
