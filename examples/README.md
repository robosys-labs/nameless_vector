# Axiom - Research Examples

Demonstrations of the semantic grounding layer for LLM validation research.

## Quick Start

```bash
# Basic graph loading and querying (research benchmark infrastructure)
cargo run --example demo_basic

# State algebra validation (core to hallucination detection)
cargo run --example demo_state_validation

# Temporal planning and goal achievement (relevant to state algebra research)
cargo run --example demo_planning

# Performance comparison (needed for Phase 4 benchmarks)
cargo run --example demo_performance --release

# Semantic grounding layer demonstration
cargo run --example demo_grounding_layer
```

## Demo Descriptions

### `demo_basic.rs` - Inference Graph Infrastructure

**What it shows:**
- Loading 20,000+ verbs from `verb_state/`
- Auto-generating edges connecting related verbs
- O(1) state-based queries
- Multi-hop relationship discovery

**Research relevance:** Provides benchmark infrastructure for cross-domain validation research.

**Sample output:**
```
📚 Loading verb data and generating connections...

✅ Graph built in 1.45s
   📊 20483 verbs loaded
   🔗 12456 edges generated

🔍 Query: What verbs apply to 'active' entities?
   Found 342 verbs (lookup took 12.4µs)
   1. deactivate
      Subjects: ["biological_body", "organization", "system"]
   2. destroy
      Subjects: ["biological_body", "device"]
   ...
```

---

### `demo_state_validation.rs` - State Algebra

**What it shows:**
- Precondition checking (can I do X now?)
- State conflict detection (wet vs dry)
- Effect application (state transitions)
- Multi-step sequence validation

**Research relevance:** Core to hallucination detection - validates LLM outputs against semantic constraints.

**Sample output:**
```
📋 Scenario 1: Can you 'destroy' wet crops?
   Current state:  physical={"wet", "crops"}
   Requirements:     physical={"active", "intact"}

   ✅ Can destroy? false
      (wet ≠ active, so NO)

📋 Scenario 2: First 'dry', then 'destroy'
   After drying: physical={"dry"}
   ✅ Can destroy now? true
      (dry crops can be active/intact, so YES)
```

---

### `demo_planning.rs` - Temporal Planning

**What it shows:**
- Goal-directed search (how do I achieve X?)
- Action sequence generation
- Precondition validation at each step
- Causal chain reasoning

**Research relevance:** Relevant to state algebra research and sequence validation metrics.

**Sample output:**
```
🎯 Goal 1: Build something from raw materials
   Initial: {"wood", "nails", "tools"}
   Goal:    {"house"}

   ✅ Plan found in 2.34ms:
      1. gather
      2. prepare
      3. assemble
      4. finish

🎯 Goal 2: Validate a predefined sequence
   Testing sequence: ["gather", "prepare", "assemble"]
   ✅ Sequence is valid
```

---

### `demo_performance.rs` - Performance Benchmarking

**What it shows:**
- O(1) graph lookups vs O(N) linear scans
- Multi-hop traversal performance
- Batch query throughput
- Memory vs speed trade-offs

**Research relevance:** Needed for Phase 4 quantization research and performance comparisons.

**Sample output:**
```
⚡ Test 1: Single verb lookup by state
   Target: verbs requiring 'active' state

   Graph lookup (index):  12.4µs → 342 results
   Linear scan (files):   1250ms → 342 results
   Speedup: 100,806x faster

⚡ Test 2: Multi-hop relationship discovery
   Finding verbs 3 hops from 'destroy'
   Graph traversal: 45.2µs → 128 connected verbs
   Estimated linear time: 3750ms
   Estimated speedup: 82,964x
```

---

## Common Usage Patterns

### Pattern 1: Find applicable verbs
```rust
let current_state = StateSet::new()
    .with_physical(vec!["wet", "crops"]);

let applicable = graph.find_by_required_state(&current_state);
// Returns all verbs that can act on wet crops
```

### Pattern 2: Plan to achieve goal
```rust
let initial = StateSet::new().with_physical(vec!["wood"]);
let goal = StateSet::new().with_physical(vec!["house"]);

let plan = temporal.plan_to_goal(&initial, &goal, &verbs, 10)?;
// Returns: ["gather", "prepare", "assemble", "finish"]
```

### Pattern 3: Check action validity
```rust
let can_destroy = current_state.satisfies(&destroy_requirements);
// Returns: true/false with conflict details
```

### Pattern 4: Build connected graph
```rust
let graph = build_connected_graph("./verb_state", None)?;
// Loads verbs + auto-generates edges from state analysis
```

---

## Integration Ideas

### Chatbot Integration
```rust
// User: "How do I build a house?"
let plan = temporal.plan_to_goal(&current_state, &goal, &verbs, 10)?;
// Response: "First gather materials, then prepare the site..."
```

### Game AI
```rust
// NPC needs to achieve goal
let applicable = graph.find_by_required_state(&npc_state);
// Select best action based on goals and constraints
```

### Workflow Automation
```rust
// Validate sequence before execution
match temporal.validate_sequence(&steps, &initial_state) {
    Ok(_) => execute_workflow(),
    Err(e) => println!("Invalid sequence: {}", e),
}
```

---

## Troubleshooting

### Demo fails to load verb_state
Make sure you're running from the project root:
```bash
cd /path/to/axiom
cargo run --example demo_basic
```

### Out of memory on large graphs
Reduce edge generation limits:
```rust
let config = EdgeInferenceConfig {
    max_edges_per_verb: 10,  // Reduce from default 50
    ..Default::default()
};
```

### Slow performance
Use `--release` flag:
```bash
cargo run --example demo_performance --release
```

---

## Next Steps

After running these demos, explore:
- Custom edge inference configurations
- Integration with your own verb data
- Extending temporal constraints
- Adding custom state validators
- Cross-domain validation (code, medical, database, legal)
- Policy DSL for compliance use cases

See the main [README.md](../README.md) for full API documentation and research methodology.
