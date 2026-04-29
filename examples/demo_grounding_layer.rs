//! Demonstration of the Semantic Grounding Layer
//!
//! This example shows how to:
//! 1. Load semantic frames from verb_state files
//! 2. Initialize the embedding-based retrieval system
//! 3. Route queries to appropriate processing tiers
//! 4. Validate LLM outputs against physical/logical constraints
//!
//! Usage:
//!   cargo run --example demo_grounding_layer

use axiom_ai::retrieval::Embedder;
use axiom_ai::frame_memory::FrameMemory;
use axiom_ai::grounding::GroundingLayer;
use axiom_ai::router::{QueryRouter, RoutingDecision, QueryComplexity, ValidationOutcome};
use std::collections::HashSet;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧠 Axiom - Semantic Grounding Layer Demo");
    println!("========================================\n");

    // Step 1: Initialize embedding model for semantic retrieval
    println!("📥 Step 1: Initializing embedding model (MiniLM-L6-v2)...");
    println!("   This provides 384-dimensional embeddings for semantic similarity.");
    let embedder = Embedder::new("sentence-transformers/all-MiniLM-L6-v2")?;
    println!("✅ Embedding model ready\n");

    // Step 2: Load semantic frames from verb_state
    println!("📂 Step 2: Loading semantic frames from ./verb_state...");
    let memory = FrameMemory::load_with_embedder("./verb_state", &embedder)?;
    println!("✅ Loaded {} verb frames\n", memory.len());

    if memory.is_empty() {
        println!("⚠️  No frames loaded. Please run the verb generator first.");
        println!("   Expected: ./verb_state/*.json files");
        return Ok(());
    }

    // Step 3: Build inference graph for relationship checking
    println!("🕸️  Step 3: Building inference graph...");
    let graph = InferenceGraph::new();
    println!("✅ Inference graph ready (empty for this demo)\n");

    // Step 4: Create the grounding layer
    println!("⚓ Step 4: Initializing grounding layer...");
    let grounding = GroundingLayer::new(memory.clone(), graph.clone());
    println!("✅ Grounding layer ready\n");

    // Step 5: Create the query router
    println!("🚦 Step 5: Configuring query router...");
    let router = RouterBuilder::new(memory, graph)
        .local_threshold(0.80)
        .small_model_threshold(0.60)
        .build();
    println!("✅ Router configured");
    println!("   - Local handle threshold: 0.80");
    println!("   - Small model threshold: 0.60");
    println!("   - Default small model: qwen2.5-0.5b\n");

    // Step 6: Demonstrate query routing
    println!("📍 Step 6: Query Routing Demonstrations");
    println!("----------------------------------------");

    let test_queries = vec![
        "I want to open the door",
        "Please explain the theory of relativity in detail",
        "Can you write a creative story about a robot",
        "The person is asleep and awake at the same time",
    ];

    let context = create_sample_context();

    for query in test_queries {
        println!("\n📝 Query: \"{}\"", query);
        
        let decision = router.route(query, &context)?;
        
        match decision {
            RoutingDecision::LocalHandle { reason } => {
                println!("   → Route: 📍 LOCAL HANDLE");
                println!("   → Reason: {}", reason);
            }
            RoutingDecision::SmallModel { model, grounding_required } => {
                println!("   → Route: 🤖 SMALL MODEL ({})", model);
                println!("   → Grounding: {}", if grounding_required { "✅ Required" } else { "❌ Optional" });
            }
            RoutingDecision::LargeModel { pre_validation_required } => {
                println!("   → Route: ☁️  LARGE MODEL");
                println!("   → Pre-validation: {}", if pre_validation_required { "✅ Required" } else { "❌ Optional" });
            }
            RoutingDecision::Reject { reason } => {
                println!("   → Route: 🚫 REJECT");
                println!("   → Reason: {}", reason);
            }
        }
    }

    // Step 7: Demonstrate output validation
    println!("\n\n✅ Step 7: Output Validation Demonstrations");
    println!("--------------------------------------------");

    let test_outputs = vec![
        ("The door is now open", create_context_with_door_closed()),
        ("The person is both asleep and awake", create_simple_context()),
        ("I will leave the room after opening the window", create_simple_context()),
    ];

    for (output, ctx) in test_outputs {
        println!("\n🤖 LLM Output: \"{}\"", output);
        
        let outcome = router.validate_output(output, &ctx)?;
        
        match outcome {
            ValidationOutcome::Accept { .. } => {
                println!("   → Result: ✅ ACCEPT");
                println!("   → The output is physically and logically valid.");
            }
            ValidationOutcome::AcceptWithWarning { warning, .. } => {
                println!("   → Result: ⚠️  ACCEPT WITH WARNING");
                println!("   → Warning: {}", warning);
            }
            ValidationOutcome::AcceptWithCaution { note, .. } => {
                println!("   → Result: ❓ ACCEPT WITH CAUTION");
                println!("   → Note: {}", note);
            }
            ValidationOutcome::Reject { reason, suggestion } => {
                println!("   → Result: 🚫 REJECT");
                println!("   → Reason: {}", reason);
                println!("   → Suggestion: {}", suggestion);
            }
            ValidationOutcome::RequestClarification { issues } => {
                println!("   → Result: ❓ CLARIFICATION NEEDED");
                for issue in issues {
                    println!("   → Issue: {}", issue);
                }
            }
        }
    }

    // Step 8: Demonstrate intent extraction
    println!("\n\n🔍 Step 8: Intent Extraction Demonstrations");
    println!("------------------------------------------");

    let intent_queries = vec![
        "I need to depart from here",
        "She wants to abandon the project",
        "Please help me acquire knowledge",
    ];

    for query in intent_queries {
        println!("\n📝 Query: \"{}\"", query);
        
        if let Some(intent) = grounding.extract_intent(query)? {
            println!("   → Extracted Verb: {}", intent.verb);
            println!("   → Confidence: {:.2}", intent.confidence);
            println!("   → Subjects: {:?}", intent.applicable_subjects);
            println!("   → Objects: {:?}", intent.applicable_objects);
        } else {
            println!("   → No high-confidence intent extracted");
        }
    }

    println!("\n\n🎉 Demo complete!");
    println!("\nSummary:");
    println!("  - Semantic grounding validates LLM outputs against structured frames");
    println!("  - Query routing optimizes for latency and accuracy");
    println!("  - Physical/logical constraints catch hallucinations before deployment");
    println!("\nNext steps:");
    println!("  1. Integrate with Qwen 2.5 0.5B for on-device inference");
    println!("  2. Add domain-specific frames (code, circuits, biology)");
    println!("  3. Deploy as sidecar to existing LLM pipelines");

    Ok(())
}

/// Create a sample context for routing tests.
fn create_sample_context() -> StateSet {
    StateSet {
        physical: ["present".to_string()].iter().cloned().collect(),
        emotional: HashSet::new(),
        positional: ["indoors".to_string()].iter().cloned().collect(),
        mental: ["aware".to_string()].iter().cloned().collect(),
    }
}

/// Create a simple empty context.
fn create_simple_context() -> StateSet {
    StateSet::new()
}

/// Create context where door is closed (for validation test).
fn create_context_with_door_closed() -> StateSet {
    StateSet {
        physical: ["present".to_string(), "door_closed".to_string()].iter().cloned().collect(),
        emotional: HashSet::new(),
        positional: ["near_door".to_string()].iter().cloned().collect(),
        mental: ["aware".to_string()].iter().cloned().collect(),
    }
}
