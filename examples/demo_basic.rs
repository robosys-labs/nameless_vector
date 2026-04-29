//! Demo: Basic Inference Graph Usage
//!
//! Run with: cargo run --example demo_basic
//!
//! This demo shows how to:
//! 1. Load 20,000+ verbs from verb_state/
//! 2. Auto-generate edges connecting related verbs
//! 3. Query the graph by state conditions
//! 4. Find applicable verbs for a given situation

use axiom_ai::edge_inference::{build_connected_graph, EdgeInferenceConfig};
use axiom_ai::state_algebra::StateSet;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║     Basic Inference Cache Demo                           ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Step 1: Build the connected graph
    println!("📚 Loading verb data and generating connections...");
    let start = Instant::now();
    
    let config = EdgeInferenceConfig {
        min_confidence: 0.6,
        state_overlap_threshold: 0.5,
        generate_negative_edges: true,
        max_edges_per_verb: 30, // Limit for demo speed
    };
    
    let graph = build_connected_graph("./verb_state", Some(config))
        .expect("Failed to build graph");
    
    let load_time = start.elapsed();
    println!();
    println!("✅ Graph built in {:?}", load_time);
    println!("   📊 {} verbs loaded", graph.verb_count());
    println!("   🔗 {} edges generated", graph.edge_count());
    println!();

    // Step 2: Query by state - "What can I do to something that's 'active'?"
    println!("🔍 Query: What verbs apply to 'active' entities?");
    let query_start = Instant::now();
    
    let active_state = StateSet::new()
        .with_physical(vec!["active"]);
    
    let applicable = graph.find_by_required_state(&active_state);
    let query_time = query_start.elapsed();
    
    println!("   Found {} verbs (lookup took {:?})", applicable.len(), query_time);
    println!();
    
    // Show first 5
    for (i, verb) in applicable.iter().take(5).enumerate() {
        println!("   {}. {}", i + 1, verb.verb);
        println!("      Subjects: {:?}", verb.applicable_subjects);
        println!("      Requirements: physical={:?}", verb.required_subject_states.physical);
    }
    println!();

    // Step 3: Query by effect - "What verbs result in 'inactive'?"
    println!("🔍 Query: What verbs result in 'inactive' state?");
    let effect_start = Instant::now();
    
    let producers = graph.find_by_effect("inactive");
    let effect_time = effect_start.elapsed();
    
    println!("   Found {} verbs (lookup took {:?})", producers.len(), effect_time);
    for (i, verb) in producers.iter().take(5).enumerate() {
        println!("   {}. {}", i + 1, verb.verb);
    }
    println!();

    // Step 4: Find connected verbs (multi-hop)
    if let Some(first_verb) = applicable.first() {
        println!("🔍 Finding verbs connected to '{}'...", first_verb.verb);
        let connected = graph.find_connected(&first_verb.id, 2);
        println!("   Found {} connected verbs within 2 hops", connected.len());
        for (i, verb_id) in connected.iter().take(5).enumerate() {
            if let Some(verb) = graph.get_verb(verb_id) {
                println!("   {}. {} (via {})", i + 1, verb.verb, verb_id);
            }
        }
    }

    println!();
    println!("✨ Demo complete! The graph enables O(1) state-based lookups");
    println!("   and automatic discovery of verb relationships.");
}
