//! Demo: Temporal Planning and Goal Achievement
//!
//! Run with: cargo run --example demo_planning
//!
//! This demo shows how the temporal planner finds action sequences
//! to achieve goals, validates preconditions at each step, and
//! handles causal reasoning.

use axiom_ai::edge_inference::{build_connected_graph, EdgeInferenceConfig};
use axiom_ai::temporal::TemporalGraph;
use axiom_ai::state_algebra::StateSet;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║     Temporal Planning Demo                               ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Build the graph
    println!("📚 Loading inference graph...");
    let config = EdgeInferenceConfig::default();
    let inference_graph = build_connected_graph("./verb_state", Some(config))
        .expect("Failed to build graph");
    
    println!("   Loaded {} verbs, {} edges", 
        inference_graph.verb_count(),
        inference_graph.edge_count()
    );
    println!();

    // Create temporal layer
    let temporal = TemporalGraph::new(inference_graph);

    // Demo 1: Simple goal planning
    println!("🎯 Goal 1: Build something from raw materials");
    println!();
    
    let initial = StateSet::new()
        .with_physical(vec!["wood", "nails", "tools"]);
    
    let goal = StateSet::new()
        .with_physical(vec!["house"]);
    
    println!("   Initial: {:?}", initial.physical);
    println!("   Goal:    {:?}", goal.physical);
    println!();

    // Find plan (simplified - in real usage, you'd pass available verbs)
    let all_verbs: Vec<String> = (0..100)
        .map(|i| format!("verb_{}", i))
        .collect();
    
    let plan_start = Instant::now();
    match temporal.plan_to_goal(&initial, &goal, &all_verbs, 5) {
        Ok(plan) => {
            let plan_time = plan_start.elapsed();
            println!("   ✅ Plan found in {:?}:", plan_time);
            for (i, step) in plan.iter().enumerate() {
                println!("      {}. {}", i + 1, step);
            }
        }
        Err(e) => {
            println!("   ⚠️  {}", e);
            println!("      (This is expected - the demo uses placeholder verb IDs)");
        }
    }
    println!();

    // Demo 2: State progression
    println!("🎯 Goal 2: Progress through state changes");
    println!();
    
    let states = vec![
        ("empty", "fill"),
        ("wet", "dry"),
        ("cold", "warm"),
        ("closed", "open"),
    ];
    
    for (start, end) in states {
        let initial = StateSet::new().with_physical(vec![start]);
        let goal = StateSet::new().with_physical(vec![end]);
        
        println!("   {} → {} progression", start, end);
        
        // Check if temporal relation exists
        // In a real scenario, we'd query for verbs that cause this transition
        println!("      Would search for verbs that produce '{}' state", end);
    }
    println!();

    // Demo 3: Validate a specific sequence
    println!("🎯 Goal 3: Validate a predefined sequence");
    println!();
    
    let test_sequence = vec![
        "gather".to_string(),
        "prepare".to_string(),
        "assemble".to_string(),
    ];
    
    let initial_for_validation = StateSet::new()
        .with_physical(vec!["materials"])
        .with_mental(vec!["skilled"]);
    
    println!("   Testing sequence: {:?}", test_sequence);
    println!("   Initial state: {:?}", initial_for_validation.physical);
    
    match temporal.validate_sequence(&test_sequence, &initial_for_validation) {
        Ok(_) => println!("   ✅ Sequence is valid"),
        Err(e) => println!("   ⚠️  Validation failed: {}", e),
    }
    println!();

    // Demo 4: Causal reasoning
    println!("🎯 Goal 4: Causal reasoning");
    println!();
    
    println!("   Query: What could cause 'destroyed' state?");
    println!("   Would search for verbs with final_object_states containing 'destroyed'");
    println!("   Possible causes: destroy, demolish, wreck, ruin, etc.");
    println!();
    
    println!("   Query: What are common prerequisites for 'destroyed'?");
    println!("   Would find verbs that enable destruction (e.g., 'expose' → 'destroy')");
    println!();

    // Summary
    println!("✨ Temporal Planning Features:");
    println!("   • Goal-directed search (BFS with state validation)");
    println!("   • Precondition checking at each step");
    println!("   • Sequence validation");
    println!("   • Causal chain detection");
    println!();
    println!("   This enables the system to answer: 'How do I achieve X?'");
    println!("   by finding valid action sequences through the state space.");
}
