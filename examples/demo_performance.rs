//! Demo: Performance Comparison
//!
//! Run with: cargo run --example demo_performance --release
//!
//! This demo compares:
//! 1. Linear scan through JSON files (old way)
//! 2. Graph-based O(1) lookups (new way)
//!
//! Shows the performance advantage of the inference cache.

use axiom_ai::edge_inference::{build_connected_graph, EdgeInferenceConfig};
use axiom_ai::state_algebra::StateSet;
use std::fs;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║     Performance Comparison Demo                          ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // First, build the graph
    println!("📚 Building inference graph...");
    let config = EdgeInferenceConfig::default();
    let graph = build_connected_graph("./verb_state", Some(config))
        .expect("Failed to build graph");
    
    let verb_count = graph.verb_count();
    let edge_count = graph.edge_count();
    println!("   Graph: {} verbs, {} edges", verb_count, edge_count);
    println!();

    // Test 1: Single lookup comparison
    println!("⚡ Test 1: Single verb lookup by state");
    println!();
    
    let target_state = StateSet::new().with_physical(vec!["active"]);
    
    // Graph lookup (O(1) via index)
    let graph_start = Instant::now();
    let graph_results = graph.find_by_required_state(&target_state);
    let graph_time = graph_start.elapsed();
    
    // Simulate linear scan (O(n) through all verbs)
    let linear_start = Instant::now();
    let linear_results = simulate_linear_scan("./verb_state", "active");
    let linear_time = linear_start.elapsed();
    
    println!("   Target: verbs requiring 'active' state");
    println!();
    println!("   Graph lookup (index):  {:?} → {} results", graph_time, graph_results.len());
    println!("   Linear scan (files):   {:?} → {} results", linear_time, linear_results.len());
    
    if linear_time.as_micros() > 0 {
        let speedup = linear_time.as_micros() as f64 / graph_time.as_micros().max(1) as f64;
        println!("   Speedup: {:.0}x faster", speedup);
    }
    println!();

    // Test 2: Multi-hop traversal
    println!("⚡ Test 2: Multi-hop relationship discovery");
    println!();
    
    if let Some(first_verb) = graph_results.first() {
        // Find connected verbs (uses graph edges)
        let multi_start = Instant::now();
        let connected = graph.find_connected(&first_verb.id, 3);
        let multi_time = multi_start.elapsed();
        
        println!("   Finding verbs 3 hops from '{}'", first_verb.verb);
        println!("   Graph traversal: {:?} → {} connected verbs", multi_time, connected.len());
        println!();
        
        // Simulating this linearly would require scanning all files for each hop
        let simulated_linear_time = linear_time * 3; // Rough estimate
        println!("   Estimated linear time: {:?}", simulated_linear_time);
        if multi_time.as_micros() > 0 {
            let speedup = simulated_linear_time.as_micros() as f64 / multi_time.as_micros() as f64;
            println!("   Estimated speedup: {:.0}x", speedup);
        }
    }
    println!();

    // Test 3: Batch queries
    println!("⚡ Test 3: Batch query performance");
    println!();
    
    let queries = vec![
        StateSet::new().with_physical(vec!["active"]),
        StateSet::new().with_physical(vec!["inactive"]),
        StateSet::new().with_physical(vec!["wet"]),
        StateSet::new().with_physical(vec!["dry"]),
        StateSet::new().with_emotional(vec!["happy"]),
    ];
    
    // Batch graph queries
    let batch_start = Instant::now();
    let mut total_results = 0;
    for query in &queries {
        let results = graph.find_by_required_state(query);
        total_results += results.len();
    }
    let batch_time = batch_start.elapsed();
    
    println!("   {} queries executed", queries.len());
    println!("   Total results: {}", total_results);
    println!("   Total time: {:?}", batch_time);
    println!("   Avg per query: {:?}", batch_time / queries.len() as u32);
    println!();

    // Summary statistics
    println!("📊 Performance Summary");
    println!();
    println!("   Graph Construction:");
    println!("      • {} verbs loaded", verb_count);
    println!("      • {} edges generated", edge_count);
    println!();
    println!("   Query Performance:");
    println!("      • O(1) state-based lookups via HashMap indices");
    println!("      • O(E) multi-hop traversal via petgraph");
    println!("      • vs O(N) linear scan through JSON files");
    println!();
    println!("   Memory vs Speed Trade-off:");
    println!("      • Graph loads all verbs into memory (~{} MB estimated)", 
        (verb_count * 500) / 1024 / 1024); // Rough estimate
    println!("      • Enables microsecond-scale queries");
    println!("      • Suitable for real-time inference");
    println!();

    println!("✨ Key Advantage: The inference cache transforms");
    println!("   20,000+ isolated verb definitions into a connected");
    println!("   knowledge graph supporting real-time reasoning.");
}

/// Simulates a linear scan through all JSON files
fn simulate_linear_scan(directory: &str, target_state: &str) -> Vec<String> {
    let mut results = Vec::new();
    
    if let Ok(entries) = fs::read_dir(directory) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "json") {
                // Simulate reading and parsing
                if let Ok(content) = fs::read_to_string(&path) {
                    // Simple string search simulation
                    if content.contains(target_state) {
                        // Extract verb names (simplified)
                        for line in content.lines() {
                            if line.contains("\"verb\"") {
                                results.push(line.to_string());
                            }
                        }
                    }
                }
            }
        }
    }
    
    results
}
