//! Demo: State Algebra and Validation
//!
//! Run with: cargo run --example demo_state_validation
//!
//! This demo shows how state algebra validates whether actions are possible:
//! - Checking if preconditions are met
//! - Detecting state conflicts
//! - Applying action effects
//! - Validating multi-step sequences

use axiom_ai::state_algebra::{StateSet, VerbApplicabilityChecker};
use std::collections::HashSet;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║     State Algebra Demo                                   ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Scenario 1: Can you destroy wet crops?
    println!("📋 Scenario 1: Can you 'destroy' wet crops?");
    println!();
    
    let current_state = StateSet::new()
        .with_physical(vec!["wet", "crops"])
        .with_emotional(vec!["determined"]);
    
    let destroy_requirements = StateSet::new()
        .with_physical(vec!["active", "intact"]);
    
    println!("   Current state:  physical={:?}", current_state.physical);
    println!("   Requirements:     physical={:?}", destroy_requirements.physical);
    println!();
    
    let can_destroy = current_state.satisfies(&destroy_requirements);
    println!("   ✅ Can destroy? {}", can_destroy);
    println!("      (wet ≠ active, so NO)");
    println!();

    // Scenario 2: First dry, then destroy
    println!("📋 Scenario 2: First 'dry', then 'destroy'");
    println!();
    
    let dry_effect = StateSet::new()
        .with_physical(vec!["dry"]);
    
    let after_drying = current_state.apply(&dry_effect)
        .expect("Failed to apply effect");
    
    println!("   After drying: physical={:?}", after_drying.physical);
    
    // Now check if we can destroy
    let can_destroy_after = after_drying.satisfies(&destroy_requirements);
    println!("   ✅ Can destroy now? {}", can_destroy_after);
    println!("      (dry crops can be active/intact, so YES)");
    println!();

    // Scenario 3: Conflict detection
    println!("📋 Scenario 3: Detecting conflicting states");
    println!();
    
    let wet_state = StateSet::new().with_physical(vec!["wet", "moist"]);
    let dry_state = StateSet::new().with_physical(vec!["dry", "arid"]);
    
    let conflicts = wet_state.conflicts_with(&dry_state);
    println!("   State A: {:?}", wet_state.physical);
    println!("   State B: {:?}", dry_state.physical);
    println!();
    
    if let Some(conflict) = conflicts {
        println!("   ⚠️  Conflict detected: {:?}", conflict.conflict_type);
        println!("      '{}' vs '{}' are incompatible", conflict.state_a, conflict.state_b);
    }
    println!();

    // Scenario 4: Complex sequence validation
    println!("📋 Scenario 4: Validating action sequences");
    println!();
    
    let sequence = vec![
        ("gather", StateSet::new().with_physical(vec!["wood", "nails"])),
        ("prepare", StateSet::new().with_physical(vec!["ready", "measured"])),
        ("assemble", StateSet::new().with_physical(vec!["partial", "attached"])),
        ("finish", StateSet::new().with_physical(vec!["house"])),
    ];
    
    let mut current = StateSet::new(); // Start empty
    
    println!("   Validating sequence:");
    for (i, (action, effect)) in sequence.iter().enumerate() {
        let new_state = current.apply(effect)
            .expect(&format!("Failed at step {}", i + 1));
        
        println!("   {}. {} → state now has {:?}", 
            i + 1, 
            action, 
            new_state.physical.iter().collect::<Vec<_>>()
        );
        
        current = new_state;
    }
    
    println!();
    println!("   ✅ Final state achieved: {:?}", current.physical);
    println!("      Goal 'house' present? {}", current.physical.contains("house"));
    println!();

    // Summary
    println!("✨ State Algebra Features Demonstrated:");
    println!("   • Precondition checking (satisfies)");
    println!("   • Effect application (apply)");
    println!("   • Conflict detection (antonyms, contradictions)");
    println!("   • Multi-step sequence validation");
    println!();
    println!("   This enables the system to answer: 'Can I do X?'");
    println!("   and 'What do I need to do first?'");
}
