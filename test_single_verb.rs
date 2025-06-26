use std::env;

// Simple test to see what the model generates
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing single verb generation...");
    
    // Simple test prompt
    let test_prompt = r#"Generate 1 verb starting with 'ab'. 

Output ONLY valid JSON array:
[{"verb":"abandon","preconditions":["req1","req2"],"physical_effects":["eff1","eff2"],"emotional_effects":["feel1","feel2"],"environmental_effects":["env1","env2"]}]

Generate 1 verb starting with 'ab'. NO explanations. JSON only."#;
    
    println!("📝 Test prompt:");
    println!("{}", test_prompt);
    println!("\n" + "=".repeat(50));
    println!("Run this with your model to see what it generates!");
    
    Ok(())
} 