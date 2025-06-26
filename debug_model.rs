// Debug script to test model output
// Run with: cargo run --bin debug_model

use std::sync::Mutex;
use candle_core::Device;
use candle_transformers::models::quantized_llama as model;
use candle_examples::token_output_stream::TokenOutputStream;
use tokenizers::Tokenizer;
use anyhow::Result;

const MODEL_PATH: &str = "./src/models/llama-2-7b-chat.Q4_K_S.gguf";

struct SimpleModel {
    model: model::ModelWeights,
    tokenizer: TokenOutputStream,
    device: Device,
}

impl SimpleModel {
    fn new() -> Result<Self> {
        // Load model (simplified version)
        let device = Device::Cpu; // Use CPU for simplicity
        
        // Load GGUF model
        let mut file = std::fs::File::open(MODEL_PATH)?;
        let model_content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
        let model = model::ModelWeights::from_gguf(model_content, &mut file, &device)?;
        
        // Create a simple tokenizer (this is a placeholder - you'd need proper tokenizer)
        // For now, just create a dummy one to test the structure
        let tokenizer = TokenOutputStream::new(
            Tokenizer::from_bytes(include_bytes!("../tokenizer.json")).unwrap()
        );
        
        Ok(Self { model, tokenizer, device })
    }
    
    fn test_prompt(&mut self, prompt: &str) -> Result<String> {
        println!("🧪 Testing prompt:");
        println!("{}", prompt);
        println!("{}", "=".repeat(50));
        
        // This is a simplified test - you'd implement the full generation logic here
        // For now, just return what the structure would look like
        
        Ok("This would show the actual model output".to_string())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧪 Debug Model Output Test");
    
    // Test prompts
    let test_prompts = vec![
        r#"[{"verb":"abandon","preconditions":["something to leave"],"physical_effects":["item left behind"],"emotional_effects":["relief or sadness"],"environmental_effects":["space becomes empty"]}]

Generate 2 verbs starting with 'ab' in the same JSON format above:"#,
        
        r#"Generate JSON array with 2 verbs starting with 'ab':
[{"verb":"example","preconditions":["req"],"physical_effects":["eff"],"emotional_effects":["feel"],"environmental_effects":["env"]}]"#,
    ];
    
    for (i, prompt) in test_prompts.iter().enumerate() {
        println!("\n🧪 Test Prompt {}:", i + 1);
        println!("{}", prompt);
        println!("{}", "=".repeat(80));
        println!("📝 To test: Run this prompt with your model and check the output format");
        println!("{}", "=".repeat(80));
    }
    
    println!("\n💡 Tips for debugging:");
    println!("1. Check if model generates explanations before JSON");
    println!("2. Look for incomplete JSON (missing closing braces)");
    println!("3. Check if model follows the exact format requested");
    println!("4. Verify that the model understands the instruction");
    
    Ok(())
} 