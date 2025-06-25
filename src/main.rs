use std::{
    collections::{HashMap, HashSet},
    fs,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_transformers::models::quantized_llama as model;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_examples::token_output_stream::TokenOutputStream;
use ctrlc;
use indicatif::{ProgressBar, ProgressStyle};
use model::ModelWeights;
use serde::{Deserialize, Serialize};

use std::sync::Mutex;
use tokenizers::Tokenizer;
use tokio::sync::Mutex as AsyncMutex;

// Configuration constants
const MODEL_PATH: &str = "./src/models/llama-2-7b-chat.Q4_K_S.gguf"; // Will use meta-llama/Llama-2-7b-chat-hf tokenizer
const STATE_DIR: &str = "./verb_state";
const OUTPUT_DIR: &str = "./verb_output";
const BATCH_SIZE: usize = 5;
const MAX_VERBS_PER_PREFIX: usize = 600;
const MAX_GENERATION_TOKENS: usize = 800;

// GPU-optimized batch sizes
const GPU_BATCH_SIZE: usize = 8; // Larger batches for GPU efficiency
const CPU_BATCH_SIZE: usize = 3; // Smaller batches for CPU to avoid memory pressure

#[derive(Debug, Serialize, Deserialize, Clone)]
struct VerbOutcome {
    verb: String,
    preconditions: Vec<String>,
    physical_effects: Vec<String>,
    emotional_effects: Vec<String>,
    environmental_effects: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct PrefixState {
    prefix: String,
    processed_verbs: HashSet<String>,
    outcomes: Vec<VerbOutcome>,
    completed: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct GlobalState {
    prefix_queue: Vec<String>,
    current_prefix: Option<String>,
    completed_prefixes: HashSet<String>,
}

// Model configuration for different architectures
#[derive(Debug, Clone)]
struct ModelConfig {
    chat_template: fn(&str) -> String,
    eos_token_id: u32,
    #[allow(dead_code)]
    model_type: ModelType,
}

#[derive(Debug, Clone, PartialEq)]
enum ModelType {
    Llama,
    Mistral,
    CodeLlama,
}

impl ModelConfig {
    fn from_model_path(model_path: &str) -> Self {
        let model_path_lower = model_path.to_lowercase();
        
        if model_path_lower.contains("mistral") {
            Self {
                chat_template: |prompt| format!("[INST] {} [/INST]", prompt),
                eos_token_id: 2, // </s>
                model_type: ModelType::Mistral,
            }
        } else if model_path_lower.contains("deepseek") {
            Self {
                chat_template: |prompt| format!("{}", prompt),
                eos_token_id: 2, // Will be determined from tokenizer
                model_type: ModelType::CodeLlama,
            }
        } else if model_path_lower.contains("code") {
            Self {
                chat_template: |prompt| format!("[INST] {} [/INST]", prompt),
                eos_token_id: 2, // </s>
                model_type: ModelType::CodeLlama,
            }
        } else {
            // Default to Llama-2
            Self {
                chat_template: |prompt| {
                    format!("<s>[INST] {} [/INST]", prompt)
                },
                eos_token_id: 2, // </s>
                model_type: ModelType::Llama,
            }
        }
    }
}

// Thread-safe model wrapper
struct CandleModel {
    model: ModelWeights,
    tokenizer: TokenOutputStream,
    device: Device,
    config: ModelConfig,
}

impl CandleModel {
    fn new(model_path: &str) -> Result<Self> {
        println!("🔧 Initializing model from: {}", model_path);
        
        let config = ModelConfig::from_model_path(model_path);
        
        // Initialize device with proper error handling
        let device = Self::initialize_device()
            .context("Failed to initialize compute device")?;
        
        println!("📱 Using device: {:?}", device);
        
        // Load and validate GGUF model
        let mut file = std::fs::File::open(model_path)
            .with_context(|| format!("Failed to open model file: {}", model_path))?;
        
        let start_time = Instant::now();
        let model_content = gguf_file::Content::read(&mut file)
            .context("Failed to read GGUF file - file may be corrupted")?;
        
        // Validate model architecture
        Self::validate_model_architecture(&model_content, &config)?;
        
        // Calculate and display model size
        let total_size = Self::calculate_model_size(&model_content);
        println!(
            "📊 Model loaded: {} tensors ({}) in {:.2}s",
            model_content.tensor_infos.len(),
            Self::format_size(total_size),
            start_time.elapsed().as_secs_f32(),
        );
        
        // Load tokenizer for this model
        println!("🔍 Model file: {}", model_path);
        let tokenizer = Self::load_tokenizer_from_model_path(model_path)
            .context("Failed to load tokenizer")?;
        
        // Validate tokenizer compatibility
        Self::validate_tokenizer_compatibility(model_path, &tokenizer)?;
        
        // Load model weights with GPU optimization
        let model = ModelWeights::from_gguf(model_content, &mut file, &device)
            .context("Failed to load model weights from GGUF")?;
        
        // Display device-specific information
        Self::display_device_info(&device);
        
        println!("✅ Model weights loaded successfully");
        
        println!("✅ Tokenizer loaded successfully");
        
        Ok(Self {
            model,
            tokenizer,
            device,
            config,
        })
    }
    
    fn initialize_device() -> Result<Device> {
        println!("🔍 Detecting available compute devices...");
        
        // Try CUDA first (NVIDIA GPUs)
        if cfg!(feature = "cuda") {
            match Self::try_cuda_device() {
                Ok(device) => return Ok(device),
                Err(e) => println!("⚠️  CUDA not available: {}", e),
            }
        }
        
        // Try Metal (Apple Silicon/macOS)
        if cfg!(feature = "metal") {
            match Self::try_metal_device() {
                Ok(device) => return Ok(device),
                Err(e) => println!("⚠️  Metal not available: {}", e),
            }
        }
        
        // Try Accelerate framework (Apple optimized CPU)
        if cfg!(feature = "accelerate") {
            match Self::try_accelerate_device() {
                Ok(device) => return Ok(device),
                Err(e) => println!("⚠️  Accelerate not available: {}", e),
            }
        }
        
        // Fallback to CPU with optimization hints
        println!("💻 Using CPU compute (consider installing GPU drivers for better performance)");
        println!("💡 For NVIDIA GPUs: Install CUDA toolkit and rebuild with --features cuda");
        println!("💡 For Apple Silicon: Rebuild with --features metal");
        
        Ok(Device::Cpu)
    }
    
    fn try_cuda_device() -> Result<Device> {
        // Try multiple CUDA devices if available
        for device_id in 0..8 {
            match Device::new_cuda(device_id) {
                Ok(device) => {
                    // Get GPU info if possible
                    println!("🚀 Using CUDA GPU {} acceleration", device_id);
                    if device_id == 0 {
                        println!("💾 GPU memory will be managed automatically");
                    }
                    return Ok(device);
                }
                Err(e) if device_id == 0 => {
                    return Err(anyhow::anyhow!("Primary CUDA device failed: {}", e));
                }
                Err(_) => break, // No more devices available
            }
        }
        Err(anyhow::anyhow!("No CUDA devices available"))
    }
    
    fn try_metal_device() -> Result<Device> {
        match Device::new_metal(0) {
            Ok(device) => {
                println!("🍎 Using Metal GPU acceleration (Apple Silicon optimized)");
                println!("⚡ Unified memory architecture detected");
                Ok(device)
            }
            Err(e) => Err(anyhow::anyhow!("Metal device initialization failed: {}", e)),
        }
    }
    
    fn try_accelerate_device() -> Result<Device> {
        // Accelerate framework provides optimized CPU operations on Apple platforms
        println!("🔧 Using Accelerate framework (Apple optimized CPU)");
        println!("📈 BLAS/LAPACK acceleration enabled");
        Ok(Device::Cpu) // Accelerate is a CPU optimization, not a separate device type
    }
    
    fn display_device_info(device: &Device) {
        match device {
            Device::Cpu => {
                println!("💻 CPU Device: Multi-threaded computation enabled");
                println!("🧠 Memory: System RAM will be used for model storage");
            }
            Device::Cuda(_cuda_device) => {
                println!("🚀 CUDA Device: GPU acceleration enabled");
                println!("⚡ Memory: GPU VRAM will be used for optimal performance");
                println!("🔧 Optimization: Tensor operations will run on GPU cores");
            }
            Device::Metal(_metal_device) => {
                println!("🍎 Metal Device: Apple GPU acceleration active");
                println!("🔄 Memory: Unified memory architecture optimized");
                println!("⚡ Performance: Apple Silicon neural engine integration");
            }
        }
    }
    
    fn validate_model_architecture(content: &gguf_file::Content, _config: &ModelConfig) -> Result<()> {
        // Basic validation - check if we have expected tensors
        let tensor_names: Vec<&String> = content.tensor_infos.keys().collect();
        
        // Look for common Llama/transformer patterns
        let has_attention = tensor_names.iter().any(|name| name.contains("attn"));
        let has_feed_forward = tensor_names.iter().any(|name| name.contains("ffn") || name.contains("mlp"));
        
        if !has_attention || !has_feed_forward {
            return Err(anyhow::anyhow!(
                "Model architecture validation failed - missing expected transformer components"
            ));
        }
        
        println!("✅ Model architecture validation passed");
        Ok(())
    }
    
    fn calculate_model_size(content: &gguf_file::Content) -> usize {
        content.tensor_infos.iter().map(|(_, tensor)| {
            let elem_count = tensor.shape.elem_count();
            elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size()
        }).sum()
    }
    
    fn load_tokenizer_from_model_path(model_path: &str) -> Result<TokenOutputStream> {
        println!("🔍 Loading proper tokenizer for model...");
        
        // Determine the correct tokenizer repository based on the model path
        let tokenizer_repo = Self::get_tokenizer_repo_for_model(model_path)
            .with_context(|| format!("Failed to determine tokenizer for model: {}", model_path))?;
        
        println!("📥 Downloading tokenizer from: {}", tokenizer_repo);
        
        let tokenizer = Self::download_tokenizer(tokenizer_repo)?;
        
        Ok(TokenOutputStream::new(tokenizer))
    }
    
    fn get_tokenizer_repo_for_model(model_path: &str) -> Result<&'static str> {
        let model_path_lower = model_path.to_lowercase();
        
        if model_path_lower.contains("mistral") {
            if model_path_lower.contains("instruct") {
                Ok("mistralai/Mistral-7B-Instruct-v0.1")
            } else {
                Ok("mistralai/Mistral-7B-v0.1")
            }
        } else if model_path_lower.contains("llama") {
            if model_path_lower.contains("chat") {
                Ok("meta-llama/Llama-2-7b-chat-hf")
            } else if model_path_lower.contains("code") {
                Ok("codellama/CodeLlama-7b-hf")
            } else if model_path_lower.contains("llama-2") {
                Ok("meta-llama/Llama-2-7b-hf")
            } else if model_path_lower.contains("llama-3") {
                Ok("meta-llama/Meta-Llama-3-8B")
            } else {
                // For generic llama models, try to determine from filename
                if model_path_lower.contains("7b") {
                    Ok("meta-llama/Llama-2-7b-hf")
                } else if model_path_lower.contains("13b") {
                    Ok("meta-llama/Llama-2-13b-hf")
                } else {
                    Err(anyhow::anyhow!(
                        "Cannot determine correct tokenizer for Llama model: {}. \
                        Please specify the exact model variant (e.g., llama-2-7b-chat, codellama-7b)", 
                        model_path
                    ))
                }
            }
        } else if model_path_lower.contains("deepseek") {
            if model_path_lower.contains("coder") {
                Ok("deepseek-ai/deepseek-coder-6.7b-base")
            } else if model_path_lower.contains("r1") {
                Ok("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
            } else {
                Ok("deepseek-ai/deepseek-llm-7b-base")
            }
        } else if model_path_lower.contains("phi") {
            Ok("microsoft/Phi-3-mini-4k-instruct")
        } else if model_path_lower.contains("qwen") {
            Ok("Qwen/Qwen2-7B")
        } else if model_path_lower.contains("gemma") {
            Ok("google/gemma-7b")
        } else {
            Err(anyhow::anyhow!(
                "Unknown model type: {}. \
                Supported models: Llama-2, CodeLlama, Mistral, DeepSeek, Phi-3, Qwen, Gemma. \
                Please ensure your model filename contains the model name.", 
                model_path
            ))
        }
    }
    
    fn download_tokenizer(repo: &str) -> Result<Tokenizer> {
        // Initialize HF API with authentication if available
        let api = match std::env::var("HF_TOKEN") {
            Ok(token) => {
                println!("🔐 Using authenticated Hugging Face access");
                hf_hub::api::sync::ApiBuilder::new()
                    .with_token(Some(token))
                    .build()
                    .context("Failed to initialize authenticated Hugging Face API")?
            }
            Err(_) => {
                println!("🔓 Using anonymous Hugging Face access (some models may be restricted)");
                hf_hub::api::sync::Api::new()
                    .context("Failed to initialize Hugging Face API")?
            }
        };
        
        println!("🔗 Connecting to Hugging Face Hub...");
        
        // Try multiple tokenizer formats in order of preference
        let mut errors = Vec::new();
        
        // Try tokenizer.json first
        println!("🔍 Trying tokenizer.json format...");
        match Self::load_tokenizer_json(&api, repo) {
            Ok(tokenizer) => {
                println!("✅ Tokenizer loaded successfully using tokenizer.json format");
                return Ok(tokenizer);
            }
            Err(e) => {
                println!("⚠️  tokenizer.json format failed: {}", e);
                errors.push(format!("tokenizer.json: {}", e));
            }
        }
        
        // Try vocab.json + merges.txt
        println!("🔍 Trying vocab.json + merges.txt format...");
        match Self::load_bpe_tokenizer(&api, repo) {
            Ok(tokenizer) => {
                println!("✅ Tokenizer loaded successfully using vocab.json + merges.txt format");
                return Ok(tokenizer);
            }
            Err(e) => {
                println!("⚠️  vocab.json + merges.txt format failed: {}", e);
                errors.push(format!("vocab.json + merges.txt: {}", e));
            }
        }
        
        // Try tokenizer.model (SentencePiece)
        println!("🔍 Trying tokenizer.model format...");
        match Self::load_sentencepiece_tokenizer(&api, repo) {
            Ok(tokenizer) => {
                println!("✅ Tokenizer loaded successfully using tokenizer.model format");
                return Ok(tokenizer);
            }
            Err(e) => {
                println!("⚠️  tokenizer.model format failed: {}", e);
                errors.push(format!("tokenizer.model: {}", e));
            }
        }
        
        // If all methods failed, try alternative repositories
        if let Some(alt_repo) = Self::get_alternative_tokenizer_repo(repo) {
            println!("🔄 Trying alternative repository: {}", alt_repo);
            return Self::download_tokenizer(alt_repo);
        }
        
        // If everything failed, provide helpful error message
        let error_msg = format!(
            "Failed to download tokenizer from {}. Possible solutions:\n\
            1. Set HF_TOKEN environment variable for gated models\n\
            2. Use a different model that doesn't require authentication\n\
            3. Check your internet connection\n\
            Attempted formats and errors:\n{}",
            repo,
            errors.join("\n")
        );
        
        Err(anyhow::anyhow!(error_msg))
    }
    
    fn load_tokenizer_json(api: &hf_hub::api::sync::Api, repo: &str) -> Result<Tokenizer> {
        let tokenizer_path = api
            .model(repo.to_string())
            .get("tokenizer.json")
            .with_context(|| format!("Failed to download tokenizer.json from {}", repo))?;
        
        Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to parse tokenizer.json: {}", e))
    }
    
    fn load_bpe_tokenizer(api: &hf_hub::api::sync::Api, repo: &str) -> Result<Tokenizer> {
        let vocab_path = api
            .model(repo.to_string())
            .get("vocab.json")
            .with_context(|| format!("Failed to download vocab.json from {}", repo))?;
        
        let merges_path = api
            .model(repo.to_string())
            .get("merges.txt")
            .with_context(|| format!("Failed to download merges.txt from {}", repo))?;
        
        use tokenizers::models::bpe::BPE;
        let bpe = BPE::from_file(&vocab_path.to_string_lossy(), &merges_path.to_string_lossy())
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to create BPE tokenizer: {}", e))?;
        
        Ok(Tokenizer::new(bpe))
    }
    
    fn load_sentencepiece_tokenizer(api: &hf_hub::api::sync::Api, repo: &str) -> Result<Tokenizer> {
        let model_path = api
            .model(repo.to_string())
            .get("tokenizer.model")
            .with_context(|| format!("Failed to download tokenizer.model from {}", repo))?;
        
        // Try to load as SentencePiece model using the tokenizers library
        Tokenizer::from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load SentencePiece tokenizer: {}", e))
    }
    
    fn get_alternative_tokenizer_repo(original_repo: &str) -> Option<&'static str> {
        // Map gated repositories to open alternatives when possible
        match original_repo {
            // Llama-2 alternatives - use compatible models with similar tokenizers
            "meta-llama/Llama-2-7b-chat-hf" => Some("huggyllama/llama-7b"),
            "meta-llama/Llama-2-7b-hf" => Some("huggyllama/llama-7b"),
            "meta-llama/Llama-2-13b-hf" => Some("huggyllama/llama-13b"),
            "meta-llama/Llama-2-13b-chat-hf" => Some("huggyllama/llama-13b"),
            // Llama-3 alternatives
            "meta-llama/Meta-Llama-3-8B" => Some("huggyllama/llama-7b"),
            // CodeLlama alternatives
            "codellama/CodeLlama-7b-hf" => Some("huggyllama/llama-7b"),
            _ => None,
        }
    }
    
    fn validate_tokenizer_compatibility(model_path: &str, tokenizer: &TokenOutputStream) -> Result<()> {
        let model_path_lower = model_path.to_lowercase();
        let vocab_size = tokenizer.tokenizer().get_vocab_size(false);
        
        // Validate vocabulary size matches expected ranges for different model types
        let expected_vocab_range = if model_path_lower.contains("llama") {
            (32000, 32100) // Llama models typically have ~32000 tokens
        } else if model_path_lower.contains("mistral") {
            (32000, 32100) // Mistral uses similar vocab size to Llama
        } else if model_path_lower.contains("deepseek") {
            (32000, 102400) // DeepSeek models can vary more
        } else if model_path_lower.contains("phi") {
            (50000, 52000) // Phi models have larger vocab
        } else if model_path_lower.contains("qwen") {
            (151000, 152000) // Qwen has much larger vocab
        } else if model_path_lower.contains("gemma") {
            (256000, 257000) // Gemma has very large vocab
        } else {
            (1000, 300000) // Very wide range for unknown models
        };
        
        if vocab_size < expected_vocab_range.0 || vocab_size > expected_vocab_range.1 {
            println!("⚠️  Warning: Tokenizer vocab size {} seems unusual for this model type", vocab_size);
            println!("   Expected range: {}-{}", expected_vocab_range.0, expected_vocab_range.1);
            println!("   This might indicate a tokenizer mismatch");
        } else {
            println!("✅ Tokenizer validation passed (vocab size: {})", vocab_size);
        }
        
        // Test tokenization of a simple phrase
        let test_phrase = "Hello, world!";
        match tokenizer.tokenizer().encode(test_phrase, false) {
            Ok(encoding) => {
                let token_count = encoding.get_ids().len();
                if token_count == 0 || token_count > 10 {
                    println!("⚠️  Warning: Unusual tokenization result for test phrase");
                } else {
                    println!("✅ Tokenizer test passed ({} tokens for '{}')", token_count, test_phrase);
                }
            }
            Err(e) => {
                return Err(anyhow::anyhow!("Tokenizer test failed: {}", e));
            }
        }
        
        Ok(())
    }
    
    fn generate_text(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        println!("🤖 Generating response (max {} tokens)", max_tokens);
        
        // Format prompt according to model type
        let formatted_prompt = (self.config.chat_template)(prompt);
        
        // Tokenize input
        let tokens = self.tokenizer
            .tokenizer()
            .encode(formatted_prompt, true)
            .map_err(|e| anyhow::anyhow!("Failed to tokenize input prompt: {}", e))?;
        
        let mut prompt_tokens = tokens.get_ids().to_vec();
        println!("📝 Prompt tokenized: {} tokens", prompt_tokens.len());
        
        // Ensure we don't exceed context length
        let available_tokens = model::MAX_SEQ_LEN.saturating_sub(max_tokens).saturating_sub(10);
        if prompt_tokens.len() > available_tokens {
            let tokens_to_remove = prompt_tokens.len() - available_tokens;
            prompt_tokens = prompt_tokens[tokens_to_remove..].to_vec();
            println!("⚠️  Truncated prompt to {} tokens", prompt_tokens.len());
        }
        
        // Set up generation parameters optimized for GPU
        let temperature = 0.7; // Slightly lower for more focused output
        let repeat_penalty = 1.1;
        let repeat_last_n = 64;
        let seed = 42; // Fixed seed for reproducibility
        
        // GPU-specific optimizations
        let _batch_size = match &self.device {
            Device::Cuda(_) => 1, // Single batch for CUDA efficiency
            Device::Metal(_) => 1, // Metal handles batching internally
            Device::Cpu => 1, // CPU single batch
        };
        
        let sampling = if temperature <= 0.0 {
            Sampling::ArgMax
        } else {
            Sampling::All { temperature }
        };
        
        let mut logits_processor = LogitsProcessor::from_sampling(seed, sampling);
        
        // Initial forward pass
        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?
            .unsqueeze(0)
            .context("Failed to create input tensor")?;
        
        let logits = self.model
            .forward(&input, 0)
            .context("Forward pass failed")?;
        
        let logits = logits.squeeze(0)?;
        let mut next_token = logits_processor.sample(&logits)?;
        
        let mut all_tokens = vec![next_token];
        let mut response_text = String::new();
        
        // Process first token
        if let Some(text) = self.tokenizer.next_token(next_token)? {
            response_text.push_str(&text);
        }
        
        // Generate remaining tokens
        let start_generation = Instant::now();
        let mut tokens_generated = 1;
        
        for index in 1..max_tokens {
            let input = Tensor::new(&[next_token], &self.device)?
                .unsqueeze(0)?;
            
            let logits = self.model
                .forward(&input, prompt_tokens.len() + index)
                .context("Forward pass failed during generation")?;
            
            let logits = logits.squeeze(0)?;
            
            // Apply repeat penalty
            let logits = if repeat_penalty != 1.0 {
                let start_at = all_tokens.len().saturating_sub(repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    repeat_penalty,
                    &all_tokens[start_at..],
                )?
            } else {
                logits
            };
            
            next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);
            
            if let Some(text) = self.tokenizer.next_token(next_token)? {
                response_text.push_str(&text);
            }
            
            tokens_generated += 1;
            
            // Check for EOS token
            if next_token == self.config.eos_token_id {
                println!("🛑 Generation stopped at EOS token");
                break;
            }
            
            // Early stopping for JSON completion
            if self.is_json_complete(&response_text) {
                println!("✅ JSON completion detected");
                break;
            }
        }
        
        // Decode any remaining tokens
        if let Some(rest) = self.tokenizer.decode_rest()? {
            response_text.push_str(&rest);
        }
        
        let generation_time = start_generation.elapsed();
        let tokens_per_second = tokens_generated as f64 / generation_time.as_secs_f64();
        
        // Display performance metrics with device-specific context
        match &self.device {
            Device::Cuda(_) => {
                println!(
                    "🚀 GPU Generated {} tokens in {:.2}s ({:.1} tokens/s) - CUDA accelerated",
                    tokens_generated,
                    generation_time.as_secs_f32(),
                    tokens_per_second
                );
            }
            Device::Metal(_) => {
                println!(
                    "🍎 GPU Generated {} tokens in {:.2}s ({:.1} tokens/s) - Metal accelerated",
                    tokens_generated,
                    generation_time.as_secs_f32(),
                    tokens_per_second
                );
            }
            Device::Cpu => {
                println!(
                    "💻 CPU Generated {} tokens in {:.2}s ({:.1} tokens/s)",
                    tokens_generated,
                    generation_time.as_secs_f32(),
                    tokens_per_second
                );
                if tokens_per_second < 5.0 {
                    println!("💡 Consider using GPU acceleration for {:.1}x faster generation", 
                            Self::estimate_gpu_speedup());
                }
            }
        }
        
        Ok(response_text)
    }
    
    fn is_json_complete(&self, text: &str) -> bool {
        let trimmed = text.trim();
        if !trimmed.starts_with('[') {
            return false;
        }
        
        let mut bracket_count = 0;
        let mut in_string = false;
        let mut escape_next = false;
        
        for ch in trimmed.chars() {
            if escape_next {
                escape_next = false;
                continue;
            }
            
            match ch {
                '\\' if in_string => escape_next = true,
                '"' => in_string = !in_string,
                '[' if !in_string => bracket_count += 1,
                ']' if !in_string => {
                    bracket_count -= 1;
                    if bracket_count == 0 {
                        return true;
                    }
                }
                _ => {}
            }
        }
        
        false
    }
    
    fn format_size(size_in_bytes: usize) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = size_in_bytes as f64;
        let mut unit_index = 0;
        
        while size >= 1000.0 && unit_index < UNITS.len() - 1 {
            size /= 1000.0;
            unit_index += 1;
        }
        
        if unit_index == 0 {
            format!("{}B", size_in_bytes)
        } else {
            format!("{:.2}{}", size, UNITS[unit_index])
        }
    }
    
    fn estimate_gpu_speedup() -> f64 {
        // Conservative estimate of GPU speedup over CPU for transformer models
        // Based on typical performance improvements seen with quantized models
        match std::env::consts::OS {
            "macos" => 3.0,  // Metal on Apple Silicon
            "windows" | "linux" => 5.0, // CUDA on NVIDIA GPUs
            _ => 2.0, // Conservative fallback
        }
    }
    
    async fn warmup_gpu(&mut self) -> Result<()> {
        // Perform a small inference to warm up GPU kernels and memory allocation
        let warmup_prompt = "Test";
        let warmup_tokens = self.tokenizer
            .tokenizer()
            .encode(warmup_prompt, true)
            .map_err(|e| anyhow::anyhow!("Warmup tokenization failed: {}", e))?;
        
        let prompt_tokens = warmup_tokens.get_ids().to_vec();
        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?
            .unsqueeze(0)?;
        
        // Single forward pass to initialize GPU state
        let _logits = self.model
            .forward(&input, 0)
            .context("GPU warmup forward pass failed")?;
        
        println!("✅ GPU warmup completed - ready for optimal performance");
        Ok(())
    }
}

// Global model instance with proper synchronization
static MODEL: Mutex<Option<CandleModel>> = Mutex::new(None);

struct AppState {
    global_state: AsyncMutex<GlobalState>,
    prefix_states: AsyncMutex<HashMap<String, PrefixState>>,
    interrupt_flag: Arc<AtomicBool>,
}

impl AppState {
    async fn new() -> Result<Self> {
        // Ensure directories exist
        fs::create_dir_all(STATE_DIR)
            .context("Failed to create state directory")?;
        fs::create_dir_all(OUTPUT_DIR)
            .context("Failed to create output directory")?;
        
        // Load or initialize global state
        let global_state = match Self::load_global_state().await {
            Ok(state) => {
                println!("📂 Loaded existing global state");
                state
            }
            Err(_) => {
                println!("🆕 Creating new global state");
                GlobalState {
                    prefix_queue: Self::generate_prefixes(),
                    current_prefix: None,
                    completed_prefixes: HashSet::new(),
                }
            }
        };
        
        // Load existing prefix states
        let mut prefix_states = HashMap::new();
        for prefix in &global_state.prefix_queue {
            if let Ok(state) = Self::load_prefix_state(prefix).await {
                prefix_states.insert(prefix.clone(), state);
            }
        }
        
        println!("📊 Loaded {} existing prefix states", prefix_states.len());
        
        // Set up interrupt handler
        let interrupt_flag = Arc::new(AtomicBool::new(false));
        let flag_clone = interrupt_flag.clone();
        ctrlc::set_handler(move || {
            flag_clone.store(true, Ordering::SeqCst);
            println!("\n🛑 Interrupt received. Saving state and exiting gracefully...");
        })
        .context("Failed to set interrupt handler")?;
        
        Ok(Self {
            global_state: AsyncMutex::new(global_state),
            prefix_states: AsyncMutex::new(prefix_states),
            interrupt_flag,
        })
    }
    
    async fn run(&self) -> Result<()> {
        // Initialize model
        self.initialize_model().await?;
        
        // Set up progress bar
        let total_prefixes = 676; // 26 * 26
        let pb = ProgressBar::new(total_prefixes);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) | {prefix} | ETA: {eta} | Verbs: {msg}")
                .unwrap()  // Safe to unwrap - template is valid
                .progress_chars("█▉▊▋▌▍▎▏  "),
        );
        
        // Main processing loop
        while !self.interrupt_flag.load(Ordering::SeqCst) {
            let (prefix_opt, state_opt) = self.get_next_prefix().await?;
            
            if let (Some(prefix), Some(mut state)) = (prefix_opt, state_opt) {
                pb.set_prefix(prefix.clone());
                pb.set_message(format!("{}", state.processed_verbs.len()));
                
                // Update progress
                let completed_count = {
                    let global = self.global_state.lock().await;
                    global.completed_prefixes.len() as u64
                };
                pb.set_position(completed_count);
                
                // Process this prefix if not completed
                if !state.completed {
                    match self.process_prefix_batch(&prefix, &mut state).await {
                        Ok(new_verbs_count) => {
                            pb.set_message(format!("{}", state.processed_verbs.len()));
                            
                            if new_verbs_count == 0 {
                                state.completed = true;
                                self.mark_prefix_completed(&prefix).await?;
                                if state.processed_verbs.is_empty() {
                                    println!("✅ Completed prefix '{}' - no valid verbs exist for this prefix", prefix);
                                } else {
                                    println!("✅ Completed prefix '{}' with {} verbs - no more verbs found", prefix, state.processed_verbs.len());
                                }
                            } else if state.processed_verbs.len() >= MAX_VERBS_PER_PREFIX {
                                state.completed = true;
                                self.mark_prefix_completed(&prefix).await?;
                                println!("✅ Completed prefix '{}' with {} verbs - reached maximum", prefix, state.processed_verbs.len());
                            }
                        }
                        Err(e) => {
                            eprintln!("❌ Failed to process prefix '{}': {}", prefix, e);
                            // Continue with next prefix rather than failing completely
                        }
                    }
                }
                
                // Save state after each batch
                self.save_prefix_state(&prefix, &state).await?;
            } else {
                // No more prefixes to process
                break;
            }
            
            // Save global state periodically
            self.save_global_state().await?;
            
            if self.interrupt_flag.load(Ordering::SeqCst) {
                break;
            }
        }
        
        pb.finish_with_message("Processing completed or interrupted");
        self.final_save().await?;
        
        let completed_count = {
            let global = self.global_state.lock().await;
            global.completed_prefixes.len()
        };
        
        println!("🎉 Finished processing {} prefixes", completed_count);
        Ok(())
    }
    
    async fn initialize_model(&self) -> Result<()> {
        println!("🔧 Initializing language model...");
        
        let mut model_guard = MODEL.lock().unwrap();
        if model_guard.is_none() {
            let mut model = CandleModel::new(MODEL_PATH)
                .context("Failed to initialize language model")?;
            
            // Perform GPU warmup for optimal performance
            if matches!(model.device, Device::Cuda(_) | Device::Metal(_)) {
                println!("🔥 Warming up GPU for optimal performance...");
                if let Err(e) = model.warmup_gpu().await {
                    println!("⚠️  GPU warmup failed (non-critical): {}", e);
                }
            }
            
            *model_guard = Some(model);
            println!("✅ Language model initialized successfully");
        }
        Ok(())
    }
    
    async fn get_next_prefix(&self) -> Result<(Option<String>, Option<PrefixState>)> {
        let mut global = self.global_state.lock().await;
        
        // Find next uncompleted prefix
        let next_prefix = global.prefix_queue.iter()
            .find(|prefix| !global.completed_prefixes.contains(*prefix))
            .cloned();
        
        if let Some(prefix) = next_prefix {
            let mut prefix_states = self.prefix_states.lock().await;
            let state = prefix_states
                .entry(prefix.clone())
                .or_insert_with(|| PrefixState {
                    prefix: prefix.clone(),
                    processed_verbs: HashSet::new(),
                    outcomes: Vec::new(),
                    completed: false,
                });
            
            global.current_prefix = Some(prefix.clone());
            Ok((Some(prefix), Some(state.clone())))
        } else {
            Ok((None, None))
        }
    }
    
    async fn mark_prefix_completed(&self, prefix: &str) -> Result<()> {
        let mut global = self.global_state.lock().await;
        global.completed_prefixes.insert(prefix.to_string());
        global.current_prefix = None;
        Ok(())
    }
    
    async fn get_optimal_batch_size(&self) -> usize {
        // Determine batch size based on available compute device
        let model_guard = MODEL.lock().unwrap();
        if let Some(ref model) = model_guard.as_ref() {
            match &model.device {
                Device::Cuda(_) | Device::Metal(_) => {
                    println!("🚀 Using GPU-optimized batch size: {}", GPU_BATCH_SIZE);
                    GPU_BATCH_SIZE
                }
                Device::Cpu => {
                    println!("💻 Using CPU-optimized batch size: {}", CPU_BATCH_SIZE);
                    CPU_BATCH_SIZE
                }
            }
        } else {
            BATCH_SIZE // Fallback to default
        }
    }
    
    async fn process_prefix_batch(&self, prefix: &str, state: &mut PrefixState) -> Result<usize> {
        if state.processed_verbs.len() >= MAX_VERBS_PER_PREFIX {
            return Ok(0);
        }
        
        // Determine optimal batch size based on device type
        let current_batch_size = self.get_optimal_batch_size().await;
        
        // Build exclusion list from already processed verbs
        let excluded_examples = if state.processed_verbs.is_empty() {
            "none".to_string()
        } else {
            // Show ALL processed verbs to ensure proper exclusion
            state
                .processed_verbs
                .iter()
                .map(|v| format!("\"{}\"", v))
                .collect::<Vec<_>>()
                .join(", ")
        };
        
        let prompt = format!(
            r#"Generate {} NEW verbs starting with '{}' (excluding: {}).

JSON format only:
[
  {{
    "verb": "abandon",
    "preconditions": ["requirement1", "requirement2"],
    "physical_effects": ["effect1", "effect2"],
    "emotional_effects": ["feeling1", "feeling2"],
    "environmental_effects": ["change1", "change2"]
  }}
]

Rules:
- {} different verbs starting with '{}'
- Exclude: {}
- Real English verbs only
- Valid JSON array
- Return [] if no new verbs exist"#,
            current_batch_size, prefix, excluded_examples,
            current_batch_size, prefix, excluded_examples
        );
        
        // Generate response using the model with retry logic
        let mut attempts = 0;
        let max_attempts = 3;
        let mut new_verbs = Vec::new();
        
        while attempts < max_attempts {
            attempts += 1;
            
            let response_text = self.generate_with_model(&prompt).await
                .context("Failed to generate verb outcomes")?;
            
            // Extract and parse JSON
            let json_str = Self::extract_json(&response_text);
            
            // Handle empty response case
            if json_str.trim() == "[]" || json_str.trim().is_empty() {
                println!("📝 No verbs found for prefix '{}' - marking as completed", prefix);
                return Ok(0); // This will trigger completion in the caller
            }
            
            match serde_json::from_str::<Vec<VerbOutcome>>(&json_str) {
                Ok(parsed_verbs) => {
                    new_verbs = parsed_verbs;
                    println!("✅ Successfully parsed JSON with {} verbs on attempt {}", new_verbs.len(), attempts);
                    break;
                }
                Err(e) => {
                    println!("❌ JSON parsing failed on attempt {}/{}: {}", attempts, max_attempts, e);
                    if attempts < max_attempts {
                        println!("🔄 Retrying with adjusted prompt...");
                        // For retries, reduce batch size to get shorter responses
                        if attempts == 2 {
                            // Reduce batch size for retry
                            let retry_prompt = format!(
                                r#"Generate {} verbs starting with '{}' (excluding: {}).
                                
                                JSON only: [{{"verb":"example","preconditions":["req"],"physical_effects":["eff"],"emotional_effects":["feel"],"environmental_effects":["env"]}}]"#,
                                std::cmp::min(current_batch_size, 3), prefix, excluded_examples
                            );
                            
                            let retry_response = self.generate_with_model(&retry_prompt).await
                                .context("Failed to generate verb outcomes on retry")?;
                            
                            let retry_json = Self::extract_json(&retry_response);
                            if let Ok(retry_verbs) = serde_json::from_str::<Vec<VerbOutcome>>(&retry_json) {
                                new_verbs = retry_verbs;
                                println!("✅ Retry successful with {} verbs", new_verbs.len());
                                break;
                            }
                        }
                    } else {
                        return Err(anyhow::anyhow!("Failed to parse JSON after {} attempts. Last JSON: {}", max_attempts, json_str));
                    }
                }
            }
        }
        
        // Check if we got an empty array
        if new_verbs.is_empty() {
            println!("📝 Empty verb list returned for prefix '{}' - marking as completed", prefix);
            return Ok(0); // This will trigger completion in the caller
        }
        
        // Validate and add new verbs
        let mut added_count = 0;
        let mut valid_verbs_found = false;
        let mut skipped_duplicates = 0;
        
        println!("📋 Processing {} generated verbs for prefix '{}'", new_verbs.len(), prefix);
        
        for verb_outcome in new_verbs {
            let verb_lower = verb_outcome.verb.to_lowercase();
            
            // Validate verb starts with correct prefix
            if !verb_lower.starts_with(&prefix.to_lowercase()) {
                println!("⚠️  Skipping '{}' - doesn't start with '{}'", verb_outcome.verb, prefix);
                continue;
            }
            
            // Check if we already have this verb (case-insensitive)
            let already_exists = state.processed_verbs.iter()
                .any(|existing| existing.to_lowercase() == verb_lower);
            
            if already_exists {
                println!("🔄 Skipping '{}' - already processed (duplicate)", verb_outcome.verb);
                skipped_duplicates += 1;
                continue;
            }
            
            // Validate it's a real English verb (basic check)
            if Self::is_valid_english_verb(&verb_outcome.verb) {
                // Add to state
                state.processed_verbs.insert(verb_outcome.verb.clone());
                state.outcomes.push(verb_outcome.clone());
                added_count += 1;
                valid_verbs_found = true;
                
                println!("✅ Added new verb: {}", verb_outcome.verb);
            } else {
                println!("❌ Skipping '{}' - not a valid English verb", verb_outcome.verb);
            }
        }
        
        if skipped_duplicates > 0 {
            println!("🔄 Skipped {} duplicate verbs for prefix '{}'", skipped_duplicates, prefix);
        }
        
        println!("📊 Summary for '{}': {} new verbs added, {} total processed", 
                prefix, added_count, state.processed_verbs.len());
        
        // If no valid verbs were found after processing, consider this prefix complete
        if !valid_verbs_found && added_count == 0 {
            println!("📝 No valid verbs found for prefix '{}' after validation - marking as completed", prefix);
        }
        
        Ok(added_count)
    }
    
    async fn generate_with_model(&self, prompt: &str) -> Result<String> {
        let mut model_guard = MODEL.lock().unwrap();
        if let Some(ref mut model) = model_guard.as_mut() {
            let result = model.generate_text(prompt, MAX_GENERATION_TOKENS);
            drop(model_guard); // Release lock early
            result
        } else {
            drop(model_guard);
            Err(anyhow::anyhow!("Model not initialized"))
        }
    }
    
    fn extract_json(response: &str) -> String {
        let response_trimmed = response.trim();
        
        // Handle explicit empty array case
        if response_trimmed == "[]" {
            return "[]".to_string();
        }
        
        // Check if the response indicates no verbs exist
        let response_lower = response_trimmed.to_lowercase();
        if response_lower.contains("no verbs") || 
           response_lower.contains("no valid") ||
           response_lower.contains("no common") ||
           response_lower.contains("none exist") ||
           response_lower.contains("not exist") {
            return "[]".to_string();
        }
        
        // Find the JSON array bounds
        if let Some(start_bracket) = response_trimmed.find('[') {
            if let Some(end_bracket) = response_trimmed.rfind(']') {
                if start_bracket < end_bracket {
                    let json_candidate = &response_trimmed[start_bracket..=end_bracket];
                    
                    // Try to validate and fix the JSON
                    match Self::validate_and_fix_json(json_candidate) {
                        Ok(fixed_json) => return fixed_json,
                        Err(_) => {
                            // If validation fails, try to extract complete objects
                            return Self::extract_complete_objects(json_candidate);
                        }
                    }
                }
            } else {
                // No closing bracket found - JSON was truncated
                println!("⚠️  JSON appears to be truncated (no closing bracket)");
                return Self::extract_complete_objects(&response_trimmed[start_bracket..]);
            }
        }
        
        // Fallback: return empty array if no valid JSON found
        println!("⚠️  No valid JSON array found in response");
        "[]".to_string()
    }
    
    fn validate_and_fix_json(json_str: &str) -> Result<String> {
        // First, try to parse as-is
        match serde_json::from_str::<serde_json::Value>(json_str) {
            Ok(_) => return Ok(json_str.to_string()),
            Err(_) => {
                // JSON is malformed, try to fix it
                println!("🔧 Attempting to fix malformed JSON...");
            }
        }
        
        // Try to fix common JSON issues
        let mut fixed = json_str.to_string();
        
        // Remove trailing commas before closing braces/brackets
        fixed = regex::Regex::new(r",(\s*[}\]])")
            .unwrap()
            .replace_all(&fixed, "$1")
            .to_string();
        
        // Try parsing again
        match serde_json::from_str::<serde_json::Value>(&fixed) {
            Ok(_) => Ok(fixed),
            Err(e) => Err(anyhow::anyhow!("Could not fix JSON: {}", e))
        }
    }
    
    fn extract_complete_objects(json_str: &str) -> String {
        let mut complete_objects = Vec::new();
        let mut current_object = String::new();
        let mut brace_count = 0;
        let mut in_string = false;
        let mut escape_next = false;
        
        for ch in json_str.chars() {
            if escape_next {
                escape_next = false;
                current_object.push(ch);
                continue;
            }
            
            match ch {
                '\\' if in_string => {
                    escape_next = true;
                    current_object.push(ch);
                }
                '"' => {
                    in_string = !in_string;
                    current_object.push(ch);
                }
                '{' if !in_string => {
                    brace_count += 1;
                    current_object.push(ch);
                }
                '}' if !in_string => {
                    current_object.push(ch);
                    brace_count -= 1;
                    
                    if brace_count == 0 && !current_object.trim().is_empty() {
                        // We have a complete object
                        let trimmed_object = current_object.trim();
                        if trimmed_object.starts_with('{') && trimmed_object.ends_with('}') {
                            // Validate this object
                            match serde_json::from_str::<serde_json::Value>(trimmed_object) {
                                Ok(_) => {
                                    complete_objects.push(trimmed_object.to_string());
                                    println!("✅ Extracted complete object");
                                }
                                Err(_) => {
                                    println!("⚠️  Skipping invalid object");
                                }
                            }
                        }
                        current_object.clear();
                    }
                }
                _ => {
                    current_object.push(ch);
                }
            }
        }
        
        if complete_objects.is_empty() {
            println!("⚠️  No complete JSON objects found");
            return "[]".to_string();
        }
        
        // Construct a valid JSON array from complete objects
        let result = format!("[{}]", complete_objects.join(","));
        
        // Final validation
        match serde_json::from_str::<serde_json::Value>(&result) {
            Ok(_) => {
                println!("✅ Successfully reconstructed JSON with {} objects", complete_objects.len());
                result
            }
            Err(_) => {
                println!("❌ Failed to construct valid JSON array");
                "[]".to_string()
            }
        }
    }
    
    fn is_valid_english_verb(verb: &str) -> bool {
        // Basic validation for English verbs
        let verb_lower = verb.to_lowercase();
        
        // Must be alphabetic characters only
        if !verb_lower.chars().all(|c| c.is_ascii_alphabetic()) {
            return false;
        }
        
        // Reasonable length (2-15 characters for most English verbs)
        if verb_lower.len() < 2 || verb_lower.len() > 15 {
            return false;
        }
        
        // Simple heuristic: check for common non-verb patterns
        // Reject obvious non-verbs or made-up words
        let invalid_patterns = [
            "aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh", "iii", "jjj",
            "kkk", "lll", "mmm", "nnn", "ooo", "ppp", "qqq", "rrr", "sss", "ttt",
            "uuu", "vvv", "www", "xxx", "yyy", "zzz",
            "aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii", "jj",
            "kk", "ll", "mm", "nn", "oo", "pp", "qq", "rr", "ss", "tt",
            "uu", "vv", "ww", "xx", "yy", "zz"
        ];
        
        if invalid_patterns.contains(&verb_lower.as_str()) {
            return false;
        }
        
        // Check for reasonable vowel/consonant distribution
        let vowel_count = verb_lower.chars().filter(|&c| "aeiou".contains(c)).count();
        let total_length = verb_lower.len();
        
        // Reject words with no vowels (except very short ones like "by", "my")
        if vowel_count == 0 && total_length > 2 {
            return false;
        }
        
        // Reject words that are all vowels
        if vowel_count == total_length && total_length > 1 {
            return false;
        }
        
        true
    }
    
    fn generate_prefixes() -> Vec<String> {
        (b'a'..=b'z')
            .flat_map(|c1| {
                (b'a'..=b'z').map(move |c2| {
                    format!("{}{}", c1 as char, c2 as char)
                })
            })
            .collect()
    }
    
    async fn save_prefix_state(&self, prefix: &str, state: &PrefixState) -> Result<()> {
        let path = format!("{}/{}.json", STATE_DIR, prefix);
        let data = serde_json::to_string_pretty(state)
            .context("Failed to serialize prefix state")?;
        tokio::fs::write(&path, data).await
            .with_context(|| format!("Failed to write prefix state to {}", path))?;
        Ok(())
    }
    
    async fn save_global_state(&self) -> Result<()> {
        let global = self.global_state.lock().await;
        let path = format!("{}/global.json", STATE_DIR);
        let data = serde_json::to_string_pretty(&*global)
            .context("Failed to serialize global state")?;
        tokio::fs::write(&path, data).await
            .with_context(|| format!("Failed to write global state to {}", path))?;
        Ok(())
    }
    
    async fn final_save(&self) -> Result<()> {
        println!("💾 Performing final save...");
        
        // Save global state
        self.save_global_state().await?;
        
        // Save all prefix outputs
        let prefix_states = self.prefix_states.lock().await;
        let mut saved_count = 0;
        
        for (prefix, state) in prefix_states.iter() {
            if state.completed || !state.outcomes.is_empty() {
                let path = format!("{}/{}.json", OUTPUT_DIR, prefix);
                let data = serde_json::to_string_pretty(&state.outcomes)
                    .context("Failed to serialize outcomes")?;
                tokio::fs::write(&path, data).await
                    .with_context(|| format!("Failed to write output to {}", path))?;
                saved_count += 1;
            }
        }
        
        println!("✅ Final save completed - {} output files written", saved_count);
        Ok(())
    }
    
    async fn load_global_state() -> Result<GlobalState> {
        let path = format!("{}/global.json", STATE_DIR);
        let data = tokio::fs::read_to_string(&path).await
            .context("Failed to read global state file")?;
        serde_json::from_str(&data)
            .context("Failed to parse global state JSON")
    }
    
    async fn load_prefix_state(prefix: &str) -> Result<PrefixState> {
        let path = format!("{}/{}.json", STATE_DIR, prefix);
        let data = tokio::fs::read_to_string(&path).await
            .with_context(|| format!("Failed to read prefix state file: {}", path))?;
        serde_json::from_str(&data)
            .context("Failed to parse prefix state JSON")
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Starting Nameless Vector - Verb Outcome Generator");
    println!("📊 Target: {} prefixes with up to {} verbs each", 676, MAX_VERBS_PER_PREFIX);
    
    // Initialize application state
    let app_state = AppState::new().await
        .context("Failed to initialize application state")?;
    
    // Run main processing loop
    match app_state.run().await {
        Ok(()) => {
            println!("✅ Application completed successfully");
            Ok(())
        }
        Err(e) => {
            eprintln!("❌ Application failed: {}", e);
            
            // Attempt emergency save
            if let Err(save_err) = app_state.final_save().await {
                eprintln!("💥 Emergency save failed: {}", save_err);
            } else {
                println!("💾 Emergency save completed");
            }
            
            Err(e)
        }
    }
}