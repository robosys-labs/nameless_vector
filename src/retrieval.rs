//! Semantic Retrieval Module
//!
//! Provides embedding-based semantic search over verb frames.
//! Ported from Oxidized-GPT, adapted for nameless_vector grounding layer.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use tokenizers::Tokenizer;

/// Embedding engine for semantic text similarity.
/// Uses MiniLM-L6-v2 (distilled BERT) for efficient 384-dim embeddings.
pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Embedder {
    /// Load embedding model from Hugging Face Hub.
    /// 
    /// Recommended model: "sentence-transformers/all-MiniLM-L6-v2"
    /// - 22MB model size
    /// - 384-dimensional embeddings
    /// - CPU-optimized for edge deployment
    pub fn new(model_id: &str) -> Result<Self> {
        let device = Device::Cpu;

        tracing::info!("Loading embedding model: {}", model_id);

        let api = hf_hub::api::sync::Api::new()
            .context("Failed to initialize Hugging Face API")?;

        let repo = api.model(model_id.to_string());

        let config_filename = repo
            .get("config.json")
            .context("Failed to download config.json")?;
        let tokenizer_filename = repo
            .get("tokenizer.json")
            .context("Failed to download tokenizer.json")?;
        let weights_filename = repo
            .get("model.safetensors")
            .context("Failed to download model.safetensors")?;

        let config = std::fs::read_to_string(&config_filename)
            .with_context(|| format!("Failed to read config from {:?}", config_filename))?;
        let config: Config = serde_json::from_str(&config)
            .context("Failed to parse config.json")?;

        let tokenizer = Tokenizer::from_file(&tokenizer_filename)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {:?}: {}", tokenizer_filename, e))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)
                .context("Failed to load model weights")?
        };
        let model = BertModel::load(vb, &config)
            .context("Failed to initialize BERT model")?;

        tracing::info!("Embedding model loaded successfully");

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Generate normalized embedding vector for text.
    /// 
    /// Returns 384-dimensional tensor suitable for cosine similarity.
    /// Embeddings are L2-normalized for efficient similarity computation.
    pub fn embed(&self, text: &str) -> Result<Tensor> {
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Failed to tokenize: {}", e))?;

        let token_ids = Tensor::new(tokens.get_ids(), &self.device)
            .context("Failed to create token_ids tensor")?
            .unsqueeze(0)
            .context("Failed to unsqueeze token_ids")?;
        let token_type_ids = Tensor::new(tokens.get_type_ids(), &self.device)
            .context("Failed to create token_type_ids tensor")?
            .unsqueeze(0)
            .context("Failed to unsqueeze token_type_ids")?;

        // Forward pass through BERT
        let embeddings = self
            .model
            .forward(&token_ids, &token_type_ids, None)
            .context("BERT forward pass failed")?;

        // Mean pooling across sequence dimension
        let (_batch, seq_len, _hidden) = embeddings
            .dims3()
            .context("Failed to get embedding dimensions")?;
        let pooled = embeddings
            .sum(1)
            .context("Failed to sum embeddings")?
            .broadcast_div(&Tensor::new(seq_len as f64, &self.device)?)
            .context("Failed to divide by seq_len")?;

        // L2 normalization
        let pooled_norm = pooled
            .sqr()
            .context("Failed to square embeddings")?
            .sum_keepdim(1)
            .context("Failed to sum squares")?
            .sqrt()
            .context("Failed to compute sqrt")?;

        let normalized = pooled
            .broadcast_div(&pooled_norm)
            .context("Failed to normalize embeddings")?;

        normalized.squeeze(0).context("Failed to squeeze embeddings")
    }

    /// Compute cosine similarity between two texts.
    /// 
    /// Since embeddings are normalized, this is just the dot product.
    pub fn similarity(&self, text_a: &str, text_b: &str) -> Result<f32> {
        let emb_a = self.embed(text_a)?;
        let emb_b = self.embed(text_b)?;

        let dot_product = (&emb_a * &emb_b)
            .context("Failed to multiply embeddings")?
            .sum_all()
            .context("Failed to sum dot product")?
            .to_scalar::<f32>()
            .context("Failed to convert to scalar")?;

        Ok(dot_product)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require internet access to download the model
    // Run with: cargo test --features test_retrieval -- --ignored
    
    #[test]
    #[ignore = "Requires model download"]
    fn test_embedder_initialization() {
        let embedder = Embedder::new("sentence-transformers/all-MiniLM-L6-v2");
        assert!(embedder.is_ok());
    }

    #[test]
    #[ignore = "Requires model download"]
    fn test_embedding_dimensions() {
        let embedder = Embedder::new("sentence-transformers/all-MiniLM-L6-v2").unwrap();
        let embedding = embedder.embed("test sentence").unwrap();
        
        // MiniLM-L6 produces 384-dimensional embeddings
        let dims = embedding.dims();
        assert_eq!(dims.len(), 1);
        assert_eq!(dims[0], 384);
    }

    #[test]
    #[ignore = "Requires model download"]
    fn test_similarity_semantic() {
        let embedder = Embedder::new("sentence-transformers/all-MiniLM-L6-v2").unwrap();
        
        // Similar concepts should have high similarity
        let sim_similar = embedder.similarity(
            "I want to leave this place",
            "She needs to depart from here"
        ).unwrap();
        
        // Different concepts should have lower similarity  
        let sim_different = embedder.similarity(
            "I want to leave this place",
            "The weather is nice today"
        ).unwrap();
        
        assert!(sim_similar > sim_different, 
            "Similar concepts should have higher similarity: {} vs {}", 
            sim_similar, sim_different);
    }
}
