//! Quantization Module
//!
//! Performance optimization through frame embedding quantization and
//! state compression. Supports f32→f16/int8 embedding reduction and
//! bloom filter state compression for memory efficiency.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Quantized embedding storage for memory efficiency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedEmbedding {
    /// Quantization format
    pub format: QuantizationFormat,
    /// Compressed data
    pub data: Vec<u8>,
    /// Original dimensions
    pub dimensions: Vec<usize>,
    /// Quantization parameters (scale, zero_point)
    pub params: QuantizationParams,
}

/// Supported quantization formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationFormat {
    /// 32-bit float (no quantization)
    F32,
    /// 16-bit integer with per-tensor scaling (2x compression)
    Int16,
    /// 8-bit integer with per-tensor scaling (4x compression)
    Int8,
    /// 4-bit integer with per-tensor scaling (experimental, 8x compression)
    Int4,
}

/// Parameters for quantization/dequantization
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct QuantizationParams {
    /// Scale factor
    pub scale: f32,
    /// Zero point (offset)
    pub zero_point: f32,
    /// Min value in original data
    pub min_val: f32,
    /// Max value in original data
    pub max_val: f32,
}

impl Default for QuantizationParams {
    fn default() -> Self {
        Self {
            scale: 1.0,
            zero_point: 0.0,
            min_val: -1.0,
            max_val: 1.0,
        }
    }
}

/// Embedding quantizer for converting between precision levels
pub struct EmbeddingQuantizer;

impl EmbeddingQuantizer {
    /// Create a new quantizer
    pub fn new() -> Self {
        Self
    }

    /// Quantize a float32 tensor to specified format
    pub fn quantize(&self, tensor: &Tensor, format: QuantizationFormat) -> Result<QuantizedEmbedding> {
        let dimensions = tensor.dims().to_vec();
        
        match format {
            QuantizationFormat::F32 => {
                // No quantization - store as-is
                let data = tensor.to_vec1::<f32>()?;
                let bytes = data.iter()
                    .flat_map(|&f| f.to_le_bytes())
                    .collect();
                
                Ok(QuantizedEmbedding {
                    format,
                    data: bytes,
                    dimensions,
                    params: QuantizationParams::default(),
                })
            }
            QuantizationFormat::Int16 => {
                // Convert to i16 with per-tensor scaling
                let data_f32 = tensor.to_vec1::<f32>()?;
                let min_val = data_f32.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max_val = data_f32.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                
                let scale = (max_val - min_val) / 65535.0;
                let zero_point = min_val;
                
                // Quantize to i16
                let data_i16: Vec<i16> = data_f32.iter()
                    .map(|&x| {
                        let normalized = (x - zero_point) / scale;
                        normalized.clamp(-32768.0, 32767.0) as i16
                    })
                    .collect();
                
                // Pack as bytes
                let bytes = data_i16.iter()
                    .flat_map(|&i| i.to_le_bytes())
                    .collect();
                
                Ok(QuantizedEmbedding {
                    format,
                    data: bytes,
                    dimensions,
                    params: QuantizationParams {
                        scale,
                        zero_point,
                        min_val,
                        max_val,
                    },
                })
            }
            QuantizationFormat::Int8 => {
                // Calculate scale and zero_point
                let data_f32 = tensor.to_vec1::<f32>()?;
                let min_val = data_f32.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max_val = data_f32.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                
                let scale = (max_val - min_val) / 255.0;
                let zero_point = min_val;
                
                // Quantize to int8
                let data_int8: Vec<u8> = data_f32.iter()
                    .map(|&x| {
                        let normalized = (x - zero_point) / scale;
                        normalized.clamp(0.0, 255.0) as u8
                    })
                    .collect();
                
                Ok(QuantizedEmbedding {
                    format,
                    data: data_int8,
                    dimensions,
                    params: QuantizationParams {
                        scale,
                        zero_point,
                        min_val,
                        max_val,
                    },
                })
            }
            QuantizationFormat::Int4 => {
                // Experimental: pack two 4-bit values per byte
                let data_f32 = tensor.to_vec1::<f32>()?;
                let min_val = data_f32.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max_val = data_f32.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                
                let scale = (max_val - min_val) / 15.0;
                let zero_point = min_val;
                
                // Quantize to 4-bit and pack
                let mut data_int4 = Vec::with_capacity(data_f32.len() / 2);
                for chunk in data_f32.chunks(2) {
                    let low = ((chunk[0] - zero_point) / scale).clamp(0.0, 15.0) as u8;
                    let high = if chunk.len() > 1 {
                        ((chunk[1] - zero_point) / scale).clamp(0.0, 15.0) as u8
                    } else {
                        0
                    };
                    data_int4.push((high << 4) | low);
                }
                
                Ok(QuantizedEmbedding {
                    format,
                    data: data_int4,
                    dimensions,
                    params: QuantizationParams {
                        scale,
                        zero_point,
                        min_val,
                        max_val,
                    },
                })
            }
        }
    }

    /// Dequantize back to f32 tensor
    pub fn dequantize(&self, quantized: &QuantizedEmbedding, device: &Device) -> Result<Tensor> {
        match quantized.format {
            QuantizationFormat::F32 => {
                let data_f32: Vec<f32> = quantized.data
                    .chunks_exact(4)
                    .map(|chunk| {
                        let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        f32::from_bits(bits)
                    })
                    .collect();
                
                Tensor::from_vec(data_f32, quantized.dimensions.as_slice(), device)
                    .context("Failed to create tensor from f32 data")
            }
            QuantizationFormat::Int16 => {
                let data_i16: Vec<i16> = quantized.data
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = i16::from_le_bytes([chunk[0], chunk[1]]);
                        bits
                    })
                    .collect();
                
                let data_f32: Vec<f32> = data_i16.iter()
                    .map(|&x| {
                        let normalized = x as f32;
                        normalized * quantized.params.scale + quantized.params.zero_point
                    })
                    .collect();
                
                Tensor::from_vec(data_f32, quantized.dimensions.as_slice(), device)
                    .context("Failed to create tensor from int16 data")
            }
            QuantizationFormat::Int8 => {
                let data_f32: Vec<f32> = quantized.data.iter()
                    .map(|&x| {
                        let normalized = x as f32;
                        normalized * quantized.params.scale + quantized.params.zero_point
                    })
                    .collect();
                
                Tensor::from_vec(data_f32, quantized.dimensions.as_slice(), device)
                    .context("Failed to create tensor from int8 data")
            }
            QuantizationFormat::Int4 => {
                // Unpack 4-bit values
                let mut data_f32 = Vec::with_capacity(quantized.data.len() * 2);
                
                for &packed in &quantized.data {
                    let low = (packed & 0x0F) as f32;
                    let high = ((packed >> 4) & 0x0F) as f32;
                    
                    data_f32.push(low * quantized.params.scale + quantized.params.zero_point);
                    data_f32.push(high * quantized.params.scale + quantized.params.zero_point);
                }
                
                // Truncate to original dimensions if needed
                let total_elements: usize = quantized.dimensions.iter().product();
                data_f32.truncate(total_elements);
                
                Tensor::from_vec(data_f32, quantized.dimensions.as_slice(), device)
                    .context("Failed to create tensor from int4 data")
            }
        }
    }

    /// Calculate expected memory savings
    pub fn memory_savings(original_bytes: usize, format: QuantizationFormat) -> f64 {
        let compressed_bytes = match format {
            QuantizationFormat::F32 => original_bytes,
            QuantizationFormat::Int16 => original_bytes / 2,
            QuantizationFormat::Int8 => original_bytes / 4,
            QuantizationFormat::Int4 => original_bytes / 8,
        };
        
        1.0 - (compressed_bytes as f64 / original_bytes as f64)
    }

    /// Estimate quantization error (cosine similarity degradation)
    pub fn estimate_error(&self, original: &Tensor, quantized: &QuantizedEmbedding, device: &Device) -> Result<f64> {
        let dequantized = self.dequantize(quantized, device)?;
        
        // Calculate cosine similarity
        let original_norm = original.broadcast_mul(original)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let dequantized_norm = dequantized.broadcast_mul(&dequantized)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let dot_product = original.broadcast_mul(&dequantized)?
            .sum_all()?
            .to_scalar::<f32>()?;
        
        let similarity = dot_product / (original_norm.sqrt() * dequantized_norm.sqrt());
        let error = 1.0 - similarity as f64;
        
        Ok(error)
    }
}

impl Default for EmbeddingQuantizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Bloom filter for efficient state set membership testing
///
/// Provides probabilistic set membership with configurable false positive rate.
/// Useful for compressing large state sets with O(1) lookup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomFilter {
    /// Bit array
    bits: Vec<u64>,
    /// Number of hash functions
    num_hashes: usize,
    /// Expected number of elements
    capacity: usize,
    /// Current element count
    count: usize,
    /// Target false positive rate
    target_fpr: f64,
}

impl BloomFilter {
    /// Create a new bloom filter with target false positive rate
    pub fn with_capacity(capacity: usize, target_fpr: f64) -> Self {
        // Calculate optimal size: m = -n * ln(p) / (ln(2)^2)
        let bit_size = ((-1.0 * capacity as f64 * target_fpr.ln()) / (2.0_f64.ln().powi(2))).ceil() as usize;
        let num_hashes = ((bit_size as f64 / capacity as f64) * 2.0_f64.ln()).ceil() as usize;
        
        // Round up to nearest 64 for u64 storage
        let u64_size = (bit_size + 63) / 64;
        
        Self {
            bits: vec![0u64; u64_size],
            num_hashes: num_hashes.max(1),
            capacity,
            count: 0,
            target_fpr,
        }
    }

    /// Create with default 1% false positive rate
    pub fn new(capacity: usize) -> Self {
        Self::with_capacity(capacity, 0.01)
    }

    /// Insert an element
    pub fn insert(&mut self, item: &str) {
        let hashes = self.hash(item);
        
        for hash in hashes {
            let idx = hash % self.bits.len();
            let bit = hash % 64;
            self.bits[idx] |= 1u64 << bit;
        }
        
        self.count += 1;
    }

    /// Check if element might be in set (may have false positives)
    pub fn might_contain(&self, item: &str) -> bool {
        let hashes = self.hash(item);
        
        for hash in hashes {
            let idx = hash % self.bits.len();
            let bit = hash % 64;
            if (self.bits[idx] & (1u64 << bit)) == 0 {
                return false;
            }
        }
        
        true
    }

    /// Get current estimated false positive rate
    pub fn current_fpr(&self) -> f64 {
        // (1 - e^(-k*n/m))^k
        let m = self.bits.len() * 64;
        let k = self.num_hashes as f64;
        let n = self.count as f64;
        
        (1.0 - (-k * n / m as f64).exp()).powf(k)
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.bits.len() * 8
    }

    /// Generate hash values for an item
    fn hash(&self, item: &str) -> Vec<usize> {
        // Use multiple hash functions via seed variation
        (0..self.num_hashes)
            .map(|i| {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                use std::hash::{Hash, Hasher};
                (item, i).hash(&mut hasher);
                hasher.finish() as usize
            })
            .collect()
    }

    /// Estimate memory savings vs storing strings directly
    pub fn estimated_savings(&self, avg_item_size: usize) -> f64 {
        let raw_size = self.count * avg_item_size;
        let compressed_size = self.memory_bytes();
        
        1.0 - (compressed_size as f64 / raw_size as f64)
    }
}

impl Default for BloomFilter {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Quantized state storage for efficient memory usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedStateStore {
    /// Quantized embeddings
    embeddings: HashMap<String, QuantizedEmbedding>,
    /// Bloom filters for state compression
    state_filters: HashMap<String, BloomFilter>,
    /// Default quantization format
    default_format: QuantizationFormat,
}

impl QuantizedStateStore {
    /// Create a new quantized state store
    pub fn new(default_format: QuantizationFormat) -> Self {
        Self {
            embeddings: HashMap::new(),
            state_filters: HashMap::new(),
            default_format,
        }
    }

    /// Store an embedding with quantization
    pub fn store_embedding(&mut self, id: String, tensor: &Tensor) -> Result<()> {
        let quantizer = EmbeddingQuantizer::new();
        let quantized = quantizer.quantize(tensor, self.default_format)?;
        self.embeddings.insert(id, quantized);
        Ok(())
    }

    /// Retrieve and dequantize an embedding
    pub fn get_embedding(&self, id: &str, device: &Device) -> Option<Result<Tensor>> {
        self.embeddings.get(id).map(|q| {
            let quantizer = EmbeddingQuantizer::new();
            quantizer.dequantize(q, device)
        })
    }

    /// Store state set as bloom filter
    pub fn store_state_set(&mut self, id: String, states: &[String]) {
        let mut filter = BloomFilter::with_capacity(states.len() * 2, 0.01);
        
        for state in states {
            filter.insert(state);
        }
        
        self.state_filters.insert(id, filter);
    }

    /// Check if state might be in set
    pub fn might_have_state(&self, id: &str, state: &str) -> bool {
        self.state_filters
            .get(id)
            .map(|f| f.might_contain(state))
            .unwrap_or(false)
    }

    /// Get total memory usage
    pub fn memory_usage(&self) -> usize {
        let embedding_bytes: usize = self.embeddings.values()
            .map(|e| e.data.len())
            .sum();
        
        let filter_bytes: usize = self.state_filters.values()
            .map(|f| f.memory_bytes())
            .sum();
        
        embedding_bytes + filter_bytes
    }

    /// Get compression statistics
    pub fn compression_stats(&self) -> CompressionStats {
        let num_embeddings = self.embeddings.len();
        let num_filters = self.state_filters.len();
        
        let total_bytes: usize = self.embeddings.values()
            .map(|e| e.data.len())
            .sum();
        
        let avg_embedding_bytes = if num_embeddings > 0 {
            total_bytes / num_embeddings
        } else {
            0
        };
        
        CompressionStats {
            num_embeddings,
            num_state_filters: num_filters,
            total_memory_bytes: self.memory_usage(),
            avg_embedding_bytes,
            default_format: self.default_format,
        }
    }
}

impl Default for QuantizedStateStore {
    fn default() -> Self {
        Self::new(QuantizationFormat::Int8)
    }
}

/// Statistics for compression
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub num_embeddings: usize,
    pub num_state_filters: usize,
    pub total_memory_bytes: usize,
    pub avg_embedding_bytes: usize,
    pub default_format: QuantizationFormat,
}

/// Feature flag configuration for optional turboquant backend
#[cfg(feature = "turboquant")]
pub mod turboquant_backend {
    //! Turboquant backend for accelerated quantization operations
    //!
    //! This module provides llama-cpp-turboquant integration for
    //! faster quantization/dequantization on CPU.

    use super::*;

    /// Turboquant-optimized quantizer
    pub struct TurboquantQuantizer;

    impl TurboquantQuantizer {
        /// Check if turboquant is available
        pub fn is_available() -> bool {
            // Would check for llama-cpp-turboquant library
            false
        }

        /// Quantize with turboquant acceleration
        pub fn quantize_fast(&self, _tensor: &Tensor, _format: QuantizationFormat) -> Result<QuantizedEmbedding> {
            // Would use llama-cpp-turboquant GGUF quantization
            unimplemented!("Turboquant backend not yet integrated")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_quantization_int16() {
        let device = Device::Cpu;
        let quantizer = EmbeddingQuantizer::new();
        
        // Create test tensor
        let data: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
        let tensor = Tensor::from_vec(data.clone(), &[1, 384], &device).unwrap();
        
        // Quantize to int16
        let quantized = quantizer.quantize(&tensor, QuantizationFormat::Int16).unwrap();
        
        // Verify size reduction (2 bytes per element vs 4)
        assert!(quantized.data.len() < data.len() * 4);
        
        // Dequantize and verify
        let dequantized = quantizer.dequantize(&quantized, &device).unwrap();
        let dequantized_data = dequantized.to_vec1::<f32>().unwrap();
        
        // Should be reasonably close to original
        for (orig, deq) in data.iter().zip(dequantized_data.iter()) {
            assert!((orig - deq).abs() < 0.01);
        }
    }

    #[test]
    fn test_quantization_int8() {
        let device = Device::Cpu;
        let quantizer = EmbeddingQuantizer::new();
        
        let data: Vec<f32> = vec![0.0, 0.5, 1.0, -0.5, -1.0];
        let tensor = Tensor::from_vec(data, &[1, 5], &device).unwrap();
        
        let quantized = quantizer.quantize(&tensor, QuantizationFormat::Int8).unwrap();
        
        assert_eq!(quantized.format, QuantizationFormat::Int8);
        assert_eq!(quantized.data.len(), 5);
        
        // Verify params are calculated
        assert!(quantized.params.scale > 0.0);
    }

    #[test]
    fn test_bloom_filter() {
        let mut filter = BloomFilter::new(100);
        
        filter.insert("test1");
        filter.insert("test2");
        
        assert!(filter.might_contain("test1"));
        assert!(filter.might_contain("test2"));
        assert!(!filter.might_contain("test3")); // Likely true, but not guaranteed
        
        // Check memory efficiency
        assert!(filter.memory_bytes() < 1000); // Much less than storing 100 strings
    }

    #[test]
    fn test_quantized_state_store() {
        let mut store = QuantizedStateStore::new(QuantizationFormat::Int8);
        let device = Device::Cpu;
        
        // Store embedding
        let data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let tensor = Tensor::from_vec(data, &[1, 4], &device).unwrap();
        store.store_embedding("test".to_string(), &tensor).unwrap();
        
        // Retrieve
        let retrieved = store.get_embedding("test", &device).unwrap().unwrap();
        assert_eq!(retrieved.dims(), &[1, 4]);
        
        // Store state set
        store.store_state_set("states".to_string(), &vec!["active".to_string(), "ready".to_string()]);
        
        assert!(store.might_have_state("states", "active"));
        assert!(!store.might_have_state("states", "unknown"));
    }
}
