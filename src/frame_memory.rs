//! Frame Memory Module
//!
//! Manages semantic frames (verb/noun schemas) with embedding-based retrieval.
//! Adapted from Oxidized-GPT memory.rs for nameless_vector grounding layer.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::Tensor;

use crate::retrieval::Embedder;
use crate::state_algebra::StateSet;

/// Represents the structured outcome of a verb action.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VerbFrame {
    pub verb: String,
    pub applicable_subjects: Vec<String>,
    pub applicable_objects: Vec<String>,
    pub required_subject_states: FrameStates,
    pub required_object_states: FrameStates,
    pub final_subject_states: FrameStates,
    pub final_object_states: FrameStates,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FrameStates {
    pub physical: Vec<String>,
    pub emotional: Vec<String>,
    pub positional: Vec<String>,
    pub mental: Vec<String>,
}

impl From<&FrameStates> for StateSet {
    fn from(states: &FrameStates) -> Self {
        StateSet {
            physical: states.physical.iter().cloned().collect(),
            emotional: states.emotional.iter().cloned().collect(),
            positional: states.positional.iter().cloned().collect(),
            mental: states.mental.iter().cloned().collect(),
        }
    }
}

/// An indexed frame with pre-computed embedding for fast retrieval.
#[derive(Debug, Clone)]
pub struct IndexedFrame {
    pub frame: VerbFrame,
    pub embedding: Tensor,
    pub text_representation: String,
}

/// Semantic memory for verb frames with embedding-based retrieval.
pub struct FrameMemory {
    /// Verb ID -> indexed frame with embedding
    pub frames: HashMap<String, IndexedFrame>,
    /// Cached embedder for on-the-fly queries
    embedder: Option<Embedder>,
}

impl FrameMemory {
    /// Create empty memory (without embedder - for loading only).
    pub fn new() -> Self {
        Self {
            frames: HashMap::new(),
            embedder: None,
        }
    }

    /// Create memory with embedder for semantic search.
    pub fn with_embedder(embedder: Embedder) -> Self {
        Self {
            frames: HashMap::new(),
            embedder: Some(embedder),
        }
    }

    /// Load frames from verb_state directory with embedding generation.
    /// 
    /// Expects JSON files matching the nameless_vector verb_state format.
    /// Load frames from directory with embedder for on-the-fly queries.
    pub fn load_with_embedder(dir: &str, embedder: Embedder) -> Result<Self> {
        let mut memory = Self::with_embedder(embedder);
        memory.load_from_dir(dir)?;
        Ok(memory)
    }

    fn load_from_dir(&mut self, dir: &str) -> Result<()> {
        let path = Path::new(dir);

        if !path.exists() {
            tracing::warn!("Memory directory {:?} does not exist yet", path);
            return Ok(());
        }

        for entry in fs::read_dir(path)
            .with_context(|| format!("Failed to read directory {:?}", path))? {
            let entry = entry
                .with_context(|| "Failed to read directory entry")?;
            let file_path = entry.path();

            if file_path.extension().and_then(|s| s.to_str()) == Some("json") {
                self.load_file(&file_path)?;
            }
        }

        tracing::info!("Loaded {} indexed verb frames", self.frames.len());
        Ok(())
    }

    /// Load a single JSON file containing verb outcomes.
    fn load_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read {:?}", path))?;

        // Collect frames to insert
        let mut frames_to_insert: Vec<(VerbFrame, Option<Tensor>, String)> = Vec::new();

        // Try parsing as PrefixState first (verb_state format)
        if let Ok(prefix_state) = serde_json::from_str::<PrefixState>(&content) {
            for outcome in prefix_state.outcomes {
                let frame = verb_outcome_to_frame(outcome);
                if let Some(ref emb) = self.embedder {
                    let text_repr = format!(
                        "Verb: {}. Subjects: {}. Objects: {}.",
                        frame.verb,
                        frame.applicable_subjects.join(", "),
                        frame.applicable_objects.join(", ")
                    );
                    let embedding = emb.embed(&text_repr)?;
                    frames_to_insert.push((frame, Some(embedding), text_repr));
                } else {
                    frames_to_insert.push((frame, None, String::new()));
                }
            }
        }
        // Try parsing as direct Vec<VerbOutcome>
        else if let Ok(outcomes) = serde_json::from_str::<Vec<VerbOutcome>>(&content) {
            for outcome in outcomes {
                let frame = verb_outcome_to_frame(outcome);
                if let Some(ref emb) = self.embedder {
                    let text_repr = format!(
                        "Verb: {}. Subjects: {}. Objects: {}.",
                        frame.verb,
                        frame.applicable_subjects.join(", "),
                        frame.applicable_objects.join(", ")
                    );
                    let embedding = emb.embed(&text_repr)?;
                    frames_to_insert.push((frame, Some(embedding), text_repr));
                } else {
                    frames_to_insert.push((frame, None, String::new()));
                }
            }
        }
        // Try parsing as single VerbOutcome
        else if let Ok(outcome) = serde_json::from_str::<VerbOutcome>(&content) {
            let frame = verb_outcome_to_frame(outcome);
            if let Some(ref emb) = self.embedder {
                let text_repr = format!(
                    "Verb: {}. Subjects: {}. Objects: {}.",
                    frame.verb,
                    frame.applicable_subjects.join(", "),
                    frame.applicable_objects.join(", ")
                );
                let embedding = emb.embed(&text_repr)?;
                frames_to_insert.push((frame, Some(embedding), text_repr));
            } else {
                frames_to_insert.push((frame, None, String::new()));
            }
        } else {
            return Err(anyhow::anyhow!(
                "Failed to parse {:?} as any known format",
                path
            ));
        }

        // Now insert all frames
        for (frame, embedding, text_repr) in frames_to_insert {
            let indexed = IndexedFrame {
                frame,
                embedding: embedding.unwrap_or_else(|| Tensor::zeros((1, 384), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap()),
                text_representation: text_repr,
            };
            self.frames.insert(indexed.frame.verb.clone(), indexed);
        }

        Ok(())
    }

    /// Index a single frame with embedding.
    fn index_frame(&mut self, frame: VerbFrame, embedder: &Embedder) -> Result<()> {
        let text_representation = format!(
            "Verb: {}. Subjects: {}. Objects: {}.",
            frame.verb,
            frame.applicable_subjects.join(", "),
            frame.applicable_objects.join(", ")
        );

        let embedding = embedder
            .embed(&text_representation)
            .with_context(|| format!("Failed to embed frame for '{}'", frame.verb))?;

        let verb_id = frame.verb.clone();
        let indexed = IndexedFrame {
            frame,
            embedding,
            text_representation,
        };

        self.frames.insert(verb_id, indexed);
        Ok(())
    }

    /// Find the most semantically similar frame to a query text.
    /// 
    /// Returns (verb_id, similarity_score, frame) or None if no embedder.
    pub fn find_closest(&self, query: &str) -> Result<Option<(String, f32, &VerbFrame)>> {
        let embedder = match &self.embedder {
            Some(e) => e,
            None => return Ok(None),
        };

        let query_emb = embedder
            .embed(query)
            .with_context(|| format!("Failed to embed query: {}", query))?;

        let mut best_match: Option<(String, f32, &VerbFrame)> = None;

        for (verb_id, indexed) in &self.frames {
            let similarity = compute_cosine_similarity(&query_emb, &indexed.embedding)
                .with_context(|| format!("Failed to compute similarity for '{}'", verb_id))?;

            if best_match.is_none() || similarity > best_match.as_ref().unwrap().1 {
                best_match = Some((verb_id.clone(), similarity, &indexed.frame));
            }
        }

        Ok(best_match)
    }

    /// Find top-K most similar frames.
    pub fn find_top_k(&self, query: &str, k: usize) -> Result<Vec<(String, f32, &VerbFrame)>> {
        let embedder = match &self.embedder {
            Some(e) => e,
            None => return Ok(Vec::new()),
        };

        let query_emb = embedder
            .embed(query)
            .with_context(|| format!("Failed to embed query: {}", query))?;

        let mut scored: Vec<(String, f32, &VerbFrame)> = Vec::new();

        for (verb_id, indexed) in &self.frames {
            let similarity = compute_cosine_similarity(&query_emb, &indexed.embedding)
                .with_context(|| format!("Failed to compute similarity for '{}'", verb_id))?;

            scored.push((verb_id.clone(), similarity, &indexed.frame));
        }

        // Sort by similarity descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        Ok(scored)
    }

    /// Get a frame by verb ID (exact match).
    pub fn get_frame(&self, verb: &str) -> Option<&VerbFrame> {
        self.frames.get(verb).map(|indexed| &indexed.frame)
    }

    /// Total number of indexed frames.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

impl Default for FrameMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute cosine similarity between two normalized embeddings.
fn compute_cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32> {
    let dot = (a * b)?.sum_all()?.to_scalar::<f32>()?;
    Ok(dot)
}

/// Legacy format for deserializing verb_state files.
#[derive(Debug, Deserialize)]
struct PrefixState {
    #[allow(dead_code)]
    prefix: String,
    outcomes: Vec<VerbOutcome>,
}

/// Legacy VerbOutcome format (matches Oxidized-GPT and nameless_generator).
#[derive(Debug, Serialize, Deserialize, Clone)]
struct VerbOutcome {
    verb: String,
    applicable_subjects: Vec<String>,
    applicable_objects: Vec<String>,
    required_subject_states: FrameStates,
    required_object_states: FrameStates,
    final_subject_states: FrameStates,
    final_object_states: FrameStates,
}

/// Convert legacy VerbOutcome to VerbFrame.
fn verb_outcome_to_frame(outcome: VerbOutcome) -> VerbFrame {
    VerbFrame {
        verb: outcome.verb,
        applicable_subjects: outcome.applicable_subjects,
        applicable_objects: outcome.applicable_objects,
        required_subject_states: outcome.required_subject_states,
        required_object_states: outcome.required_object_states,
        final_subject_states: outcome.final_subject_states,
        final_object_states: outcome.final_object_states,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_memory_empty() {
        let memory = FrameMemory::new();
        assert!(memory.is_empty());
        assert_eq!(memory.len(), 0);
    }

    #[test]
    fn test_verb_outcome_conversion() {
        let outcome = VerbOutcome {
            verb: "test".to_string(),
            applicable_subjects: vec!["biological_body".to_string()],
            applicable_objects: vec!["object".to_string()],
            required_subject_states: FrameStates {
                physical: vec!["present".to_string()],
                emotional: vec![],
                positional: vec![],
                mental: vec!["aware".to_string()],
            },
            required_object_states: FrameStates {
                physical: vec!["exists".to_string()],
                emotional: vec![],
                positional: vec![],
                mental: vec![],
            },
            final_subject_states: FrameStates {
                physical: vec![],
                emotional: vec!["satisfied".to_string()],
                positional: vec![],
                mental: vec![],
            },
            final_object_states: FrameStates {
                physical: vec!["modified".to_string()],
                emotional: vec![],
                positional: vec![],
                mental: vec![],
            },
        };

        let frame = verb_outcome_to_frame(outcome);
        assert_eq!(frame.verb, "test");
        assert_eq!(frame.applicable_subjects, vec!["biological_body"]);
    }
}
