//! Edge Inference Module
//! 
//! Automatically generates edges between verbs by analyzing state compatibility.
//! Bridges the gap between isolated verb definitions and connected inference graphs.

use std::collections::{HashMap, HashSet};
use anyhow::Result;
use crate::inference_graph::{InferenceGraph, VerbNode, EdgeType, InferenceEdge};
use crate::state_algebra::StateSet;

/// Configuration for edge inference
pub struct EdgeInferenceConfig {
    /// Minimum confidence threshold for generated edges (0.0-1.0)
    pub min_confidence: f32,
    /// Minimum overlap ratio for state matching (0.0-1.0)
    pub state_overlap_threshold: f32,
    /// Whether to generate negative (disables) edges
    pub generate_negative_edges: bool,
    /// Maximum edges per verb (prevent explosion)
    pub max_edges_per_verb: usize,
}

impl Default for EdgeInferenceConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            state_overlap_threshold: 0.5,
            generate_negative_edges: true,
            max_edges_per_verb: 50,
        }
    }
}

/// Edge generator that creates relationships between verbs based on state analysis
pub struct EdgeGenerator {
    config: EdgeInferenceConfig,
}

/// Types of inferred relationships
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferredRelation {
    Enables,      // A's final states satisfy B's required states
    Disables,     // A's final states conflict with B's required states
    Precedes,    // A temporally precedes B (state progression)
    Causes,      // Strong causal link (specific cause-effect patterns)
}

/// Result of edge inference between two verbs
#[derive(Debug, Clone)]
pub struct InferredEdge {
    pub from: String,
    pub to: String,
    pub relation: InferredRelation,
    pub confidence: f32,
    pub reason: String,
}

impl EdgeGenerator {
    pub fn new(config: EdgeInferenceConfig) -> Self {
        Self { config }
    }

    /// Generate all edges for a collection of verbs
    pub fn generate_edges(&self, verbs: &[VerbNode]) -> Vec<InferredEdge> {
        let mut edges = Vec::new();
        let verb_map: HashMap<String, &VerbNode> = verbs
            .iter()
            .map(|v| (v.id.clone(), v))
            .collect();

        // Pre-compute state signatures for efficiency
        let signatures: HashMap<String, StateSignature> = verbs
            .iter()
            .map(|v| (v.id.clone(), StateSignature::from_verb(v)))
            .collect();

        for (i, verb_a) in verbs.iter().enumerate() {
            let mut verb_edges = 0;
            
            for verb_b in verbs.iter().skip(i + 1) {
                if verb_edges >= self.config.max_edges_per_verb {
                    break;
                }

                let sig_a = signatures.get(&verb_a.id).unwrap();
                let sig_b = signatures.get(&verb_b.id).unwrap();

                // Check A -> B relationships
                if let Some(edge) = self.infer_edge(verb_a, verb_b, sig_a, sig_b) {
                    verb_edges += 1;
                    edges.push(edge);
                }

                // Check B -> A relationships (asymmetric relations)
                if let Some(edge) = self.infer_edge(verb_b, verb_a, sig_b, sig_a) {
                    verb_edges += 1;
                    edges.push(edge);
                }
            }
        }

        edges
    }

    /// Infer edge from verb A to verb B
    fn infer_edge(
        &self,
        from: &VerbNode,
        to: &VerbNode,
        sig_from: &StateSignature,
        sig_to: &StateSignature,
    ) -> Option<InferredEdge> {
        // Skip if same verb
        if from.id == to.id {
            return None;
        }

        // Check for Enables relationship
        let enable_score = self.calculate_enable_score(sig_from, sig_to);
        if enable_score >= self.config.min_confidence {
            return Some(InferredEdge {
                from: from.id.clone(),
                to: to.id.clone(),
                relation: InferredRelation::Enables,
                confidence: enable_score,
                reason: format!(
                    "{}'s final states ({:?}) satisfy {}'s requirements ({:?})",
                    from.verb, sig_from.final_states, to.verb, sig_to.required_states
                ),
            });
        }

        // Check for Disables relationship (if enabled)
        if self.config.generate_negative_edges {
            let disable_score = self.calculate_disable_score(sig_from, sig_to);
            if disable_score >= self.config.min_confidence {
                return Some(InferredEdge {
                    from: from.id.clone(),
                    to: to.id.clone(),
                    relation: InferredRelation::Disables,
                    confidence: disable_score,
                    reason: format!(
                        "{}'s final states conflict with {}'s requirements",
                        from.verb, to.verb
                    ),
                });
            }
        }

        // Check for Precedes relationship (state progression)
        let precede_score = self.calculate_precede_score(sig_from, sig_to);
        if precede_score >= self.config.min_confidence {
            return Some(InferredEdge {
                from: from.id.clone(),
                to: to.id.clone(),
                relation: InferredRelation::Precedes,
                confidence: precede_score,
                reason: format!(
                    "State progression from {} to {}",
                    from.verb, to.verb
                ),
            });
        }

        None
    }

    /// Calculate how well A's final states enable B
    fn calculate_enable_score(&self, sig_a: &StateSignature, sig_b: &StateSignature) -> f32 {
        if sig_b.required_states.is_empty() {
            return 0.0; // Nothing required, no enabling needed
        }

        let intersection: HashSet<_> = sig_a.final_states
            .intersection(&sig_b.required_states)
            .collect();

        let overlap_ratio = intersection.len() as f32 / sig_b.required_states.len() as f32;
        
        if overlap_ratio < self.config.state_overlap_threshold {
            return 0.0;
        }

        // Weight by confidence factors
        let coverage = overlap_ratio;
        let specificity = if sig_a.final_states.len() <= sig_b.required_states.len() {
            1.0 // A is more specific, higher confidence
        } else {
            sig_b.required_states.len() as f32 / sig_a.final_states.len() as f32
        };

        coverage * specificity * 0.95 // Slight penalty for being inferred
    }

    /// Calculate how much A's final states conflict with B
    fn calculate_disable_score(&self, sig_a: &StateSignature, sig_b: &StateSignature) -> f32 {
        // Look for antonyms and contradictions
        let contradictions = find_contradictions(&sig_a.final_states, &sig_b.required_states);
        
        if contradictions.is_empty() {
            return 0.0;
        }

        let conflict_ratio = contradictions.len() as f32 / sig_b.required_states.len().max(1) as f32;
        
        if conflict_ratio < self.config.state_overlap_threshold {
            return 0.0;
        }

        conflict_ratio * 0.9 // Penalty for inferred negative edges
    }

    /// Calculate temporal precedence score
    fn calculate_precede_score(&self, sig_a: &StateSignature, sig_b: &StateSignature) -> f32 {
        // Precedence occurs when A produces states that B naturally follows
        // e.g., "wet" precedes "dry"
        
        let progression_states = vec![
            ("wet", "dry"),
            ("closed", "open"),
            ("locked", "unlocked"),
            ("asleep", "awake"),
            ("hungry", "fed"),
            ("dirty", "clean"),
            ("broken", "fixed"),
            ("cold", "warm"),
            ("dark", "light"),
            ("empty", "full"),
        ];

        let mut matches = 0;
        for (before, after) in &progression_states {
            if sig_a.final_states.iter().any(|s| s.contains(before))
                && sig_b.final_states.iter().any(|s| s.contains(after))
            {
                matches += 1;
            }
        }

        if matches == 0 {
            return 0.0;
        }

        (matches as f32 / progression_states.len() as f32).min(0.95)
    }

    /// Apply generated edges to an inference graph
    pub fn apply_to_graph(&self, graph: &mut InferenceGraph, edges: &[InferredEdge]) {
        for edge in edges {
            let edge_type = match edge.relation {
                InferredRelation::Enables => EdgeType::Enables,
                InferredRelation::Disables => EdgeType::Disables,
                InferredRelation::Precedes => EdgeType::Precedes,
                InferredRelation::Causes => EdgeType::Causes,
            };

            let inference_edge = InferenceEdge {
                edge_type,
                confidence: edge.confidence,
                conditions: vec![edge.reason.clone()],
            };

            // Add edge to graph (graph handles duplicate checking)
            let _ = graph.add_edge(&edge.from, &edge.to, inference_edge);
        }
    }
}

/// Pre-computed state signature for efficient comparison
#[derive(Debug, Clone)]
struct StateSignature {
    required_states: HashSet<String>,
    final_states: HashSet<String>,
}

impl StateSignature {
    fn from_verb(verb: &VerbNode) -> Self {
        let mut required = HashSet::new();
        let mut final_states = HashSet::new();

        // Collect all required states
        required.extend(verb.required_subject_states.physical.iter().cloned());
        required.extend(verb.required_subject_states.emotional.iter().cloned());
        required.extend(verb.required_subject_states.positional.iter().cloned());
        required.extend(verb.required_subject_states.mental.iter().cloned());
        required.extend(verb.required_object_states.physical.iter().cloned());
        required.extend(verb.required_object_states.emotional.iter().cloned());
        required.extend(verb.required_object_states.positional.iter().cloned());
        required.extend(verb.required_object_states.mental.iter().cloned());

        // Collect all final states
        final_states.extend(verb.final_subject_states.physical.iter().cloned());
        final_states.extend(verb.final_subject_states.emotional.iter().cloned());
        final_states.extend(verb.final_subject_states.positional.iter().cloned());
        final_states.extend(verb.final_subject_states.mental.iter().cloned());
        final_states.extend(verb.final_object_states.physical.iter().cloned());
        final_states.extend(verb.final_object_states.emotional.iter().cloned());
        final_states.extend(verb.final_object_states.positional.iter().cloned());
        final_states.extend(verb.final_object_states.mental.iter().cloned());

        Self {
            required_states: required,
            final_states,
        }
    }
}

/// Find contradicting state pairs
fn find_contradictions(states_a: &HashSet<String>, states_b: &HashSet<String>) -> Vec<(String, String)> {
    let mut contradictions = Vec::new();

    // Define known antonym pairs
    let antonyms: Vec<(&str, &str)> = vec![
        ("active", "inactive"),
        ("on", "off"),
        ("open", "closed"),
        ("locked", "unlocked"),
        ("wet", "dry"),
        ("hot", "cold"),
        ("full", "empty"),
        ("alive", "dead"),
        ("present", "absent"),
        ("visible", "hidden"),
        ("intact", "broken"),
        ("clean", "dirty"),
        ("functional", "broken"),
        ("connected", "disconnected"),
    ];

    for (a, b) in antonyms {
        let a_in_a = states_a.iter().any(|s| s.contains(a));
        let b_in_b = states_b.iter().any(|s| s.contains(b));
        let b_in_a = states_a.iter().any(|s| s.contains(b));
        let a_in_b = states_b.iter().any(|s| s.contains(a));

        if (a_in_a && b_in_b) || (b_in_a && a_in_b) {
            contradictions.push((a.to_string(), b.to_string()));
        }
    }

    contradictions
}

/// Batch edge generation for large verb collections
pub fn generate_edges_batch(
    verbs: &[VerbNode],
    config: EdgeInferenceConfig,
) -> Vec<InferredEdge> {
    let generator = EdgeGenerator::new(config);
    generator.generate_edges(verbs)
}

/// Convenience function to build a fully connected graph from verb files
pub fn build_connected_graph(
    verb_directory: &str,
    config: Option<EdgeInferenceConfig>,
) -> Result<InferenceGraph> {
    use crate::inference_graph::load_verbs_from_directory;

    let mut graph = InferenceGraph::new();
    let config = config.unwrap_or_default();

    // Load verbs
    let verbs = load_verbs_from_directory(verb_directory)?;
    
    // Add verbs to graph
    for verb in &verbs {
        graph.add_verb(verb.clone());
    }

    // Generate and add edges
    let generator = EdgeGenerator::new(config);
    let edges = generator.generate_edges(&verbs);
    generator.apply_to_graph(&mut graph, &edges);

    println!("Built connected graph: {} verbs, {} edges", 
        graph.verb_count(), 
        edges.len()
    );

    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_algebra::StateSet;

    fn create_test_verb(
        id: &str,
        required: Vec<&str>,
        final_states: Vec<&str>,
    ) -> VerbNode {
        VerbNode {
            id: id.to_string(),
            verb: id.to_string(),
            applicable_subjects: vec!["test".to_string()],
            applicable_objects: vec!["test".to_string()],
            required_subject_states: StateSet {
                physical: required.iter().map(|s| s.to_string()).collect(),
                emotional: HashSet::new(),
                positional: HashSet::new(),
                mental: HashSet::new(),
            },
            required_object_states: StateSet {
                physical: HashSet::new(),
                emotional: HashSet::new(),
                positional: HashSet::new(),
                mental: HashSet::new(),
            },
            final_subject_states: StateSet {
                physical: HashSet::new(),
                emotional: HashSet::new(),
                positional: HashSet::new(),
                mental: HashSet::new(),
            },
            final_object_states: StateSet {
                physical: final_states.iter().map(|s| s.to_string()).collect(),
                emotional: HashSet::new(),
                positional: HashSet::new(),
                mental: HashSet::new(),
            },
            version: "1.0.0".to_string(),
        }
    }

    #[test]
    fn test_enable_detection() {
        let verb_a = create_test_verb("prepare", vec![], vec!["ready", "prepared"]);
        let verb_b = create_test_verb("execute", vec!["ready", "prepared"], vec![]);

        let config = EdgeInferenceConfig::default();
        let generator = EdgeGenerator::new(config);

        let edges = generator.generate_edges(&[verb_a, verb_b]);
        
        assert!(!edges.is_empty());
        assert!(edges.iter().any(|e| e.relation == InferredRelation::Enables));
    }

    #[test]
    fn test_disable_detection() {
        let verb_a = create_test_verb("destroy", vec![], vec!["broken", "inactive"]);
        let verb_b = create_test_verb("use", vec!["functional", "intact"], vec![]);

        let config = EdgeInferenceConfig {
            generate_negative_edges: true,
            ..Default::default()
        };
        let generator = EdgeGenerator::new(config);

        let edges = generator.generate_edges(&[verb_a, verb_b]);
        
        // Should detect that destroy disables use
        assert!(edges.iter().any(|e| e.relation == InferredRelation::Disables));
    }

    #[test]
    fn test_precede_detection() {
        let verb_a = create_test_verb("wet", vec![], vec!["wet", "moist"]);
        let verb_b = create_test_verb("dry", vec![], vec!["dry", "arid"]);

        let config = EdgeInferenceConfig::default();
        let generator = EdgeGenerator::new(config);

        let edges = generator.generate_edges(&[verb_a, verb_b]);
        
        // May detect wet -> dry as temporal progression
        println!("Generated edges: {:?}", edges);
    }
}
