//! Inference Graph Module
//! 
//! Provides graph-based indexing of verbs for efficient multi-hop inference.
//! Replaces O(n) file scans with O(1) or O(log n) lookups.

use std::collections::{HashMap, HashSet};
use anyhow::{Context, Result};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use crate::state_algebra::{StateSet, VerbApplicabilityChecker};

/// Unique identifier for a verb in the graph
pub type VerbId = String;

/// A node in the inference graph representing a verb
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerbNode {
    pub id: VerbId,
    pub verb: String,
    pub applicable_subjects: Vec<String>,
    pub applicable_objects: Vec<String>,
    pub required_subject_states: StateSet,
    pub required_object_states: StateSet,
    pub final_subject_states: StateSet,
    pub final_object_states: StateSet,
    // Schema version for migrations
    pub version: String,
}

/// Types of edges between verbs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeType {
    /// Verb A enables verb B (makes B possible)
    Enables,
    /// Verb A disables verb B (prevents B)
    Disables,
    /// Verb A precedes verb B (temporal sequence)
    Precedes,
    /// Verb A causes verb B (causal chain)
    Causes,
    /// Verbs are mutually exclusive (cannot both happen)
    Mutex,
    /// Verbs can occur concurrently
    Concurrent,
}

/// An edge in the inference graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceEdge {
    pub edge_type: EdgeType,
    pub confidence: f32, // 0.0 to 1.0
    pub conditions: Vec<String>, // State conditions for this edge to be valid
}

/// The main inference graph structure
pub struct InferenceGraph {
    graph: DiGraph<VerbNode, InferenceEdge>,
    // Indices for O(1) lookups
    verb_to_index: HashMap<VerbId, NodeIndex>,
    subject_index: HashMap<String, HashSet<VerbId>>,
    object_index: HashMap<String, HashSet<VerbId>>,
    state_index: HashMap<String, HashSet<VerbId>>, // State -> verbs requiring it
    effect_index: HashMap<String, HashSet<VerbId>>, // State -> verbs producing it
}

impl InferenceGraph {
    /// Create a new empty inference graph
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            verb_to_index: HashMap::new(),
            subject_index: HashMap::new(),
            object_index: HashMap::new(),
            state_index: HashMap::new(),
            effect_index: HashMap::new(),
        }
    }

    /// Add a verb to the graph
    pub fn add_verb(&mut self, node: VerbNode) -> NodeIndex {
        let id = node.id.clone();
        let index = self.graph.add_node(node.clone());
        self.verb_to_index.insert(id.clone(), index);

        // Update indices
        for subject in &node.applicable_subjects {
            self.subject_index
                .entry(subject.clone())
                .or_default()
                .insert(id.clone());
        }

        for object in &node.applicable_objects {
            self.object_index
                .entry(object.clone())
                .or_default()
                .insert(id.clone());
        }

        // Index required states
        for state in &node.required_subject_states.physical {
            self.state_index
                .entry(format!("subject:physical:{}", state))
                .or_default()
                .insert(id.clone());
        }
        for state in &node.required_object_states.physical {
            self.state_index
                .entry(format!("object:physical:{}", state))
                .or_default()
                .insert(id.clone());
        }

        // Index produced states (effects)
        for state in &node.final_subject_states.physical {
            self.effect_index
                .entry(format!("subject:physical:{}", state))
                .or_default()
                .insert(id.clone());
        }
        for state in &node.final_object_states.physical {
            self.effect_index
                .entry(format!("object:physical:{}", state))
                .or_default()
                .insert(id.clone());
        }

        index
    }

    /// Add an edge between two verbs
    pub fn add_edge(
        &mut self,
        from: &VerbId,
        to: &VerbId,
        edge: InferenceEdge,
    ) -> Result<()> {
        let from_idx = self.verb_to_index.get(from)
            .with_context(|| format!("Verb {} not found", from))?;
        let to_idx = self.verb_to_index.get(to)
            .with_context(|| format!("Verb {} not found", to))?;

        self.graph.add_edge(*from_idx, *to_idx, edge);
        Ok(())
    }

    /// Find all verbs applicable to a given subject type
    pub fn find_verbs_by_subject(&self, subject_type: &str) -> Vec<&VerbNode> {
        self.subject_index
            .get(subject_type)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.verb_to_index.get(id))
                    .filter_map(|idx| self.graph.node_weight(*idx))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find all verbs applicable to a given object type
    pub fn find_verbs_by_object(&self, object_type: &str) -> Vec<&VerbNode> {
        self.object_index
            .get(object_type)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.verb_to_index.get(id))
                    .filter_map(|idx| self.graph.node_weight(*idx))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find verbs that can be applied given current subject/object states
    pub fn find_applicable_verbs(
        &self,
        subject_state: &StateSet,
        object_state: &StateSet,
        subject_type: Option<&str>,
        object_type: Option<&str>,
    ) -> Vec<&VerbNode> {
        let candidates: HashSet<&VerbId> = match (subject_type, object_type) {
            (Some(subj), Some(obj)) => {
                // Intersection of verbs applicable to both subject and object types
                let subject_verbs = self.subject_index.get(subj).map(|s| s.iter().collect::<HashSet<_>>()).unwrap_or_default();
                let object_verbs = self.object_index.get(obj).map(|o| o.iter().collect::<HashSet<_>>()).unwrap_or_default();
                subject_verbs.intersection(&object_verbs).copied().collect()
            }
            (Some(subj), None) => self.subject_index.get(subj).map(|s| s.iter().collect()).unwrap_or_default(),
            (None, Some(obj)) => self.object_index.get(obj).map(|o| o.iter().collect()).unwrap_or_default(),
            (None, None) => self.verb_to_index.keys().collect(),
        };

        candidates
            .iter()
            .filter_map(|id| self.verb_to_index.get(*id))
            .filter_map(|idx| self.graph.node_weight(*idx))
            .filter(|node| {
                VerbApplicabilityChecker::can_apply(
                    subject_state,
                    object_state,
                    &node.required_subject_states,
                    &node.required_object_states,
                )
            })
            .collect()
    }

    /// Find what happens next (verbs enabled by current state)
    pub fn find_enabled_verbs(&self, current_state: &StateSet) -> Vec<&VerbNode> {
        self.effect_index
            .iter()
            .filter(|(key, _)| current_state.physical.contains(key.strip_prefix("subject:physical:").unwrap_or(key)))
            .flat_map(|(_, ids)| ids)
            .filter_map(|id| self.verb_to_index.get(id))
            .filter_map(|idx| self.graph.node_weight(*idx))
            .collect()
    }

    /// Multi-hop inference: find paths from current state to goal state
    pub fn find_state_transition_path(
        &self,
        current_state: &StateSet,
        goal_state: &StateSet,
        max_depth: usize,
    ) -> Vec<Vec<&VerbNode>> {
        let mut paths = Vec::new();
        let mut current_path: Vec<&VerbNode> = Vec::new();
        let mut visited: HashSet<VerbId> = HashSet::new();

        self.dfs_paths(
            current_state.clone(),
            goal_state,
            max_depth,
            0,
            &mut current_path,
            &mut visited,
            &mut paths,
        );

        paths
    }

    fn dfs_paths<'a>(
        &'a self,
        current_state: StateSet,
        goal_state: &StateSet,
        max_depth: usize,
        depth: usize,
        path: &mut Vec<&'a VerbNode>,
        visited: &mut HashSet<VerbId>,
        paths: &mut Vec<Vec<&'a VerbNode>>,
    ) {
        if depth > max_depth {
            return;
        }

        // Check if goal is satisfied
        if current_state.satisfies(goal_state) {
            paths.push(path.clone());
            return;
        }

        // Find applicable verbs
        let applicable = self.find_applicable_verbs(
            &current_state,
            &StateSet::new(), // Object state - could be parameterized
            None,
            None,
        );

        for verb in applicable {
            if visited.contains(&verb.id) {
                continue;
            }

            // Compute new state
            if let Ok((new_subject, _)) = VerbApplicabilityChecker::apply_verb(
                &current_state,
                &StateSet::new(),
                &verb.final_subject_states,
                &verb.final_object_states,
            ) {
                let new_state = current_state.merge(&new_subject).unwrap_or(new_subject);
                
                visited.insert(verb.id.clone());
                path.push(verb);

                self.dfs_paths(
                    new_state,
                    goal_state,
                    max_depth,
                    depth + 1,
                    path,
                    visited,
                    paths,
                );

                path.pop();
                visited.remove(&verb.id);
            }
        }
    }

    /// Find temporal/causal chains (verbs that follow from a given verb)
    pub fn find_successors(&self, verb_id: &VerbId, edge_types: Option<&[EdgeType]>) -> Vec<(&VerbNode, &InferenceEdge)> {
        let idx = match self.verb_to_index.get(verb_id) {
            Some(i) => *i,
            None => return Vec::new(),
        };

        self.graph
            .edges(idx)
            .filter(|edge| {
                if let Some(types) = edge_types {
                    types.contains(&edge.weight().edge_type)
                } else {
                    true
                }
            })
            .filter_map(|edge| {
                self.graph
                    .node_weight(edge.target())
                    .map(|node| (node, edge.weight()))
            })
            .collect()
    }

    /// Get a verb by ID
    pub fn get_verb(&self, id: &VerbId) -> Option<&VerbNode> {
        self.verb_to_index
            .get(id)
            .and_then(|idx| self.graph.node_weight(*idx))
    }

    /// Total number of verbs in the graph
    pub fn verb_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Total number of relationships (edges) in the graph
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
}

impl Default for InferenceGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Load verbs from the verb_state directory into the graph
pub fn load_verbs_from_directory(path: &str) -> Result<Vec<VerbNode>> {
    let mut verbs = Vec::new();
    let entries = std::fs::read_dir(path)
        .with_context(|| format!("Failed to read directory: {}", path))?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().map_or(false, |ext| ext == "json") {
            let content = std::fs::read_to_string(&path)?;
            
            // Try to parse as PrefixState first (for verb_state files)
            if let Ok(state) = serde_json::from_str::<PrefixState>(&content) {
                for outcome in state.outcomes {
                    let node = verb_outcome_to_node(&outcome);
                    verbs.push(node);
                }
            }
        }
    }

    Ok(verbs)
}

/// Convert VerbOutcome to VerbNode
fn verb_outcome_to_node(outcome: &VerbOutcome) -> VerbNode {
    VerbNode {
        id: outcome.verb.clone(),
        verb: outcome.verb.clone(),
        applicable_subjects: outcome.applicable_subjects.clone(),
        applicable_objects: outcome.applicable_objects.clone(),
        required_subject_states: state_set_from_required(&outcome.required_subject_states),
        required_object_states: state_set_from_required(&outcome.required_object_states),
        final_subject_states: state_set_from_final(&outcome.final_subject_states),
        final_object_states: state_set_from_final(&outcome.final_object_states),
        version: "1.0.0".to_string(),
    }
}

/// Helper to convert RequiredStates to StateSet
fn state_set_from_required(states: &RequiredStates) -> StateSet {
    StateSet {
        physical: states.physical.iter().cloned().collect(),
        emotional: states.emotional.iter().cloned().collect(),
        positional: states.positional.iter().cloned().collect(),
        mental: states.mental.iter().cloned().collect(),
    }
}

/// Helper to convert FinalStates to StateSet
fn state_set_from_final(states: &FinalStates) -> StateSet {
    StateSet {
        physical: states.physical.iter().cloned().collect(),
        emotional: states.emotional.iter().cloned().collect(),
        positional: states.positional.iter().cloned().collect(),
        mental: states.mental.iter().cloned().collect(),
    }
}

// Legacy structures for deserialization
#[derive(Debug, Deserialize)]
struct PrefixState {
    prefix: String,
    outcomes: Vec<VerbOutcome>,
}

#[derive(Debug, Deserialize, Clone)]
struct VerbOutcome {
    verb: String,
    applicable_subjects: Vec<String>,
    applicable_objects: Vec<String>,
    required_subject_states: RequiredStates,
    required_object_states: RequiredStates,
    final_subject_states: FinalStates,
    final_object_states: FinalStates,
}

#[derive(Debug, Deserialize, Clone)]
struct RequiredStates {
    physical: Vec<String>,
    emotional: Vec<String>,
    positional: Vec<String>,
    mental: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct FinalStates {
    physical: Vec<String>,
    emotional: Vec<String>,
    positional: Vec<String>,
    mental: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_construction() {
        let mut graph = InferenceGraph::new();

        let node = VerbNode {
            id: "test".to_string(),
            verb: "test".to_string(),
            applicable_subjects: vec!["biological_body".to_string()],
            applicable_objects: vec!["object".to_string()],
            required_subject_states: StateSet::new(),
            required_object_states: StateSet::new(),
            final_subject_states: StateSet::new(),
            final_object_states: StateSet::new(),
            version: "1.0.0".to_string(),
        };

        graph.add_verb(node);
        assert_eq!(graph.verb_count(), 1);
    }

    #[test]
    fn test_subject_index() {
        let mut graph = InferenceGraph::new();

        let node = VerbNode {
            id: "destroy".to_string(),
            verb: "destroy".to_string(),
            applicable_subjects: vec!["storm".to_string()],
            applicable_objects: vec!["crops".to_string()],
            required_subject_states: StateSet::new(),
            required_object_states: StateSet::new(),
            final_subject_states: StateSet::new(),
            final_object_states: StateSet::new(),
            version: "1.0.0".to_string(),
        };

        graph.add_verb(node);

        let verbs = graph.find_verbs_by_subject("storm");
        assert_eq!(verbs.len(), 1);
        assert_eq!(verbs[0].verb, "destroy");
    }
}
