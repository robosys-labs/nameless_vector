//! Temporal Chaining Module
//! 
//! Provides temporal logic for verb sequences, causal chains, and planning.
//! P3 requirement for time-aware interaction inference.

use std::collections::{HashMap, HashSet, VecDeque};
use anyhow::{Context, Result};
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};

use crate::inference_graph::{InferenceGraph, VerbNode};
use crate::state_algebra::StateSet;

/// Temporal relation types between events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalRelation {
    /// A strictly precedes B (A ends before B starts)
    Precedes,
    /// A meets B (A ends when B starts)
    Meets,
    /// A overlaps with B (A starts before B, ends after B starts)
    Overlaps,
    /// A during B (A starts and ends during B)
    During,
    /// A starts B (A and B start together, A ends first)
    Starts,
    /// A finishes B (A and B end together, A starts after B)
    Finishes,
    /// A equals B (same start and end)
    Equals,
    /// A enables B (A makes B possible)
    Enables,
    /// A disables B (A prevents B)
    Disables,
    /// A causes B (A directly leads to B)
    Causes,
    /// Inverse relations
    MetBy,
    OverlappedBy,
    Contains,
    StartedBy,
    FinishedBy,
    EnabledBy,
    DisabledBy,
    CausedBy,
}

impl TemporalRelation {
    /// Check if this relation is transitive
    pub fn is_transitive(&self) -> bool {
        matches!(
            self,
            TemporalRelation::Precedes
                | TemporalRelation::Causes
                | TemporalRelation::Enables
                | TemporalRelation::CausedBy
        )
    }

    /// Get the inverse relation
    pub fn inverse(&self) -> Self {
        match self {
            TemporalRelation::Precedes => TemporalRelation::Precedes, // Not strictly inverse but useful
            TemporalRelation::Meets => TemporalRelation::MetBy,
            TemporalRelation::Overlaps => TemporalRelation::OverlappedBy,
            TemporalRelation::During => TemporalRelation::Contains,
            TemporalRelation::Starts => TemporalRelation::StartedBy,
            TemporalRelation::Finishes => TemporalRelation::FinishedBy,
            TemporalRelation::Equals => TemporalRelation::Equals,
            TemporalRelation::Enables => TemporalRelation::EnabledBy,
            TemporalRelation::Disables => TemporalRelation::DisabledBy,
            TemporalRelation::Causes => TemporalRelation::CausedBy,
            // Inverse relations map back to original
            TemporalRelation::MetBy => TemporalRelation::Meets,
            TemporalRelation::OverlappedBy => TemporalRelation::Overlaps,
            TemporalRelation::Contains => TemporalRelation::During,
            TemporalRelation::StartedBy => TemporalRelation::Starts,
            TemporalRelation::FinishedBy => TemporalRelation::Finishes,
            TemporalRelation::EnabledBy => TemporalRelation::Enables,
            TemporalRelation::DisabledBy => TemporalRelation::Disables,
            TemporalRelation::CausedBy => TemporalRelation::Causes,
        }
    }
}

/// A temporal event with timing constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEvent {
    pub verb_id: String,
    pub start_time: Option<Timestamp>,
    pub duration: Option<Duration>,
    pub constraints: Vec<TemporalConstraint>,
}

/// Timestamp representation (relative or absolute)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Timestamp {
    /// Absolute Unix timestamp
    Absolute(u64),
    /// Relative offset from sequence start
    Relative(u64),
    /// Unspecified (flexible)
    Unspecified,
}

/// Duration specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Duration {
    /// Fixed duration in milliseconds
    Fixed(u64),
    /// Minimum duration
    AtLeast(u64),
    /// Maximum duration
    AtMost(u64),
    /// Range (min, max)
    Range(u64, u64),
    /// Unspecified
    Unspecified,
}

/// Temporal constraint on events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConstraint {
    pub relation: TemporalRelation,
    pub target_verb: String,
    pub optional: bool,
}

/// Temporal graph extending inference graph with timing
pub struct TemporalGraph {
    /// Base inference graph
    pub inference_graph: InferenceGraph,
    /// Temporal constraints between verbs
    pub temporal_edges: HashMap<(String, String), TemporalEdge>,
    /// Causal chains (directed acyclic graphs)
    pub causal_chains: Vec<Vec<String>>,
}

#[derive(Debug, Clone)]
struct TemporalEdge {
    relation: TemporalRelation,
    confidence: f32,
    conditions: Vec<String>,
}

impl TemporalGraph {
    /// Create new temporal graph from inference graph
    pub fn new(inference_graph: InferenceGraph) -> Self {
        Self {
            inference_graph,
            temporal_edges: HashMap::new(),
            causal_chains: vec![],
        }
    }

    /// Add temporal edge between verbs
    pub fn add_temporal_edge(
        &mut self,
        from: &str,
        to: &str,
        relation: TemporalRelation,
        confidence: f32,
    ) {
        self.temporal_edges.insert(
            (from.to_string(), to.to_string()),
            TemporalEdge {
                relation,
                confidence,
                conditions: vec![],
            },
        );
    }

    /// Check if verb A can precede verb B given current state
    pub fn can_precede(&self, verb_a: &str, verb_b: &str, current_state: &StateSet) -> bool {
        // Check if temporal edge exists
        if let Some(edge) = self.temporal_edges.get(&(verb_a.to_string(), verb_b.to_string())) {
            match edge.relation {
                TemporalRelation::Precedes
                | TemporalRelation::Meets
                | TemporalRelation::Enables => true,
                TemporalRelation::Disables => false,
                _ => true, // Other relations may allow precedence
            }
        } else {
            // No explicit constraint - check if B's preconditions are satisfied after A
            if let Some(node_a) = self.inference_graph.get_verb(&verb_a.to_string()) {
                if let Some(node_b) = self.inference_graph.get_verb(&verb_b.to_string()) {
                    // Apply A's effects and check if B's requirements are met
                    let after_a = current_state.apply(&node_a.final_subject_states);
                    match after_a {
                        Ok(new_state) => new_state.satisfies(&node_b.required_subject_states),
                        Err(_) => false,
                    }
                } else {
                    false
                }
            } else {
                false
            }
        }
    }

    /// Find valid next verbs after a given verb
    pub fn find_successors(&self, verb: &str, current_state: &StateSet) -> Vec<&VerbNode> {
        // Get all verbs that can follow
        let all_verbs: Vec<_> = (0..self.inference_graph.verb_count())
            .filter_map(|i| {
                let verb_id = format!("verb_{}", i); // This is a placeholder
                self.inference_graph.get_verb(&verb_id)
            })
            .collect();

        all_verbs
            .into_iter()
            .filter(|v| self.can_precede(verb, &v.id, current_state))
            .collect()
    }

    /// Plan a sequence of verbs to reach goal state
    pub fn plan_to_goal(
        &self,
        start_state: &StateSet,
        goal_state: &StateSet,
        available_verbs: &[String],
        max_depth: usize,
    ) -> Result<Vec<String>> {
        // BFS to find shortest path
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut parent: HashMap<String, (String, StateSet)> = HashMap::new();

        queue.push_back(("start".to_string(), start_state.clone(), 0));
        visited.insert("start".to_string());

        while let Some((current_verb, current_state, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            // Check if goal is satisfied
            if current_state.satisfies(goal_state) {
                // Reconstruct path
                let mut path = vec![];
                let mut v = current_verb;
                while v != "start" {
                    path.push(v.clone());
                    if let Some((p, _)) = parent.get(&v) {
                        v = p.clone();
                    } else {
                        break;
                    }
                }
                path.reverse();
                return Ok(path);
            }

            // Try all available verbs
            for verb_id in available_verbs {
                if visited.contains(verb_id) {
                    continue;
                }

                if let Some(verb) = self.inference_graph.get_verb(verb_id) {
                    // Check if verb is applicable
                    if !current_state.satisfies(&verb.required_subject_states) {
                        continue;
                    }

                    // Check temporal constraints
                    if current_verb != "start" && !self.can_precede(&current_verb, verb_id, &current_state) {
                        continue;
                    }

                    // Apply verb effects
                    if let Ok(new_state) = current_state.apply(&verb.final_subject_states) {
                        visited.insert(verb_id.clone());
                        parent.insert(verb_id.clone(), (current_verb.clone(), current_state.clone()));
                        queue.push_back((verb_id.clone(), new_state, depth + 1));
                    }
                }
            }
        }

        Err(anyhow::anyhow!(
            "No valid plan found to reach goal state within {} steps",
            max_depth
        ))
    }

    /// Detect cycles in temporal constraints (error if found)
    pub fn detect_cycles(&self) -> Result<()> {
        // Build graph for cycle detection
        let mut graph: HashMap<String, Vec<String>> = HashMap::new();

        for ((from, to), edge) in &self.temporal_edges {
            // Only consider strict precedence relations for cycle detection
            if matches!(
                edge.relation,
                TemporalRelation::Precedes | TemporalRelation::Causes
            ) {
                graph.entry(from.clone()).or_default().push(to.clone());
            }
        }

        // DFS to detect cycles
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        fn has_cycle(
            node: &str,
            graph: &HashMap<String, Vec<String>>,
            visited: &mut HashSet<String>,
            rec_stack: &mut HashSet<String>,
        ) -> bool {
            visited.insert(node.to_string());
            rec_stack.insert(node.to_string());

            if let Some(neighbors) = graph.get(node) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        if has_cycle(neighbor, graph, visited, rec_stack) {
                            return true;
                        }
                    } else if rec_stack.contains(neighbor) {
                        return true;
                    }
                }
            }

            rec_stack.remove(node);
            false
        }

        for node in graph.keys() {
            if !visited.contains(node) {
                if has_cycle(node, &graph, &mut visited, &mut rec_stack) {
                    return Err(anyhow::anyhow!(
                        "Temporal constraint cycle detected - invalid constraint set"
                    ));
                }
            }
        }

        Ok(())
    }

    /// Infer implicit temporal relations from explicit ones
    pub fn infer_transitive_relations(&mut self) {
        let mut new_edges: Vec<(String, String, TemporalRelation, f32)> = vec![];

        // For all pairs of edges where A -> B and B -> C, infer A -> C
        for ((from_a, to_a), edge_a) in &self.temporal_edges {
            for ((from_b, to_b), edge_b) in &self.temporal_edges {
                if to_a == from_b && edge_a.relation.is_transitive() && edge_b.relation.is_transitive() {
                    // Infer A -> C
                    let confidence = edge_a.confidence * edge_b.confidence;
                    new_edges.push((
                        from_a.clone(),
                        to_b.clone(),
                        edge_a.relation,
                        confidence,
                    ));
                }
            }
        }

        // Add inferred edges
        for (from, to, relation, confidence) in new_edges {
            self.temporal_edges.insert(
                (from, to),
                TemporalEdge {
                    relation,
                    confidence,
                    conditions: vec!["inferred".to_string()],
                },
            );
        }
    }

    /// Validate sequence of verbs against temporal constraints
    pub fn validate_sequence(&self, sequence: &[String], initial_state: &StateSet) -> Result<()> {
        let mut current_state = initial_state.clone();

        for i in 0..sequence.len() {
            let verb_id = &sequence[i];

            // Check verb exists
            let verb = self
                .inference_graph
                .get_verb(verb_id)
                .ok_or_else(|| anyhow::anyhow!("Unknown verb: {}", verb_id))?;

            // Check preconditions
            if !current_state.satisfies(&verb.required_subject_states) {
                return Err(anyhow::anyhow!(
                    "Preconditions not met for verb {} at position {}",
                    verb_id,
                    i
                ));
            }

            // Check temporal constraints with previous verb
            if i > 0 {
                let prev_verb = &sequence[i - 1];
                if !self.can_precede(prev_verb, verb_id, &current_state) {
                    return Err(anyhow::anyhow!(
                        "Temporal constraint violated: {} cannot precede {} at position {}",
                        prev_verb,
                        verb_id,
                        i
                    ));
                }
            }

            // Apply effects
            current_state = current_state.apply(&verb.final_subject_states)?;
        }

        Ok(())
    }
}

/// Causal reasoning utilities
pub struct CausalReasoner;

impl CausalReasoner {
    /// Check if event A causes event B (direct or indirect)
    pub fn causes(
        graph: &TemporalGraph,
        event_a: &str,
        event_b: &str,
        max_hops: usize,
    ) -> bool {
        // BFS for causal path
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();

        queue.push_back((event_a.to_string(), 0));
        visited.insert(event_a.to_string());

        while let Some((current, hops)) = queue.pop_front() {
            if hops > max_hops {
                continue;
            }

            if current == event_b {
                return true;
            }

            // Find causal successors
            for ((from, to), edge) in &graph.temporal_edges {
                if from == &current
                    && matches!(edge.relation, TemporalRelation::Causes)
                    && !visited.contains(to)
                {
                    visited.insert(to.clone());
                    queue.push_back((to.clone(), hops + 1));
                }
            }
        }

        false
    }

    /// Find common causes of multiple events
    pub fn find_common_causes(graph: &TemporalGraph, events: &[String]) -> Vec<String> {
        if events.is_empty() {
            return vec![];
        }

        // Find all causes for each event
        let mut causes_per_event: Vec<HashSet<String>> = vec![];

        for event in events {
            let mut causes = HashSet::new();
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();

            queue.push_back(event.clone());
            visited.insert(event.clone());

            while let Some(current) = queue.pop_front() {
                for ((from, to), edge) in &graph.temporal_edges {
                    if to == &current && matches!(edge.relation, TemporalRelation::Causes) {
                        causes.insert(from.clone());
                        if !visited.contains(from) {
                            visited.insert(from.clone());
                            queue.push_back(from.clone());
                        }
                    }
                }
            }

            causes_per_event.push(causes);
        }

        // Find intersection
        if causes_per_event.is_empty() {
            return vec![];
        }

        let mut common = causes_per_event[0].clone();
        for causes in &causes_per_event[1..] {
            common = common.intersection(causes).cloned().collect();
        }

        common.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_algebra::StateSet;

    #[test]
    fn test_temporal_relation_properties() {
        assert!(TemporalRelation::Precedes.is_transitive());
        assert!(TemporalRelation::Causes.is_transitive());
        assert!(!TemporalRelation::Overlaps.is_transitive());
    }

    #[test]
    fn test_temporal_graph_cycle_detection() {
        let inference = InferenceGraph::new();
        let mut temporal = TemporalGraph::new(inference);

        // Add non-cyclic edges
        temporal.add_temporal_edge("a", "b", TemporalRelation::Precedes, 1.0);
        temporal.add_temporal_edge("b", "c", TemporalRelation::Precedes, 1.0);

        assert!(temporal.detect_cycles().is_ok());

        // Add cyclic edge
        temporal.add_temporal_edge("c", "a", TemporalRelation::Precedes, 1.0);
        assert!(temporal.detect_cycles().is_err());
    }

    #[test]
    fn test_sequence_validation() {
        let inference = InferenceGraph::new();
        let temporal = TemporalGraph::new(inference);
        let initial_state = StateSet::new();

        // Empty sequence should pass
        assert!(temporal.validate_sequence(&[], &initial_state).is_ok());
    }
}
