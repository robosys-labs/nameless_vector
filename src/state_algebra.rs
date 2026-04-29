//! State Algebra Module
//! 
//! Provides formal operations on state sets for deterministic inference.
//! Core to the meaning-based inference cache - enables computing state transitions
//! and validating verb applicability.

use std::collections::HashSet;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// A set of states across multiple dimensions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateSet {
    pub physical: HashSet<String>,
    pub emotional: HashSet<String>,
    pub positional: HashSet<String>,
    pub mental: HashSet<String>,
}

impl Default for StateSet {
    fn default() -> Self {
        Self {
            physical: HashSet::new(),
            emotional: HashSet::new(),
            positional: HashSet::new(),
            mental: HashSet::new(),
        }
    }
}

impl StateSet {
    /// Create a new empty state set
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if this state set satisfies all requirements
    /// All states in requirements must be present in self
    pub fn satisfies(&self, requirements: &StateSet) -> bool {
        requirements.physical.iter().all(|r| self.physical.contains(r)) &&
        requirements.emotional.iter().all(|r| self.emotional.contains(r)) &&
        requirements.positional.iter().all(|r| self.positional.contains(r)) &&
        requirements.mental.iter().all(|r| self.mental.contains(r))
    }

    /// Check if this state set conflicts with another
    /// Returns the first conflict found, or None if compatible
    pub fn conflicts_with(&self, other: &StateSet) -> Option<Conflict> {
        // Check for explicit contradictions in physical states
        for state in &self.physical {
            if let Some(conflict) = Self::check_state_conflict(state, &other.physical, StateDimension::Physical) {
                return Some(conflict);
            }
        }
        
        // Check emotional conflicts
        for state in &self.emotional {
            if let Some(conflict) = Self::check_state_conflict(state, &other.emotional, StateDimension::Emotional) {
                return Some(conflict);
            }
        }
        
        // Check positional conflicts
        for state in &self.positional {
            if let Some(conflict) = Self::check_state_conflict(state, &other.positional, StateDimension::Positional) {
                return Some(conflict);
            }
        }
        
        // Check mental conflicts
        for state in &self.mental {
            if let Some(conflict) = Self::check_state_conflict(state, &other.mental, StateDimension::Mental) {
                return Some(conflict);
            }
        }
        
        None
    }

    /// Check if a specific state conflicts with any states in a set
    fn check_state_conflict(state: &str, other_states: &HashSet<String>, dimension: StateDimension) -> Option<Conflict> {
        // Check for direct antonyms and successor conflicts
        if let Some(antonym) = get_antonym(state) {
            if other_states.contains(&antonym) {
                return Some(Conflict {
                    state_a: state.to_string(),
                    state_b: antonym,
                    dimension,
                    conflict_type: ConflictType::Antonym,
                });
            }
        }
        
        // Check for mutually exclusive states (e.g., "wet" and "dry")
        if let Some(exclusives) = get_mutually_exclusive_states(state) {
            for exclusive in exclusives {
                if other_states.contains(&exclusive) {
                    return Some(Conflict {
                        state_a: state.to_string(),
                        state_b: exclusive,
                        dimension,
                        conflict_type: ConflictType::MutuallyExclusive,
                    });
                }
            }
        }
        
        None
    }

    /// Apply effects to this state set, producing a new state set
    /// Handles state transitions and conflict resolution
    pub fn apply(&self, effects: &StateSet) -> Result<StateSet> {
        let mut result = self.clone();
        
        // Check for conflicts first
        if let Some(conflict) = self.conflicts_with(effects) {
            return Err(anyhow::anyhow!(
                "Cannot apply effects due to conflict: {} vs {} in {:?} dimension",
                conflict.state_a, conflict.state_b, conflict.dimension
            ));
        }
        
        // Apply physical effects with successor state resolution
        for effect in &effects.physical {
            if let Some(predecessor) = get_predecessor_state(effect) {
                result.physical.remove(&predecessor);
            }
            result.physical.insert(effect.clone());
        }
        
        // Apply emotional effects
        for effect in &effects.emotional {
            if let Some(predecessor) = get_predecessor_state(effect) {
                result.emotional.remove(&predecessor);
            }
            result.emotional.insert(effect.clone());
        }
        
        // Apply positional effects
        for effect in &effects.positional {
            if let Some(predecessor) = get_predecessor_state(effect) {
                result.positional.remove(&predecessor);
            }
            result.positional.insert(effect.clone());
        }
        
        // Apply mental effects
        for effect in &effects.mental {
            if let Some(predecessor) = get_predecessor_state(effect) {
                result.mental.remove(&predecessor);
            }
            result.mental.insert(effect.clone());
        }
        
        Ok(result)
    }

    /// Merge two state sets, checking for conflicts
    pub fn merge(&self, other: &StateSet) -> Result<StateSet> {
        if let Some(conflict) = self.conflicts_with(other) {
            return Err(anyhow::anyhow!(
                "Cannot merge state sets due to conflict: {} vs {} in {:?} dimension",
                conflict.state_a, conflict.state_b, conflict.dimension
            ));
        }
        
        let mut result = self.clone();
        result.physical.extend(other.physical.iter().cloned());
        result.emotional.extend(other.emotional.iter().cloned());
        result.positional.extend(other.positional.iter().cloned());
        result.mental.extend(other.mental.iter().cloned());
        
        Ok(result)
    }

    /// Check if state set is empty (no states in any dimension)
    pub fn is_empty(&self) -> bool {
        self.physical.is_empty() && 
        self.emotional.is_empty() && 
        self.positional.is_empty() && 
        self.mental.is_empty()
    }
}

/// State dimensions for conflict reporting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateDimension {
    Physical,
    Emotional,
    Positional,
    Mental,
}

/// Types of state conflicts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictType {
    Antonym,
    MutuallyExclusive,
    CausalInconsistency,
}

/// A detected conflict between two states
#[derive(Debug, Clone)]
pub struct Conflict {
    pub state_a: String,
    pub state_b: String,
    pub dimension: StateDimension,
    pub conflict_type: ConflictType,
}

/// State hierarchy and antonym relationships
/// These define valid state transitions and contradictions

/// Get the antonym (opposite) of a state if known
fn get_antonym(state: &str) -> Option<String> {
    let antonyms: &[(&str, &str)] = &[
        ("hot", "cold"),
        ("wet", "dry"),
        ("active", "inactive"),
        ("awake", "asleep"),
        ("alive", "dead"),
        ("open", "closed"),
        ("present", "absent"),
        ("attached", "detached"),
        ("happy", "sad"),
        ("calm", "angry"),
        ("confident", "fearful"),
        ("near", "far"),
        ("above", "below"),
        ("inside", "outside"),
        ("aware", "unaware"),
        ("focused", "distracted"),
        ("prepared", "unprepared"),
    ];
    
    for (a, b) in antonyms {
        if state.eq_ignore_ascii_case(a) {
            return Some(b.to_string());
        }
        if state.eq_ignore_ascii_case(b) {
            return Some(a.to_string());
        }
    }
    
    None
}

/// Get mutually exclusive states (states that cannot coexist)
fn get_mutually_exclusive_states(state: &str) -> Option<Vec<String>> {
    // Define mutually exclusive groups
    let exclusive_groups: &[&[&str]] = &[
        &["solid", "liquid", "gas", "plasma"],
        &["standing", "sitting", "lying"],
        &["day", "night"],
    ];
    
    for group in exclusive_groups {
        if group.iter().any(|s| s.eq_ignore_ascii_case(state)) {
            // Return all other states in the group as mutually exclusive
            let others: Vec<String> = group
                .iter()
                .filter(|s| !s.eq_ignore_ascii_case(state))
                .map(|s| s.to_string())
                .collect();
            return Some(others);
        }
    }
    
    None
}

/// Get predecessor state in a progression (e.g., "wet" -> "damp" -> "dry")
fn get_predecessor_state(state: &str) -> Option<String> {
    // Define state progressions
    let progressions: &[&[&str]] = &[
        &["soaked", "wet", "damp", "dry"],
        &["freezing", "cold", "cool", "warm", "hot", "boiling"],
        &["asleep", "drowsy", "awake", "alert"],
    ];
    
    for progression in progressions {
        if let Some(pos) = progression.iter().position(|s| s.eq_ignore_ascii_case(state)) {
            if pos > 0 {
                // Return the previous state in the progression
                return Some(progression[pos - 1].to_string());
            }
        }
    }
    
    None
}

/// Verb applicability checker
pub struct VerbApplicabilityChecker;

impl VerbApplicabilityChecker {
    /// Check if a verb can be applied given subject/object states
    pub fn can_apply(
        subject_state: &StateSet,
        object_state: &StateSet,
        required_subject: &StateSet,
        required_object: &StateSet,
    ) -> bool {
        subject_state.satisfies(required_subject) && 
        object_state.satisfies(required_object)
    }

    /// Compute the resulting state after applying a verb
    pub fn apply_verb(
        subject_state: &StateSet,
        object_state: &StateSet,
        final_subject: &StateSet,
        final_object: &StateSet,
    ) -> Result<(StateSet, StateSet)> {
        let new_subject = subject_state.apply(final_subject)?;
        let new_object = object_state.apply(final_object)?;
        
        Ok((new_subject, new_object))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_satisfaction() {
        let current = StateSet {
            physical: ["hot".to_string(), "wet".to_string()].iter().cloned().collect(),
            emotional: HashSet::new(),
            positional: HashSet::new(),
            mental: HashSet::new(),
        };

        let requirements = StateSet {
            physical: ["hot".to_string()].iter().cloned().collect(),
            ..Default::default()
        };

        assert!(current.satisfies(&requirements));
    }

    #[test]
    fn test_state_conflict() {
        let state_a = StateSet {
            physical: ["hot".to_string()].iter().cloned().collect(),
            ..Default::default()
        };

        let state_b = StateSet {
            physical: ["cold".to_string()].iter().cloned().collect(),
            ..Default::default()
        };

        assert!(state_a.conflicts_with(&state_b).is_some());
    }

    #[test]
    fn test_state_apply() {
        let initial = StateSet {
            physical: ["wet".to_string()].iter().cloned().collect(),
            ..Default::default()
        };

        let effects = StateSet {
            physical: ["dry".to_string()].iter().cloned().collect(),
            ..Default::default()
        };

        let result = initial.apply(&effects).unwrap();
        assert!(result.physical.contains("dry"));
        assert!(!result.physical.contains("wet")); // Predecessor removed
    }

    #[test]
    fn test_verb_applicability() {
        let subject = StateSet {
            physical: ["present".to_string(), "able".to_string()].iter().cloned().collect(),
            mental: ["aware".to_string()].iter().cloned().collect(),
            ..Default::default()
        };

        let object = StateSet {
            physical: ["exists".to_string()].iter().cloned().collect(),
            ..Default::default()
        };

        let req_subject = StateSet {
            physical: ["present".to_string()].iter().cloned().collect(),
            mental: ["aware".to_string()].iter().cloned().collect(),
            ..Default::default()
        };

        let req_object = StateSet {
            physical: ["exists".to_string()].iter().cloned().collect(),
            ..Default::default()
        };

        assert!(VerbApplicabilityChecker::can_apply(&subject, &object, &req_subject, &req_object));
    }
}
