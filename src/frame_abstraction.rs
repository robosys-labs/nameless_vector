//! Frame Abstraction Layer
//!
//! Provides cross-domain frame normalization and abstraction for unified
//! semantic validation across different domains (code, medical, legal, database).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::domains::{Domain, DomainFrame, DomainEntity};
use crate::frame_memory::{FrameStates, VerbFrame};
use crate::state_algebra::StateSet;

/// An abstracted semantic frame that can represent concepts from any domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractFrame {
    /// Unique identifier for this frame
    pub id: String,
    /// The action/verb this frame represents
    pub action: String,
    /// The entity performing the action (subject)
    pub subject: AbstractEntity,
    /// The entity being acted upon (object), if any
    pub object: Option<AbstractEntity>,
    /// Required preconditions for this action
    pub preconditions: Vec<AbstractConstraint>,
    /// Effects/outputs of this action
    pub effects: Vec<AbstractEffect>,
    /// Domain this frame originated from
    pub source_domain: Domain,
    /// Original domain-specific frame (for debugging)
    pub original_frame: Option<serde_json::Value>,
}

/// An abstracted entity that can represent any domain concept
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractEntity {
    /// Entity type (e.g., "function", "table", "patient", "contract")
    pub entity_type: String,
    /// Entity name/identifier
    pub name: String,
    /// Entity properties as key-value pairs
    pub properties: HashMap<String, PropertyValue>,
    /// Entity capabilities (what actions it can perform/undergo)
    pub capabilities: Vec<String>,
}

/// Property value types for entity properties
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PropertyValue {
    String(String),
    Number(f64),
    Boolean(bool),
    List(Vec<String>),
    Object(HashMap<String, PropertyValue>),
}

/// Abstract constraint representing preconditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractConstraint {
    /// Constraint type (e.g., "exists", "accessible", "valid_state")
    pub constraint_type: String,
    /// Target of the constraint (what it applies to)
    pub target: String,
    /// Expected value or state
    pub expected: PropertyValue,
    /// Severity if constraint is violated
    pub severity: ConstraintSeverity,
}

/// Severity levels for constraint violations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintSeverity {
    Warning,
    Error,
    Critical,
}

/// Abstract effect representing state changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractEffect {
    /// Effect type (e.g., "state_change", "creation", "deletion")
    pub effect_type: String,
    /// Target of the effect
    pub target: String,
    /// New state or value after effect
    pub new_state: PropertyValue,
    /// Previous state (if applicable)
    pub previous_state: Option<PropertyValue>,
}

/// Frame abstraction engine for normalizing domain-specific frames
pub struct FrameAbstractionEngine;

impl FrameAbstractionEngine {
    /// Create a new abstraction engine
    pub fn new() -> Self {
        Self
    }

    /// Convert a domain-specific frame to an abstract frame
    pub fn abstract_frame(&self, domain_frame: &DomainFrame, domain: Domain) -> Result<AbstractFrame> {
        let id = format!("{}_{}_{}", 
            Self::domain_prefix(domain),
            domain_frame.frame_type,
            domain_frame.subject
        );

        let subject = AbstractEntity {
            entity_type: domain_frame.frame_type.clone(),
            name: domain_frame.subject.clone(),
            properties: HashMap::new(),
            capabilities: vec![domain_frame.action.clone()],
        };

        let preconditions = domain_frame
            .preconditions
            .iter()
            .map(|p| AbstractConstraint {
                constraint_type: "required".to_string(),
                target: domain_frame.subject.clone(),
                expected: PropertyValue::String(p.clone()),
                severity: ConstraintSeverity::Error,
            })
            .collect();

        let effects = domain_frame
            .effects
            .iter()
            .map(|e| AbstractEffect {
                effect_type: "state_change".to_string(),
                target: domain_frame.subject.clone(),
                new_state: PropertyValue::String(e.clone()),
                previous_state: None,
            })
            .collect();

        Ok(AbstractFrame {
            id,
            action: domain_frame.action.clone(),
            subject,
            object: None,
            preconditions,
            effects,
            source_domain: domain,
            original_frame: Some(serde_json::to_value(domain_frame).unwrap_or_default()),
        })
    }

    /// Convert a legacy VerbFrame to an abstract frame
    pub fn abstract_verb_frame(&self, verb_frame: &VerbFrame) -> Result<AbstractFrame> {
        let id = format!("general_{}", verb_frame.verb);

        let subject = AbstractEntity {
            entity_type: "actor".to_string(),
            name: verb_frame.applicable_subjects.join(", "),
            properties: HashMap::new(),
            capabilities: verb_frame.applicable_subjects.clone(),
        };

        let preconditions = self.frame_states_to_constraints(&verb_frame.required_subject_states, "subject");
        let effects = self.frame_states_to_effects(&verb_frame.final_subject_states, "subject");

        Ok(AbstractFrame {
            id,
            action: verb_frame.verb.clone(),
            subject,
            object: Some(AbstractEntity {
                entity_type: "object".to_string(),
                name: verb_frame.applicable_objects.join(", "),
                properties: HashMap::new(),
                capabilities: verb_frame.applicable_objects.clone(),
            }),
            preconditions,
            effects,
            source_domain: Domain::General,
            original_frame: Some(serde_json::to_value(verb_frame).unwrap_or_default()),
        })
    }

    /// Convert a domain entity to an abstract entity
    pub fn abstract_entity(&self, entity: &DomainEntity) -> Result<AbstractEntity> {
        let properties = if let serde_json::Value::Object(map) = &entity.properties {
            map.iter()
                .map(|(k, v)| {
                    let prop_val = match v {
                        serde_json::Value::String(s) => PropertyValue::String(s.clone()),
                        serde_json::Value::Number(n) => PropertyValue::Number(n.as_f64().unwrap_or(0.0)),
                        serde_json::Value::Bool(b) => PropertyValue::Boolean(*b),
                        serde_json::Value::Array(arr) => {
                            PropertyValue::List(arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect())
                        }
                        _ => PropertyValue::String(v.to_string()),
                    };
                    (k.clone(), prop_val)
                })
                .collect()
        } else {
            HashMap::new()
        };

        Ok(AbstractEntity {
            entity_type: entity.entity_type.clone(),
            name: entity.name.clone(),
            properties,
            capabilities: Vec::new(),
        })
    }

    /// Convert abstract frame back to domain frame (if possible)
    pub fn concretize_frame(&self, abstract_frame: &AbstractFrame, target_domain: Domain) -> Result<DomainFrame> {
        // This would contain domain-specific conversion logic
        // For now, provide a basic implementation
        let constraints: Vec<String> = abstract_frame
            .preconditions
            .iter()
            .map(|c| format!("{}: {:?}", c.constraint_type, c.expected))
            .collect();

        let effects: Vec<String> = abstract_frame
            .effects
            .iter()
            .map(|e| format!("{:?}", e.new_state))
            .collect();

        Ok(DomainFrame {
            frame_type: abstract_frame.subject.entity_type.clone(),
            subject: abstract_frame.subject.name.clone(),
            action: abstract_frame.action.clone(),
            constraints,
            preconditions: Vec::new(),
            effects,
        })
    }

    /// Merge two abstract frames (for multi-domain scenarios)
    pub fn merge_frames(&self, frame1: &AbstractFrame, frame2: &AbstractFrame) -> Result<AbstractFrame> {
        let mut merged = frame1.clone();
        
        // Merge preconditions (deduplicate)
        for precond in &frame2.preconditions {
            if !merged.preconditions.iter().any(|p| p.target == precond.target && p.constraint_type == precond.constraint_type) {
                merged.preconditions.push(precond.clone());
            }
        }

        // Merge effects
        merged.effects.extend(frame2.effects.clone());

        // Update ID to reflect merge
        merged.id = format!("{}_merged_{}", frame1.id, frame2.id);

        Ok(merged)
    }

    /// Check if two abstract frames are compatible (can be merged)
    pub fn are_compatible(&self, frame1: &AbstractFrame, frame2: &AbstractFrame) -> bool {
        // Check for conflicting preconditions
        for pre1 in &frame1.preconditions {
            for pre2 in &frame2.preconditions {
                if pre1.target == pre2.target && pre1.constraint_type == pre2.constraint_type {
                    // If same target and type but different expected values, they're conflicting
                    if pre1.expected != pre2.expected {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Convert frame states to abstract constraints
    fn frame_states_to_constraints(&self, states: &FrameStates, target: &str) -> Vec<AbstractConstraint> {
        let mut constraints = Vec::new();

        for state in &states.physical {
            constraints.push(AbstractConstraint {
                constraint_type: "physical_state".to_string(),
                target: target.to_string(),
                expected: PropertyValue::String(state.clone()),
                severity: ConstraintSeverity::Error,
            });
        }

        for state in &states.emotional {
            constraints.push(AbstractConstraint {
                constraint_type: "emotional_state".to_string(),
                target: target.to_string(),
                expected: PropertyValue::String(state.clone()),
                severity: ConstraintSeverity::Warning,
            });
        }

        for state in &states.positional {
            constraints.push(AbstractConstraint {
                constraint_type: "positional_state".to_string(),
                target: target.to_string(),
                expected: PropertyValue::String(state.clone()),
                severity: ConstraintSeverity::Error,
            });
        }

        for state in &states.mental {
            constraints.push(AbstractConstraint {
                constraint_type: "mental_state".to_string(),
                target: target.to_string(),
                expected: PropertyValue::String(state.clone()),
                severity: ConstraintSeverity::Warning,
            });
        }

        constraints
    }

    /// Convert frame states to abstract effects
    fn frame_states_to_effects(&self, states: &FrameStates, target: &str) -> Vec<AbstractEffect> {
        let mut effects = Vec::new();

        for state in &states.physical {
            effects.push(AbstractEffect {
                effect_type: "physical_change".to_string(),
                target: target.to_string(),
                new_state: PropertyValue::String(state.clone()),
                previous_state: None,
            });
        }

        for state in &states.emotional {
            effects.push(AbstractEffect {
                effect_type: "emotional_change".to_string(),
                target: target.to_string(),
                new_state: PropertyValue::String(state.clone()),
                previous_state: None,
            });
        }

        for state in &states.positional {
            effects.push(AbstractEffect {
                effect_type: "positional_change".to_string(),
                target: target.to_string(),
                new_state: PropertyValue::String(state.clone()),
                previous_state: None,
            });
        }

        for state in &states.mental {
            effects.push(AbstractEffect {
                effect_type: "mental_change".to_string(),
                target: target.to_string(),
                new_state: PropertyValue::String(state.clone()),
                previous_state: None,
            });
        }

        effects
    }

    /// Get domain prefix for IDs
    fn domain_prefix(domain: Domain) -> &'static str {
        match domain {
            Domain::General => "gen",
            Domain::Code => "code",
            Domain::Medical => "med",
            Domain::Legal => "leg",
            Domain::Database => "db",
        }
    }
}

impl Default for FrameAbstractionEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Cross-domain frame registry for managing frames from multiple domains
#[derive(Debug, Clone)]
pub struct CrossDomainFrameRegistry {
    frames: HashMap<String, AbstractFrame>,
    domain_indices: HashMap<Domain, Vec<String>>,
}

impl CrossDomainFrameRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            frames: HashMap::new(),
            domain_indices: HashMap::new(),
        }
    }

    /// Register an abstract frame
    pub fn register(&mut self, frame: AbstractFrame) -> Result<()> {
        let domain = frame.source_domain;
        let id = frame.id.clone();
        
        self.frames.insert(id.clone(), frame);
        self.domain_indices
            .entry(domain)
            .or_insert_with(Vec::new)
            .push(id);
        
        Ok(())
    }

    /// Get frame by ID
    pub fn get(&self, id: &str) -> Option<&AbstractFrame> {
        self.frames.get(id)
    }

    /// Get all frames for a domain
    pub fn get_by_domain(&self, domain: Domain) -> Vec<&AbstractFrame> {
        self.domain_indices
            .get(&domain)
            .map(|ids| ids.iter().filter_map(|id| self.frames.get(id)).collect())
            .unwrap_or_default()
    }

    /// Find frames by action type
    pub fn find_by_action(&self, action: &str) -> Vec<&AbstractFrame> {
        self.frames
            .values()
            .filter(|f| f.action == action)
            .collect()
    }

    /// Find compatible frames (can be merged)
    pub fn find_compatible(&self, frame: &AbstractFrame) -> Vec<&AbstractFrame> {
        let engine = FrameAbstractionEngine::new();
        self.frames
            .values()
            .filter(|f| f.id != frame.id && engine.are_compatible(frame, f))
            .collect()
    }

    /// Total number of registered frames
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

impl Default for CrossDomainFrameRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::Domain;
    use crate::frame_memory::FrameStates;

    #[test]
    fn test_abstract_frame_creation() {
        let engine = FrameAbstractionEngine::new();
        
        let domain_frame = DomainFrame {
            frame_type: "function".to_string(),
            subject: "myFunc".to_string(),
            action: "execute".to_string(),
            constraints: vec!["must_exist".to_string()],
            preconditions: vec!["defined".to_string()],
            effects: vec!["returns_value".to_string()],
        };

        let abstracted = engine.abstract_frame(&domain_frame, Domain::Code).unwrap();
        
        assert_eq!(abstracted.action, "execute");
        assert_eq!(abstracted.subject.name, "myFunc");
        assert!(!abstracted.preconditions.is_empty());
    }

    #[test]
    fn test_verb_frame_abstraction() {
        let engine = FrameAbstractionEngine::new();
        
        let verb_frame = VerbFrame {
            verb: "walk".to_string(),
            applicable_subjects: vec!["person".to_string()],
            applicable_objects: vec!["path".to_string()],
            required_subject_states: FrameStates {
                physical: vec!["able".to_string()],
                emotional: vec![],
                positional: vec!["standing".to_string()],
                mental: vec![],
            },
            required_object_states: FrameStates {
                physical: vec!["exists".to_string()],
                emotional: vec![],
                positional: vec![],
                mental: vec![],
            },
            final_subject_states: FrameStates {
                physical: vec!["moving".to_string()],
                emotional: vec![],
                positional: vec![],
                mental: vec![],
            },
            final_object_states: FrameStates {
                physical: vec![],
                emotional: vec![],
                positional: vec![],
                mental: vec![],
            },
        };

        let abstracted = engine.abstract_verb_frame(&verb_frame).unwrap();
        
        assert_eq!(abstracted.action, "walk");
        assert!(!abstracted.preconditions.is_empty());
        assert!(!abstracted.effects.is_empty());
    }

    #[test]
    fn test_frame_registry() {
        let mut registry = CrossDomainFrameRegistry::new();
        
        let frame = AbstractFrame {
            id: "test_1".to_string(),
            action: "test".to_string(),
            subject: AbstractEntity {
                entity_type: "test".to_string(),
                name: "test_entity".to_string(),
                properties: HashMap::new(),
                capabilities: vec![],
            },
            object: None,
            preconditions: vec![],
            effects: vec![],
            source_domain: Domain::General,
            original_frame: None,
        };

        registry.register(frame.clone()).unwrap();
        assert_eq!(registry.len(), 1);
        
        let retrieved = registry.get("test_1").unwrap();
        assert_eq!(retrieved.id, "test_1");
    }
}
