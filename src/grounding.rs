//! Semantic Grounding Layer
//!
//! Validates LLM outputs against structured semantic frames.
//! Catches hallucinations, physical impossibilities, and logical contradictions.

use anyhow::{Context, Result};

use crate::frame_memory::{FrameMemory, VerbFrame};
use crate::inference_graph::{InferenceGraph, VerbNode};
use crate::state_algebra::{StateSet, VerbApplicabilityChecker};

/// Result of grounding validation.
#[derive(Debug, Clone, PartialEq)]
pub enum GroundingResult {
    /// Output is fully grounded and valid.
    Valid,
    /// Output violates known physical/logical constraints.
    Invalid { reason: String, severity: ViolationSeverity },
    /// Cannot determine validity (no matching frame found).
    Unknown { explanation: String },
    /// Output needs clarification or has ambiguities.
    NeedsClarification { issues: Vec<String> },
}

/// Severity levels for constraint violations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ViolationSeverity {
    /// Minor inconsistency, may be acceptable.
    Warning,
    /// Significant issue, should be flagged.
    Error,
    /// Critical violation, output should be rejected.
    Critical,
}

/// The grounding layer that validates LLM outputs.
pub struct GroundingLayer<'a> {
    /// Semantic memory for frame retrieval.
    memory: &'a FrameMemory,
    /// Inference graph for relationship checking.
    graph: &'a InferenceGraph,
}

impl<'a> GroundingLayer<'a> {
    /// Create new grounding layer.
    pub fn new(memory: &'a FrameMemory, graph: &'a InferenceGraph) -> Self {
        Self { memory, graph }
    }

    /// Validate an LLM output against semantic constraints.
    ///
    /// # Arguments
    /// * `output` - The text output from an LLM to validate
    /// * `context` - Current world state (subject/object states)
    ///
    /// # Returns
    /// GroundingResult indicating validity and any issues found.
    pub fn validate(&self, output: &str, context: &StateSet) -> Result<GroundingResult> {
        // Step 1: Find the most relevant semantic frame
        let frame_result = self
            .memory
            .find_closest(output)
            .context("Failed to retrieve semantic frame")?;

        let (verb_id, similarity, frame) = match frame_result {
            Some(result) => result,
            None => {
                return Ok(GroundingResult::Unknown {
                    explanation: "No relevant semantic frame found for validation".to_string(),
                })
            }
        };

        // Step 2: Check similarity threshold
        if similarity < 0.5 {
            return Ok(GroundingResult::Unknown {
                explanation: format!(
                    "Best matching frame '{}' has low similarity ({:.2})",
                    verb_id, similarity
                ),
            });
        }

        // Step 3: Convert frame to state sets
        let required_subject: StateSet = (&frame.required_subject_states).into();
        let required_object: StateSet = (&frame.required_object_states).into();
        let final_subject: StateSet = (&frame.final_subject_states).into();
        let final_object: StateSet = (&frame.final_object_states).into();

        // Step 4: Check preconditions (can this action happen in current context?)
        let preconditions_met =
            VerbApplicabilityChecker::can_apply(context, &StateSet::new(), &required_subject, &required_object);

        if !preconditions_met {
            // Check what's missing
            let missing_states = self.identify_missing_states(context, &required_subject);
            return Ok(GroundingResult::Invalid {
                reason: format!(
                    "Action '{}' requires states not present in context: {:?}",
                    verb_id, missing_states
                ),
                severity: ViolationSeverity::Error,
            });
        }

        // Step 5: Check for contradictions in the proposed state change
        match context.apply(&final_subject) {
            Ok(new_state) => {
                // Check if new state creates any contradictions
                if let Some(conflict) = context.conflicts_with(&final_subject) {
                    return Ok(GroundingResult::Invalid {
                        reason: format!(
                            "State transition creates contradiction: {} vs {} in {:?} dimension",
                            conflict.state_a, conflict.state_b, conflict.dimension
                        ),
                        severity: ViolationSeverity::Critical,
                    });
                }

                // Valid transition
                tracing::info!(
                    "Validated action '{}' (similarity: {:.2})",
                    verb_id,
                    similarity
                );
                Ok(GroundingResult::Valid)
            }
            Err(e) => {
                // Cannot apply the state transition
                Ok(GroundingResult::Invalid {
                    reason: format!("Invalid state transition for '{}': {}", verb_id, e),
                    severity: ViolationSeverity::Critical,
                })
            }
        }
    }

    /// Validate with top-k frame matching for ambiguous outputs.
    ///
    /// Useful when the LLM output might match multiple semantic frames.
    pub fn validate_with_alternatives(
        &self,
        output: &str,
        context: &StateSet,
        k: usize,
    ) -> Result<Vec<(String, f32, GroundingResult)>> {
        let frames = self
            .memory
            .find_top_k(output, k)
            .context("Failed to retrieve top-k frames")?;

        let mut results = Vec::new();

        for (verb_id, similarity, frame) in frames {
            // Validate against this specific frame
            let result = self.validate_against_frame(output, context, frame, similarity)?;
            results.push((verb_id, similarity, result));
        }

        Ok(results)
    }

    /// Validate output against a specific frame (internal helper).
    fn validate_against_frame(
        &self,
        _output: &str,
        context: &StateSet,
        frame: &VerbFrame,
        similarity: f32,
    ) -> Result<GroundingResult> {
        let required_subject = (&frame.required_subject_states).into();
        let required_object = (&frame.required_object_states).into();
        let final_subject = (&frame.final_subject_states).into();

        // Check preconditions
        let preconditions_met =
            VerbApplicabilityChecker::can_apply(context, &StateSet::new(), &required_subject, &required_object);

        if !preconditions_met {
            let missing_states = self.identify_missing_states(context, &required_subject);
            return Ok(GroundingResult::Invalid {
                reason: format!("Missing required states: {:?}", missing_states),
                severity: ViolationSeverity::Error,
            });
        }

        // Check for conflicts
        if let Some(conflict) = context.conflicts_with(&final_subject) {
            return Ok(GroundingResult::Invalid {
                reason: format!(
                    "State contradiction: {} vs {}",
                    conflict.state_a, conflict.state_b
                ),
                severity: ViolationSeverity::Critical,
            });
        }

        // Check if transition is valid
        match context.apply(&final_subject) {
            Ok(_) => Ok(GroundingResult::Valid),
            Err(e) => Ok(GroundingResult::Invalid {
                reason: format!("Invalid state transition: {}", e),
                severity: ViolationSeverity::Critical,
            }),
        }
    }

    /// Extract structured intent from natural language.
    ///
    /// Maps free-form text to the closest known semantic frame.
    pub fn extract_intent(&self, text: &str) -> Result<Option<Intent>> {
        let result = self
            .memory
            .find_closest(text)
            .context("Failed to extract intent")?;

        match result {
            Some((verb_id, similarity, frame)) => {
                if similarity >= 0.6 {
                    Ok(Some(Intent {
                        verb: verb_id,
                        applicable_subjects: frame.applicable_subjects.clone(),
                        applicable_objects: frame.applicable_objects.clone(),
                        confidence: similarity,
                    }))
                } else {
                    Ok(None)
                }
            }
            None => Ok(None),
        }
    }

    /// Identify which required states are missing from context.
    fn identify_missing_states(&self, context: &StateSet, required: &StateSet) -> Vec<String> {
        let mut missing = Vec::new();

        for state in &required.physical {
            if !context.physical.contains(state) {
                missing.push(format!("physical:{}", state));
            }
        }
        for state in &required.emotional {
            if !context.emotional.contains(state) {
                missing.push(format!("emotional:{}", state));
            }
        }
        for state in &required.positional {
            if !context.positional.contains(state) {
                missing.push(format!("positional:{}", state));
            }
        }
        for state in &required.mental {
            if !context.mental.contains(state) {
                missing.push(format!("mental:{}", state));
            }
        }

        missing
    }
}

/// Structured intent extracted from natural language.
#[derive(Debug, Clone)]
pub struct Intent {
    pub verb: String,
    pub applicable_subjects: Vec<String>,
    pub applicable_objects: Vec<String>,
    pub confidence: f32,
}

/// Convenience function for quick validation.
pub fn quick_validate<'a>(
    memory: &'a FrameMemory,
    graph: &'a InferenceGraph,
    output: &str,
    context: &StateSet,
) -> Result<GroundingResult> {
    let layer = GroundingLayer::new(memory, graph);
    layer.validate(output, context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_memory::{FrameStates, VerbFrame};

    fn create_test_frame() -> VerbFrame {
        VerbFrame {
            verb: "open".to_string(),
            applicable_subjects: vec!["biological_body".to_string()],
            applicable_objects: vec!["object".to_string()],
            required_subject_states: FrameStates {
                physical: vec!["present".to_string()],
                emotional: vec![],
                positional: vec!["near_object".to_string()],
                mental: vec!["aware".to_string()],
            },
            required_object_states: FrameStates {
                physical: vec!["exists".to_string(), "closed".to_string()],
                emotional: vec![],
                positional: vec!["accessible".to_string()],
                mental: vec![],
            },
            final_subject_states: FrameStates {
                physical: vec![],
                emotional: vec!["satisfied".to_string()],
                positional: vec![],
                mental: vec![],
            },
            final_object_states: FrameStates {
                physical: vec!["open".to_string()],
                emotional: vec![],
                positional: vec![],
                mental: vec![],
            },
        }
    }

    #[test]
    fn test_grounding_result_variants() {
        assert_eq!(GroundingResult::Valid, GroundingResult::Valid);

        let invalid = GroundingResult::Invalid {
            reason: "test".to_string(),
            severity: ViolationSeverity::Error,
        };
        assert!(matches!(invalid, GroundingResult::Invalid { .. }));
    }

    #[test]
    fn test_violation_severity_ordering() {
        assert_ne!(ViolationSeverity::Warning, ViolationSeverity::Error);
        assert_ne!(ViolationSeverity::Error, ViolationSeverity::Critical);
    }
}
