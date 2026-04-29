//! Query Routing Module
//!
//! Routes queries to appropriate processing tier based on complexity and context.
//! Implements the semantic firewall pattern: validate before/after LLM calls.

use anyhow::Result;

use crate::frame_memory::FrameMemory;
use crate::grounding::{GroundingLayer, GroundingResult, ViolationSeverity};
use crate::inference_graph::InferenceGraph;
use crate::state_algebra::StateSet;

/// Routing decision for query handling.
#[derive(Debug, Clone, PartialEq)]
pub enum RoutingDecision {
    /// Handle locally with deterministic lookup (no LLM needed).
    LocalHandle { reason: String },
    /// Route to small on-device model (e.g., Qwen 0.5B) with grounding.
    SmallModel { model: String, grounding_required: bool },
    /// Route to large cloud model (e.g., GPT-4) with pre-validation.
    LargeModel { pre_validation_required: bool },
    /// Reject query due to constraint violations.
    Reject { reason: String },
}

/// Complexity assessment for query routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QueryComplexity {
    /// Simple lookup or state transition.
    Simple,
    /// Multi-step reasoning required.
    Moderate,
    /// Complex reasoning, creative generation, or open-ended.
    Complex,
}

/// The query router that implements intelligent tier selection.
pub struct QueryRouter<'a> {
    /// Grounding layer for validation.
    grounding: GroundingLayer<'a>,
    /// Similarity threshold for local handling.
    local_threshold: f32,
    /// Similarity threshold for small model (below this -> large model).
    small_model_threshold: f32,
}

impl<'a> QueryRouter<'a> {
    /// Create new router with default thresholds.
    pub fn new(grounding: GroundingLayer<'a>) -> Self {
        Self {
            grounding,
            local_threshold: 0.85,
            small_model_threshold: 0.60,
        }
    }

    /// Create router with custom thresholds.
    pub fn with_thresholds(
        grounding: GroundingLayer<'a>,
        local_threshold: f32,
        small_model_threshold: f32,
    ) -> Self {
        Self {
            grounding,
            local_threshold,
            small_model_threshold,
        }
    }

    /// Route a query based on content and context.
    ///
    /// # Arguments
    /// * `query` - The user query text
    /// * `context` - Current world state
    ///
    /// # Returns
    /// RoutingDecision indicating which tier should handle the query.
    pub fn route(&self, query: &str, context: &StateSet) -> Result<RoutingDecision> {
        // Step 1: Try to extract structured intent
        let intent = self.grounding.extract_intent(query)?;

        // Step 2: Assess complexity
        let complexity = self.assess_complexity(query, &intent);

        // Step 3: Pre-validate against constraints
        let precheck = self.grounding.validate(query, context)?;

        // Step 4: Make routing decision
        match (&precheck, complexity) {
            // Critical violations -> reject immediately
            (
                GroundingResult::Invalid {
                    severity: ViolationSeverity::Critical,
                    reason,
                },
                _,
            ) => Ok(RoutingDecision::Reject {
                reason: reason.clone(),
            }),

            // High confidence match + simple query -> handle locally
            (GroundingResult::Valid, QueryComplexity::Simple) => {
                if let Some(ref extracted) = intent {
                    if extracted.confidence >= self.local_threshold {
                        return Ok(RoutingDecision::LocalHandle {
                            reason: format!(
                                "High confidence match ({:.2}) for simple query",
                                extracted.confidence
                            ),
                        });
                    }
                }

                // Still simple but lower confidence -> small model
                Ok(RoutingDecision::SmallModel {
                    model: "qwen2.5-0.5b".to_string(),
                    grounding_required: true,
                })
            }

            // Moderate complexity or non-critical validation issues -> small model with grounding
            (_, QueryComplexity::Moderate) => Ok(RoutingDecision::SmallModel {
                model: "qwen2.5-0.5b".to_string(),
                grounding_required: true,
            }),

            // Complex queries -> large model with pre-validation
            (_, QueryComplexity::Complex) => Ok(RoutingDecision::LargeModel {
                pre_validation_required: true,
            }),

            // Unknown/ambiguous -> small model to disambiguate
            (GroundingResult::Unknown { explanation }, _) => {
                tracing::info!("Unknown intent: {}", explanation);
                Ok(RoutingDecision::SmallModel {
                    model: "qwen2.5-0.5b".to_string(),
                    grounding_required: true,
                })
            }

            // Default: small model with grounding
            _ => Ok(RoutingDecision::SmallModel {
                model: "qwen2.5-0.5b".to_string(),
                grounding_required: true,
            }),
        }
    }

    /// Assess query complexity based on content and intent.
    fn assess_complexity(&self, query: &str, intent: &Option<crate::grounding::Intent>) -> QueryComplexity {
        let query_lower = query.to_lowercase();

        // Check for complexity indicators
        let complex_indicators = [
            "explain",
            "why",
            "compare",
            "analyze",
            "creative",
            "write",
            "generate",
            "imagine",
            "what if",
            "how would",
        ];

        let moderate_indicators = [
            "then",
            "after",
            "before",
            "and then",
            "sequence",
            "plan",
            "steps",
        ];

        // Check complexity indicators
        if complex_indicators.iter().any(|&ind| query_lower.contains(ind)) {
            return QueryComplexity::Complex;
        }

        if moderate_indicators.iter().any(|&ind| query_lower.contains(ind)) {
            return QueryComplexity::Moderate;
        }

        // Check for multiple actions (temporal complexity)
        if let Some(ref extracted) = intent {
            // If we have an intent but low confidence, might need reasoning
            if extracted.confidence < 0.7 {
                return QueryComplexity::Moderate;
            }
        }

        // Check query length (longer often means more complex)
        let word_count = query.split_whitespace().count();
        if word_count > 20 {
            return QueryComplexity::Moderate;
        }

        QueryComplexity::Simple
    }

    /// Post-process LLM output with grounding validation.
    ///
    /// This should be called after receiving output from SmallModel or LargeModel.
    pub fn validate_output(
        &self,
        llm_output: &str,
        context: &StateSet,
    ) -> Result<ValidationOutcome> {
        let result = self.grounding.validate(llm_output, context)?;

        let outcome = match result {
            GroundingResult::Valid => ValidationOutcome::Accept {
                output: llm_output.to_string(),
            },
            GroundingResult::Invalid { reason, severity } => match severity {
                ViolationSeverity::Warning => ValidationOutcome::AcceptWithWarning {
                    output: llm_output.to_string(),
                    warning: reason,
                },
                _ => ValidationOutcome::Reject {
                    reason,
                    suggestion: "The response contained physically or logically impossible content.".to_string(),
                },
            },
            GroundingResult::Unknown { explanation } => ValidationOutcome::AcceptWithCaution {
                output: llm_output.to_string(),
                note: format!("Could not fully validate: {}", explanation),
            },
            GroundingResult::NeedsClarification { issues } => ValidationOutcome::RequestClarification {
                issues,
            },
        };

        Ok(outcome)
    }
}

/// Final outcome after validation.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationOutcome {
    /// Output is valid, accept as-is.
    Accept { output: String },
    /// Accept but include warning to user.
    AcceptWithWarning { output: String, warning: String },
    /// Accept but note limited validation.
    AcceptWithCaution { output: String, note: String },
    /// Reject and explain why.
    Reject { reason: String, suggestion: String },
    /// Request clarification from user.
    RequestClarification { issues: Vec<String> },
}

/// Builder for creating configured routers.
pub struct RouterBuilder<'a> {
    memory: &'a FrameMemory,
    graph: &'a InferenceGraph,
    local_threshold: f32,
    small_model_threshold: f32,
}

impl<'a> RouterBuilder<'a> {
    /// Start building a router.
    pub fn new(memory: &'a FrameMemory, graph: &'a InferenceGraph) -> Self {
        Self {
            memory,
            graph,
            local_threshold: 0.85,
            small_model_threshold: 0.60,
        }
    }

    /// Set local handling threshold.
    pub fn local_threshold(mut self, threshold: f32) -> Self {
        self.local_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set small model threshold.
    pub fn small_model_threshold(mut self, threshold: f32) -> Self {
        self.small_model_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Build the router.
    pub fn build(self) -> QueryRouter<'a> {
        let grounding = GroundingLayer::new(self.memory, self.graph);
        QueryRouter::with_thresholds(grounding, self.local_threshold, self.small_model_threshold)
    }
}

/// Convenience function for quick routing.
pub fn quick_route<'a>(
    memory: &'a FrameMemory,
    graph: &'a InferenceGraph,
    query: &str,
    context: &StateSet,
) -> Result<RoutingDecision> {
    let grounding = GroundingLayer::new(memory, graph);
    let router = QueryRouter::new(grounding);
    router.route(query, context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_memory::FrameMemory;
    use crate::inference_graph::InferenceGraph;

    #[test]
    fn test_routing_decision_variants() {
        let local = RoutingDecision::LocalHandle {
            reason: "test".to_string(),
        };
        assert!(matches!(local, RoutingDecision::LocalHandle { .. }));

        let reject = RoutingDecision::Reject {
            reason: "violation".to_string(),
        };
        assert!(matches!(reject, RoutingDecision::Reject { .. }));
    }

    #[test]
    fn test_complexity_ordering() {
        assert!(QueryComplexity::Simple < QueryComplexity::Moderate);
        assert!(QueryComplexity::Moderate < QueryComplexity::Complex);
    }

    #[test]
    fn test_validation_outcome_variants() {
        let accept = ValidationOutcome::Accept {
            output: "test".to_string(),
        };
        assert!(matches!(accept, ValidationOutcome::Accept { .. }));

        let reject = ValidationOutcome::Reject {
            reason: "error".to_string(),
            suggestion: "fix".to_string(),
        };
        assert!(matches!(reject, ValidationOutcome::Reject { .. }));
    }
}
