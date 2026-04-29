//! Policy DSL Module
//!
//! Domain-specific language for defining compliance and security policies
//! that compile to semantic frames for validation. Supports GDPR, content
//! moderation, database access, and custom policy definitions.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::domains::{Domain, DomainContext, DomainEntity};
use crate::frame_abstraction::{AbstractConstraint, AbstractFrame, ConstraintSeverity};
use crate::grounding::ViolationSeverity;

/// A policy definition in the DSL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    /// Policy name/identifier
    pub name: String,
    /// Policy description
    pub description: String,
    /// Policy version
    pub version: String,
    /// Domain this policy applies to
    pub domain: Domain,
    /// Policy rules
    pub rules: Vec<PolicyRule>,
    /// Policy metadata
    pub metadata: PolicyMetadata,
}

/// Metadata for a policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyMetadata {
    /// Author/organization that created the policy
    pub author: String,
    /// Creation date
    pub created_at: String,
    /// Last updated date
    pub updated_at: String,
    /// Compliance standards this policy addresses
    pub compliance_standards: Vec<String>,
    /// Severity level for violations
    pub default_severity: PolicySeverity,
}

/// Severity levels for policy violations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicySeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl From<PolicySeverity> for ViolationSeverity {
    fn from(severity: PolicySeverity) -> Self {
        match severity {
            PolicySeverity::Info => ViolationSeverity::Warning,
            PolicySeverity::Warning => ViolationSeverity::Warning,
            PolicySeverity::Error => ViolationSeverity::Error,
            PolicySeverity::Critical => ViolationSeverity::Critical,
        }
    }
}

/// A single rule within a policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRule {
    /// Rule identifier
    pub id: String,
    /// Rule description
    pub description: String,
    /// Condition that triggers this rule
    pub condition: PolicyCondition,
    /// Action to take when condition is met
    pub action: PolicyAction,
    /// Severity of this specific rule
    pub severity: PolicySeverity,
}

/// Conditions that can trigger policy rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyCondition {
    /// Match specific entity type
    EntityType(String),
    /// Match specific action/verb
    Action(String),
    /// Match specific property value
    Property { key: String, value: String },
    /// Match if context contains specific data
    ContextContains(String),
    /// Match text pattern (regex)
    Pattern(String),
    /// Negation of another condition
    Not(Box<PolicyCondition>),
    /// Logical AND of multiple conditions
    All(Vec<PolicyCondition>),
    /// Logical OR of multiple conditions
    Any(Vec<PolicyCondition>),
}

/// Actions to take when a policy rule is triggered
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyAction {
    /// Allow the action
    Allow,
    /// Deny the action
    Deny { reason: String },
    /// Require additional approval
    RequireApproval { approver: String },
    /// Log for audit
    Audit { message: String },
    /// Transform the output
    Transform { transformation: String },
    /// Add metadata/tag
    Tag { tag: String },
}

/// Policy engine for compiling and enforcing policies
pub struct PolicyEngine {
    /// Loaded policies
    policies: Vec<Policy>,
    /// Compiled abstract frames for validation
    compiled_frames: Vec<AbstractFrame>,
}

impl PolicyEngine {
    /// Create a new policy engine
    pub fn new() -> Self {
        Self {
            policies: Vec::new(),
            compiled_frames: Vec::new(),
        }
    }

    /// Load a policy into the engine
    pub fn load_policy(&mut self, policy: Policy) -> Result<()> {
        // Compile policy to abstract frames
        let frames = self.compile_policy(&policy)?;
        self.compiled_frames.extend(frames);
        self.policies.push(policy);
        Ok(())
    }

    /// Compile a policy to abstract semantic frames
    fn compile_policy(&self, policy: &Policy) -> Result<Vec<AbstractFrame>> {
        let mut frames = Vec::new();

        for rule in &policy.rules {
            let frame = self.compile_rule_to_frame(rule, &policy.domain)?;
            frames.push(frame);
        }

        Ok(frames)
    }

    /// Compile a single policy rule to an abstract frame
    fn compile_rule_to_frame(&self, rule: &PolicyRule, domain: &Domain) -> Result<AbstractFrame> {
        let preconditions = self.compile_condition_to_constraints(&rule.condition)?;

        let effects = match &rule.action {
            PolicyAction::Allow => vec![],
            PolicyAction::Deny { reason } => vec![crate::frame_abstraction::AbstractEffect {
                effect_type: "policy_violation".to_string(),
                target: rule.id.clone(),
                new_state: crate::frame_abstraction::PropertyValue::String(reason.clone()),
                previous_state: None,
            }],
            _ => vec![],
        };

        Ok(AbstractFrame {
            id: format!("policy_{}_{}", rule.id, rule.description.replace(" ", "_")),
            action: self.extract_action_from_condition(&rule.condition),
            subject: crate::frame_abstraction::AbstractEntity {
                entity_type: "policy_target".to_string(),
                name: rule.id.clone(),
                properties: HashMap::new(),
                capabilities: vec![],
            },
            object: None,
            preconditions,
            effects,
            source_domain: *domain,
            original_frame: Some(serde_json::to_value(rule).unwrap_or_default()),
        })
    }

    /// Compile a policy condition to abstract constraints
    fn compile_condition_to_constraints(&self, condition: &PolicyCondition) -> Result<Vec<AbstractConstraint>> {
        let mut constraints = Vec::new();

        match condition {
            PolicyCondition::EntityType(entity_type) => {
                constraints.push(AbstractConstraint {
                    constraint_type: "entity_type".to_string(),
                    target: "subject".to_string(),
                    expected: crate::frame_abstraction::PropertyValue::String(entity_type.clone()),
                    severity: ConstraintSeverity::Error,
                });
            }
            PolicyCondition::Action(action) => {
                constraints.push(AbstractConstraint {
                    constraint_type: "action_type".to_string(),
                    target: "action".to_string(),
                    expected: crate::frame_abstraction::PropertyValue::String(action.clone()),
                    severity: ConstraintSeverity::Error,
                });
            }
            PolicyCondition::Property { key, value } => {
                constraints.push(AbstractConstraint {
                    constraint_type: format!("property_{}", key),
                    target: key.clone(),
                    expected: crate::frame_abstraction::PropertyValue::String(value.clone()),
                    severity: ConstraintSeverity::Error,
                });
            }
            PolicyCondition::All(conditions) => {
                for cond in conditions {
                    let mut sub_constraints = self.compile_condition_to_constraints(cond)?;
                    constraints.append(&mut sub_constraints);
                }
            }
            PolicyCondition::Any(conditions) => {
                // For OR conditions, we create a single constraint with all options
                // This is a simplified approach - full implementation would need disjunctive constraints
                for cond in conditions {
                    let mut sub_constraints = self.compile_condition_to_constraints(cond)?;
                    constraints.append(&mut sub_constraints);
                }
            }
            PolicyCondition::Not(cond) => {
                // Negation requires special handling in the constraint system
                // For now, we compile the inner condition but mark it as negated
                let mut sub_constraints = self.compile_condition_to_constraints(cond)?;
                for constraint in &mut sub_constraints {
                    constraint.constraint_type = format!("NOT_{}", constraint.constraint_type);
                }
                constraints.append(&mut sub_constraints);
            }
            _ => {
                // Other conditions may not directly translate to constraints
                // They might require runtime evaluation
            }
        }

        Ok(constraints)
    }

    /// Extract action name from condition
    fn extract_action_from_condition(&self, condition: &PolicyCondition) -> String {
        match condition {
            PolicyCondition::Action(action) => action.clone(),
            PolicyCondition::All(conditions) => {
                conditions.iter()
                    .find_map(|c| match c {
                        PolicyCondition::Action(a) => Some(a.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| "unknown".to_string())
            }
            _ => "unknown".to_string(),
        }
    }

    /// Evaluate a policy against a domain context
    pub fn evaluate(&self, entities: &[DomainEntity], context: &DomainContext) -> PolicyEvaluationResult {
        let mut violations = Vec::new();
        let mut passed_rules = Vec::new();

        for policy in &self.policies {
            // Check if policy applies to this domain
            if policy.domain != context.domain {
                continue;
            }

            for rule in &policy.rules {
                let matches = self.evaluate_condition(&rule.condition, entities, context);

                if matches {
                    match &rule.action {
                        PolicyAction::Deny { reason } => {
                            violations.push(PolicyViolation {
                                policy_name: policy.name.clone(),
                                rule_id: rule.id.clone(),
                                violation_type: reason.clone(),
                                severity: rule.severity,
                                entities: entities.iter().map(|e| e.name.clone()).collect(),
                            });
                        }
                        PolicyAction::Audit { message } => {
                            // Log audit event
                            passed_rules.push(format!("{}: {}", rule.id, message));
                        }
                        _ => {
                            passed_rules.push(rule.id.clone());
                        }
                    }
                }
            }
        }

        PolicyEvaluationResult {
            allowed: violations.is_empty(),
            violations,
            audit_log: passed_rules,
        }
    }

    /// Evaluate a condition against entities and context
    fn evaluate_condition(&self, condition: &PolicyCondition, entities: &[DomainEntity], context: &DomainContext) -> bool {
        match condition {
            PolicyCondition::EntityType(entity_type) => {
                entities.iter().any(|e| e.entity_type == *entity_type)
            }
            PolicyCondition::Action(action) => {
                // Check if any entity has this action capability
                entities.iter().any(|e| {
                    e.properties.get("actions")
                        .and_then(|v| v.as_str())
                        .map(|s| s.contains(action))
                        .unwrap_or(false)
                })
            }
            PolicyCondition::Property { key, value } => {
                entities.iter().any(|e| {
                    e.properties.get(key)
                        .map(|v| v.to_string().contains(value))
                        .unwrap_or(false)
                })
            }
            PolicyCondition::ContextContains(key) => {
                context.schema.as_ref()
                    .map(|s| s.get(key).is_some())
                    .unwrap_or(false)
            }
            PolicyCondition::Pattern(pattern) => {
                // Simple pattern matching - in production, use regex
                entities.iter().any(|e| e.name.contains(pattern))
            }
            PolicyCondition::Not(cond) => {
                !self.evaluate_condition(cond, entities, context)
            }
            PolicyCondition::All(conditions) => {
                conditions.iter().all(|c| self.evaluate_condition(c, entities, context))
            }
            PolicyCondition::Any(conditions) => {
                conditions.iter().any(|c| self.evaluate_condition(c, entities, context))
            }
        }
    }

    /// Get all loaded policies
    pub fn get_policies(&self) -> &[Policy] {
        &self.policies
    }

    /// Get policies by domain
    pub fn get_policies_by_domain(&self, domain: Domain) -> Vec<&Policy> {
        self.policies.iter().filter(|p| p.domain == domain).collect()
    }
}

impl Default for PolicyEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of policy evaluation
#[derive(Debug, Clone)]
pub struct PolicyEvaluationResult {
    /// Whether the action is allowed
    pub allowed: bool,
    /// Policy violations found
    pub violations: Vec<PolicyViolation>,
    /// Audit log entries
    pub audit_log: Vec<String>,
}

/// A specific policy violation
#[derive(Debug, Clone)]
pub struct PolicyViolation {
    /// Name of the policy that was violated
    pub policy_name: String,
    /// ID of the rule that triggered
    pub rule_id: String,
    /// Type of violation
    pub violation_type: String,
    /// Severity of the violation
    pub severity: PolicySeverity,
    /// Entities involved
    pub entities: Vec<String>,
}

/// Pre-built policy packages
pub struct PolicyPackages;

impl PolicyPackages {
    /// Create GDPR compliance policy
    pub fn gdpr_policy() -> Policy {
        Policy {
            name: "GDPR_Data_Protection".to_string(),
            description: "General Data Protection Regulation compliance policy".to_string(),
            version: "1.0.0".to_string(),
            domain: Domain::Database,
            rules: vec![
                PolicyRule {
                    id: "gdpr_001".to_string(),
                    description: "No PII in unencrypted storage".to_string(),
                    condition: PolicyCondition::All(vec![
                        PolicyCondition::EntityType("table".to_string()),
                        PolicyCondition::Property { 
                            key: "contains_pii".to_string(), 
                            value: "true".to_string() 
                        },
                    ]),
                    action: PolicyAction::RequireApproval { 
                        approver: "dpo".to_string() 
                    },
                    severity: PolicySeverity::Critical,
                },
                PolicyRule {
                    id: "gdpr_002".to_string(),
                    description: "Data retention limits".to_string(),
                    condition: PolicyCondition::EntityType("query".to_string()),
                    action: PolicyAction::Audit { 
                        message: "Data access logged for retention compliance".to_string() 
                    },
                    severity: PolicySeverity::Info,
                },
            ],
            metadata: PolicyMetadata {
                author: "Axiom Framework".to_string(),
                created_at: "2025-04-29".to_string(),
                updated_at: "2025-04-29".to_string(),
                compliance_standards: vec!["GDPR".to_string()],
                default_severity: PolicySeverity::Error,
            },
        }
    }

    /// Create content moderation policy
    pub fn content_moderation_policy() -> Policy {
        Policy {
            name: "Content_Moderation".to_string(),
            description: "Content safety and moderation policy".to_string(),
            version: "1.0.0".to_string(),
            domain: Domain::General,
            rules: vec![
                PolicyRule {
                    id: "mod_001".to_string(),
                    description: "Block harmful content".to_string(),
                    condition: PolicyCondition::Any(vec![
                        PolicyCondition::Pattern("harmful".to_string()),
                        PolicyCondition::Pattern("dangerous".to_string()),
                    ]),
                    action: PolicyAction::Deny { 
                        reason: "Content violates safety policy".to_string() 
                    },
                    severity: PolicySeverity::Critical,
                },
            ],
            metadata: PolicyMetadata {
                author: "Axiom Framework".to_string(),
                created_at: "2025-04-29".to_string(),
                updated_at: "2025-04-29".to_string(),
                compliance_standards: vec!["Safety".to_string()],
                default_severity: PolicySeverity::Error,
            },
        }
    }

    /// Create database access policy
    pub fn database_access_policy() -> Policy {
        Policy {
            name: "Database_Access_Control".to_string(),
            description: "Role-based database access policy".to_string(),
            version: "1.0.0".to_string(),
            domain: Domain::Database,
            rules: vec![
                PolicyRule {
                    id: "db_001".to_string(),
                    description: "No DELETE without WHERE clause".to_string(),
                    condition: PolicyCondition::All(vec![
                        PolicyCondition::Action("DELETE".to_string()),
                        PolicyCondition::Not(Box::new(
                            PolicyCondition::Pattern("WHERE".to_string())
                        )),
                    ]),
                    action: PolicyAction::Deny { 
                        reason: "DELETE operations require WHERE clause".to_string() 
                    },
                    severity: PolicySeverity::Critical,
                },
                PolicyRule {
                    id: "db_002".to_string(),
                    description: "Read-only tables cannot be modified".to_string(),
                    condition: PolicyCondition::All(vec![
                        PolicyCondition::EntityType("table".to_string()),
                        PolicyCondition::Property { 
                            key: "access_level".to_string(), 
                            value: "read_only".to_string() 
                        },
                        PolicyCondition::Any(vec![
                            PolicyCondition::Action("INSERT".to_string()),
                            PolicyCondition::Action("UPDATE".to_string()),
                            PolicyCondition::Action("DELETE".to_string()),
                        ]),
                    ]),
                    action: PolicyAction::Deny { 
                        reason: "Table is read-only".to_string() 
                    },
                    severity: PolicySeverity::Error,
                },
            ],
            metadata: PolicyMetadata {
                author: "Axiom Framework".to_string(),
                created_at: "2025-04-29".to_string(),
                updated_at: "2025-04-29".to_string(),
                compliance_standards: vec!["Database_Security".to_string()],
                default_severity: PolicySeverity::Error,
            },
        }
    }

    /// Get all pre-built policies
    pub fn all_policies() -> Vec<Policy> {
        vec![
            Self::gdpr_policy(),
            Self::content_moderation_policy(),
            Self::database_access_policy(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_engine_creation() {
        let engine = PolicyEngine::new();
        assert!(engine.get_policies().is_empty());
    }

    #[test]
    fn test_gdpr_policy() {
        let policy = PolicyPackages::gdpr_policy();
        assert_eq!(policy.name, "GDPR_Data_Protection");
        assert!(!policy.rules.is_empty());
    }

    #[test]
    fn test_database_access_policy() {
        let policy = PolicyPackages::database_access_policy();
        assert_eq!(policy.name, "Database_Access_Control");
        assert!(!policy.rules.is_empty());
    }

    #[test]
    fn test_policy_evaluation() {
        let mut engine = PolicyEngine::new();
        let policy = PolicyPackages::database_access_policy();
        engine.load_policy(policy).unwrap();

        let context = DomainContext {
            domain: Domain::Database,
            schema: None,
            policies: vec![],
            known_entities: vec![],
        };

        let entities = vec![DomainEntity {
            entity_type: "table".to_string(),
            name: "users".to_string(),
            properties: serde_json::json!({
                "access_level": "read_only"
            }),
        }];

        let result = engine.evaluate(&entities, &context);
        // Should pass since no INSERT/UPDATE/DELETE action
        assert!(result.allowed);
    }

    #[test]
    fn test_policy_compilation() {
        let engine = PolicyEngine::new();
        let policy = PolicyPackages::content_moderation_policy();
        let frames = engine.compile_policy(&policy).unwrap();
        assert!(!frames.is_empty());
    }
}
