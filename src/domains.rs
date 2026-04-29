//! Domain Adapters Module
//!
//! Provides cross-domain validation capabilities for code, medical, legal,
//! database, and general domains. Each adapter converts domain-specific
//! concepts into semantic frames for validation.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use crate::grounding::ViolationSeverity;

/// Supported validation domains
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Domain {
    General,
    Code,
    Medical,
    Legal,
    Database,
}

/// Domain-specific entity detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainEntity {
    pub entity_type: String,
    pub name: String,
    pub properties: serde_json::Value,
}

/// Domain adapter trait for converting domain concepts to semantic frames
pub trait DomainAdapter: Send + Sync {
    /// Get the domain this adapter handles
    fn domain(&self) -> Domain;
    
    /// Detect entities in domain-specific text
    fn detect_entities(&self, text: &str) -> Result<Vec<DomainEntity>>;
    
    /// Convert domain constraints to semantic frames
    fn to_semantic_frames(&self, entities: &[DomainEntity]) -> Result<Vec<DomainFrame>>;
    
    /// Validate domain-specific output
    fn validate(&self, output: &str, context: &DomainContext) -> Result<DomainValidationResult>;
}

/// A semantic frame for domain validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainFrame {
    pub frame_type: String,
    pub subject: String,
    pub action: String,
    pub constraints: Vec<String>,
    pub preconditions: Vec<String>,
    pub effects: Vec<String>,
}

/// Context for domain validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainContext {
    pub domain: Domain,
    pub schema: Option<serde_json::Value>,
    pub policies: Vec<String>,
    pub known_entities: Vec<DomainEntity>,
}

/// Result of domain validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainValidationResult {
    pub is_valid: bool,
    pub violations: Vec<DomainViolation>,
    pub confidence: f32,
}

/// A domain validation violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainViolation {
    pub violation_type: String,
    pub description: String,
    pub severity: ViolationSeverity,
    pub suggestion: Option<String>,
}

/// Code domain adapter for programming language validation
pub struct CodeDomainAdapter;

impl CodeDomainAdapter {
    pub fn new() -> Self {
        Self
    }
}

impl DomainAdapter for CodeDomainAdapter {
    fn domain(&self) -> Domain {
        Domain::Code
    }
    
    fn detect_entities(&self, text: &str) -> Result<Vec<DomainEntity>> {
        // Detect functions, variables, types, etc.
        let mut entities = Vec::new();
        
        // Simple pattern matching for demonstration
        // In production, this would use proper AST parsing
        for word in text.split_whitespace() {
            if word.ends_with("()") {
                entities.push(DomainEntity {
                    entity_type: "function".to_string(),
                    name: word.trim_end_matches("()").to_string(),
                    properties: serde_json::json!({}),
                });
            }
        }
        
        Ok(entities)
    }
    
    fn to_semantic_frames(&self, entities: &[DomainEntity]) -> Result<Vec<DomainFrame>> {
        let mut frames = Vec::new();
        
        for entity in entities {
            if entity.entity_type == "function" {
                frames.push(DomainFrame {
                    frame_type: "function_call".to_string(),
                    subject: entity.name.clone(),
                    action: "execute".to_string(),
                    constraints: vec!["must_exist".to_string()],
                    preconditions: vec!["function_defined".to_string()],
                    effects: vec!["returns_result".to_string()],
                });
            }
        }
        
        Ok(frames)
    }
    
    fn validate(&self, output: &str, context: &DomainContext) -> Result<DomainValidationResult> {
        let mut violations = Vec::new();
        
        // Check for undefined function references
        let entities = self.detect_entities(output)?;
        for entity in &entities {
            if entity.entity_type == "function" {
                let is_known = context.known_entities.iter()
                    .any(|e| e.name == entity.name && e.entity_type == "function");
                
                if !is_known {
                    violations.push(DomainViolation {
                        violation_type: "undefined_function".to_string(),
                        description: format!("Function '{}' is not defined", entity.name),
                        severity: ViolationSeverity::Error,
                        suggestion: Some(format!("Define function '{}' before calling it", entity.name)),
                    });
                }
            }
        }
        
        Ok(DomainValidationResult {
            is_valid: violations.is_empty(),
            violations,
            confidence: 0.9,
        })
    }
}

/// Database domain adapter for SQL and schema validation
pub struct DatabaseDomainAdapter;

impl DatabaseDomainAdapter {
    pub fn new() -> Self {
        Self
    }
}

impl DomainAdapter for DatabaseDomainAdapter {
    fn domain(&self) -> Domain {
        Domain::Database
    }
    
    fn detect_entities(&self, text: &str) -> Result<Vec<DomainEntity>> {
        let mut entities = Vec::new();
        
        // Detect table and column references in SQL-like text
        // This is simplified - production would use SQL parsing
        for word in text.split_whitespace() {
            if word.starts_with("FROM ") || word.starts_with("from ") {
                let table = word.split_whitespace().nth(1)
                    .unwrap_or("")
                    .trim_end_matches(",")
                    .to_string();
                if !table.is_empty() {
                    entities.push(DomainEntity {
                        entity_type: "table".to_string(),
                        name: table,
                        properties: serde_json::json!({}),
                    });
                }
            }
        }
        
        Ok(entities)
    }
    
    fn to_semantic_frames(&self, entities: &[DomainEntity]) -> Result<Vec<DomainFrame>> {
        let mut frames = Vec::new();
        
        for entity in entities {
            if entity.entity_type == "table" {
                frames.push(DomainFrame {
                    frame_type: "table_access".to_string(),
                    subject: entity.name.clone(),
                    action: "query".to_string(),
                    constraints: vec!["table_exists".to_string()],
                    preconditions: vec!["has_access".to_string()],
                    effects: vec!["returns_data".to_string()],
                });
            }
        }
        
        Ok(frames)
    }
    
    fn validate(&self, output: &str, context: &DomainContext) -> Result<DomainValidationResult> {
        let mut violations = Vec::new();
        
        // Check schema constraints
        if let Some(ref schema) = context.schema {
            let entities = self.detect_entities(output)?;
            
            for entity in &entities {
                if entity.entity_type == "table" {
                    // Check if table exists in schema
                    let tables = schema.get("tables")
                        .and_then(|t| t.as_array())
                        .map(|arr| arr.iter()
                            .filter_map(|v| v.as_str())
                            .collect::<Vec<_>>())
                        .unwrap_or_default();
                    
                    if !tables.contains(&entity.name.as_str()) {
                        violations.push(DomainViolation {
                            violation_type: "unknown_table".to_string(),
                            description: format!("Table '{}' does not exist in schema", entity.name),
                            severity: ViolationSeverity::Error,
                            suggestion: Some(format!("Available tables: {:?}", tables)),
                        });
                    }
                }
            }
        }
        
        // Check access policies
        for policy in &context.policies {
            if policy.contains("no_delete") && output.to_uppercase().contains("DELETE") {
                violations.push(DomainViolation {
                    violation_type: "policy_violation".to_string(),
                    description: "DELETE operations are not allowed by policy".to_string(),
                    severity: ViolationSeverity::Critical,
                    suggestion: Some("Use SELECT or ensure proper authorization".to_string()),
                });
            }
        }
        
        Ok(DomainValidationResult {
            is_valid: violations.is_empty(),
            violations,
            confidence: 0.85,
        })
    }
}

/// Medical domain adapter for healthcare validation
pub struct MedicalDomainAdapter;

impl MedicalDomainAdapter {
    pub fn new() -> Self {
        Self
    }
}

impl DomainAdapter for MedicalDomainAdapter {
    fn domain(&self) -> Domain {
        Domain::Medical
    }
    
    fn detect_entities(&self, text: &str) -> Result<Vec<DomainEntity>> {
        // Simplified medical entity detection
        // Production would use medical NLP models
        let mut entities = Vec::new();
        
        // Common medical terms for demonstration
        let medical_terms = ["patient", "medication", "dosage", "diagnosis", "symptom"];
        for term in &medical_terms {
            if text.to_lowercase().contains(term) {
                entities.push(DomainEntity {
                    entity_type: "medical_term".to_string(),
                    name: term.to_string(),
                    properties: serde_json::json!({}),
                });
            }
        }
        
        Ok(entities)
    }
    
    fn to_semantic_frames(&self, _entities: &[DomainEntity]) -> Result<Vec<DomainFrame>> {
        // Medical-specific frame conversion
        Ok(Vec::new())
    }
    
    fn validate(&self, output: &str, _context: &DomainContext) -> Result<DomainValidationResult> {
        let mut violations = Vec::new();
        
        // Check for dosage format
        if output.contains("mg") {
            // Simple check for dosage patterns
            let has_number_before_mg = output.chars()
                .collect::<Vec<_>>()
                .windows(3)
                .any(|w| w[2] == 'm' && w[1].is_ascii_digit());
            
            if !has_number_before_mg {
                violations.push(DomainViolation {
                    violation_type: "invalid_dosage".to_string(),
                    description: "Dosage appears to be missing numeric value".to_string(),
                    severity: ViolationSeverity::Critical,
                    suggestion: Some("Verify dosage format (e.g., '500 mg')".to_string()),
                });
            }
        }
        
        Ok(DomainValidationResult {
            is_valid: violations.is_empty(),
            violations,
            confidence: 0.8,
        })
    }
}

/// Legal domain adapter for legal document validation
pub struct LegalDomainAdapter;

impl LegalDomainAdapter {
    pub fn new() -> Self {
        Self
    }
}

impl DomainAdapter for LegalDomainAdapter {
    fn domain(&self) -> Domain {
        Domain::Legal
    }
    
    fn detect_entities(&self, text: &str) -> Result<Vec<DomainEntity>> {
        // Simplified legal entity detection
        let mut entities = Vec::new();
        
        let legal_terms = ["contract", "clause", "party", "obligation", "liability"];
        for term in &legal_terms {
            if text.to_lowercase().contains(term) {
                entities.push(DomainEntity {
                    entity_type: "legal_term".to_string(),
                    name: term.to_string(),
                    properties: serde_json::json!({}),
                });
            }
        }
        
        Ok(entities)
    }
    
    fn to_semantic_frames(&self, _entities: &[DomainEntity]) -> Result<Vec<DomainFrame>> {
        Ok(Vec::new())
    }
    
    fn validate(&self, output: &str, _context: &DomainContext) -> Result<DomainValidationResult> {
        let mut violations = Vec::new();
        
        // Check for ambiguous language
        let ambiguous_terms = ["may", "might", "possibly", "perhaps"];
        for term in &ambiguous_terms {
            if output.to_lowercase().contains(term) {
                violations.push(DomainViolation {
                    violation_type: "ambiguous_language".to_string(),
                    description: format!("Potentially ambiguous term '{}' found", term),
                    severity: ViolationSeverity::Warning,
                    suggestion: Some("Consider using more precise language".to_string()),
                });
            }
        }
        
        Ok(DomainValidationResult {
            is_valid: violations.is_empty(),
            violations,
            confidence: 0.75,
        })
    }
}

/// Factory for creating domain adapters
pub struct DomainAdapterFactory;

impl DomainAdapterFactory {
    /// Create an adapter for the specified domain
    pub fn create(domain: Domain) -> Result<Box<dyn DomainAdapter>> {
        match domain {
            Domain::Code => Ok(Box::new(CodeDomainAdapter::new())),
            Domain::Database => Ok(Box::new(DatabaseDomainAdapter::new())),
            Domain::Medical => Ok(Box::new(MedicalDomainAdapter::new())),
            Domain::Legal => Ok(Box::new(LegalDomainAdapter::new())),
            Domain::General => Err(anyhow::anyhow!("General domain does not require specialized adapter")),
        }
    }
    
    /// Get all available domains
    pub fn available_domains() -> Vec<Domain> {
        vec![
            Domain::Code,
            Domain::Database,
            Domain::Medical,
            Domain::Legal,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_domain_adapter() {
        let adapter = CodeDomainAdapter::new();
        assert_eq!(adapter.domain(), Domain::Code);
        
        let entities = adapter.detect_entities("call function()").unwrap();
        assert!(!entities.is_empty());
    }

    #[test]
    fn test_database_domain_adapter() {
        let adapter = DatabaseDomainAdapter::new();
        assert_eq!(adapter.domain(), Domain::Database);
        
        let context = DomainContext {
            domain: Domain::Database,
            schema: Some(serde_json::json!({
                "tables": ["users", "orders"]
            })),
            policies: vec!["no_delete".to_string()],
            known_entities: vec![],
        };
        
        let result = adapter.validate("SELECT * FROM users", &context).unwrap();
        assert!(result.is_valid);
        
        let result = adapter.validate("SELECT * FROM nonexistent", &context).unwrap();
        assert!(!result.is_valid);
    }

    #[test]
    fn test_factory() {
        let adapter = DomainAdapterFactory::create(Domain::Code).unwrap();
        assert_eq!(adapter.domain(), Domain::Code);
    }
}
