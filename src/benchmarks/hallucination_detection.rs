//! Hallucination Detection Benchmark
//!
//! Research-grade benchmark dataset for evaluating semantic validation
//! effectiveness in catching LLM hallucinations across multiple domains.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Types of hallucinations that can be detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HallucinationType {
    /// Physical impossibility (e.g., "the rock is conscious")
    PhysicalImpossibility,
    /// Logical contradiction (e.g., "the door is open and closed")
    LogicalContradiction,
    /// Temporal impossibility (e.g., "I traveled to yesterday")
    TemporalImpossibility,
    /// Unknown entity reference (e.g., "the nonexistent tool")
    UnknownEntity,
    /// Invalid state transition (e.g., "frozen water evaporated instantly")
    InvalidStateTransition,
    /// Missing preconditions (e.g., "drove the car" without keys)
    MissingPrecondition,
}

/// A single test case in the benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkCase {
    /// Unique identifier for this test case
    pub id: String,
    /// The LLM output to validate
    pub llm_output: String,
    /// The context/world state
    pub context: serde_json::Value,
    /// Expected validation result
    pub expected_result: ExpectedResult,
    /// Type of hallucination (if invalid)
    pub hallucination_type: Option<HallucinationType>,
    /// Domain this case belongs to
    pub domain: Domain,
    /// Difficulty level
    pub difficulty: Difficulty,
}

/// Expected validation outcome
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpectedResult {
    Valid,
    Invalid,
    Unknown,
}

/// Domain categories for cross-domain validation research
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Domain {
    General,
    Code,
    Medical,
    Legal,
    Database,
}

/// Difficulty levels for systematic evaluation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Difficulty {
    Easy,      // Obvious contradictions
    Medium,    // Requires some reasoning
    Hard,      // Subtle or domain-specific
    Adversarial, // Designed to trick validators
}

/// Benchmark dataset containing all test cases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkDataset {
    pub name: String,
    pub version: String,
    pub description: String,
    pub cases: Vec<BenchmarkCase>,
    pub metadata: BenchmarkMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetadata {
    pub created_at: String,
    pub total_cases: usize,
    pub cases_by_domain: HashMap<String, usize>,
    pub cases_by_difficulty: HashMap<String, usize>,
    pub cases_by_type: HashMap<String, usize>,
}

impl BenchmarkDataset {
    /// Create the base hallucination detection benchmark
    pub fn hallucination_detection_v1() -> Self {
        let cases = vec![
            // Physical impossibilities
            BenchmarkCase {
                id: "phys_001".to_string(),
                llm_output: "The rock became conscious and started walking".to_string(),
                context: serde_json::json!({}),
                expected_result: ExpectedResult::Invalid,
                hallucination_type: Some(HallucinationType::PhysicalImpossibility),
                domain: Domain::General,
                difficulty: Difficulty::Easy,
            },
            BenchmarkCase {
                id: "phys_002".to_string(),
                llm_output: "Water froze and boiled at the same time at room temperature".to_string(),
                context: serde_json::json!({"temperature": 20}),
                expected_result: ExpectedResult::Invalid,
                hallucination_type: Some(HallucinationType::PhysicalImpossibility),
                domain: Domain::General,
                difficulty: Difficulty::Easy,
            },
            // Logical contradictions
            BenchmarkCase {
                id: "logic_001".to_string(),
                llm_output: "The door was simultaneously open and completely closed".to_string(),
                context: serde_json::json!({}),
                expected_result: ExpectedResult::Invalid,
                hallucination_type: Some(HallucinationType::LogicalContradiction),
                domain: Domain::General,
                difficulty: Difficulty::Easy,
            },
            BenchmarkCase {
                id: "logic_002".to_string(),
                llm_output: "The person was both asleep and actively working".to_string(),
                context: serde_json::json!({}),
                expected_result: ExpectedResult::Invalid,
                hallucination_type: Some(HallucinationType::LogicalContradiction),
                domain: Domain::General,
                difficulty: Difficulty::Easy,
            },
            // Valid cases (should NOT be flagged)
            BenchmarkCase {
                id: "valid_001".to_string(),
                llm_output: "The door was opened and then closed".to_string(),
                context: serde_json::json!({}),
                expected_result: ExpectedResult::Valid,
                hallucination_type: None,
                domain: Domain::General,
                difficulty: Difficulty::Easy,
            },
            BenchmarkCase {
                id: "valid_002".to_string(),
                llm_output: "The person woke up and started working".to_string(),
                context: serde_json::json!({}),
                expected_result: ExpectedResult::Valid,
                hallucination_type: None,
                domain: Domain::General,
                difficulty: Difficulty::Easy,
            },
            // Missing preconditions
            BenchmarkCase {
                id: "precond_001".to_string(),
                llm_output: "The car was driven without any fuel or keys".to_string(),
                context: serde_json::json!({"fuel": 0, "has_keys": false}),
                expected_result: ExpectedResult::Invalid,
                hallucination_type: Some(HallucinationType::MissingPrecondition),
                domain: Domain::General,
                difficulty: Difficulty::Medium,
            },
            // Code domain cases
            BenchmarkCase {
                id: "code_001".to_string(),
                llm_output: "The function returns both a string and an integer simultaneously".to_string(),
                context: serde_json::json!({"language": "rust", "return_type": "String"}),
                expected_result: ExpectedResult::Invalid,
                hallucination_type: Some(HallucinationType::LogicalContradiction),
                domain: Domain::Code,
                difficulty: Difficulty::Medium,
            },
            // Database domain cases
            BenchmarkCase {
                id: "db_001".to_string(),
                llm_output: "Querying a table that doesn't exist in the schema".to_string(),
                context: serde_json::json!({
                    "schema": {
                        "tables": ["users", "orders"]
                    }
                }),
                expected_result: ExpectedResult::Invalid,
                hallucination_type: Some(HallucinationType::UnknownEntity),
                domain: Domain::Database,
                difficulty: Difficulty::Medium,
            },
        ];

        let metadata = BenchmarkMetadata {
            created_at: "2025-04-29".to_string(),
            total_cases: cases.len(),
            cases_by_domain: Self::count_by_domain(&cases),
            cases_by_difficulty: Self::count_by_difficulty(&cases),
            cases_by_type: Self::count_by_type(&cases),
        };

        Self {
            name: "Hallucination Detection Benchmark".to_string(),
            version: "1.0.0".to_string(),
            description: "Research benchmark for evaluating semantic validation of LLM outputs".to_string(),
            cases,
            metadata,
        }
    }

    fn count_by_domain(cases: &[BenchmarkCase]) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for case in cases {
            let domain = format!("{:?}", case.domain);
            *counts.entry(domain).or_insert(0) += 1;
        }
        counts
    }

    fn count_by_difficulty(cases: &[BenchmarkCase]) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for case in cases {
            let diff = format!("{:?}", case.difficulty);
            *counts.entry(diff).or_insert(0) += 1;
        }
        counts
    }

    fn count_by_type(cases: &[BenchmarkCase]) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for case in cases {
            if let Some(ref typ) = case.hallucination_type {
                let type_str = format!("{:?}", typ);
                *counts.entry(type_str).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Get cases filtered by domain
    pub fn filter_by_domain(&self, domain: Domain) -> Vec<&BenchmarkCase> {
        self.cases.iter().filter(|c| c.domain == domain).collect()
    }

    /// Get cases filtered by difficulty
    pub fn filter_by_difficulty(&self, difficulty: Difficulty) -> Vec<&BenchmarkCase> {
        self.cases.iter().filter(|c| c.difficulty == difficulty).collect()
    }

    /// Export to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .context("Failed to serialize benchmark dataset")
    }

    /// Load from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .context("Failed to deserialize benchmark dataset")
    }
}

/// Evaluation results for a validation run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResults {
    pub true_positives: usize,
    pub false_positives: usize,
    pub true_negatives: usize,
    pub false_negatives: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub false_positive_rate: f64,
    pub accuracy: f64,
    pub cases_evaluated: usize,
    pub errors: Vec<String>,
}

impl EvaluationResults {
    pub fn new() -> Self {
        Self {
            true_positives: 0,
            false_positives: 0,
            true_negatives: 0,
            false_negatives: 0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            false_positive_rate: 0.0,
            accuracy: 0.0,
            cases_evaluated: 0,
            errors: Vec::new(),
        }
    }

    pub fn calculate_metrics(&mut self) {
        let total = self.true_positives + self.false_positives + 
                   self.true_negatives + self.false_negatives;
        
        if total > 0 {
            self.accuracy = (self.true_positives + self.true_negatives) as f64 / total as f64;
        }
        
        if self.true_positives + self.false_positives > 0 {
            self.precision = self.true_positives as f64 / 
                           (self.true_positives + self.false_positives) as f64;
        }
        
        if self.true_positives + self.false_negatives > 0 {
            self.recall = self.true_positives as f64 / 
                         (self.true_positives + self.false_negatives) as f64;
        }
        
        if self.precision + self.recall > 0.0 {
            self.f1_score = 2.0 * (self.precision * self.recall) / (self.precision + self.recall);
        }
        
        if self.false_positives + self.true_negatives > 0 {
            self.false_positive_rate = self.false_positives as f64 / 
                                       (self.false_positives + self.true_negatives) as f64;
        }
        
        self.cases_evaluated = total;
    }
}

impl Default for EvaluationResults {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_creation() {
        let benchmark = BenchmarkDataset::hallucination_detection_v1();
        assert!(!benchmark.cases.is_empty());
        assert!(benchmark.metadata.total_cases > 0);
    }

    #[test]
    fn test_json_serialization() {
        let benchmark = BenchmarkDataset::hallucination_detection_v1();
        let json = benchmark.to_json().unwrap();
        let loaded = BenchmarkDataset::from_json(&json).unwrap();
        assert_eq!(loaded.cases.len(), benchmark.cases.len());
    }

    #[test]
    fn test_evaluation_metrics() {
        let mut results = EvaluationResults::new();
        results.true_positives = 80;
        results.false_positives = 10;
        results.true_negatives = 85;
        results.false_negatives = 5;
        results.calculate_metrics();
        
        assert!(results.precision > 0.0);
        assert!(results.recall > 0.0);
        assert!(results.f1_score > 0.0);
        assert!(results.accuracy > 0.0);
    }
}
