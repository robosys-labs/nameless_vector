//! Benchmarks Module
//!
//! Research-grade benchmarks for evaluating Axiom's semantic validation capabilities.

pub mod hallucination_detection;

pub use hallucination_detection::{
    BenchmarkDataset,
    BenchmarkCase,
    EvaluationResults,
    HallucinationType,
    ExpectedResult,
    Domain,
    Difficulty,
};
