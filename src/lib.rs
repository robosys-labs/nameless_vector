//! Axiom - Semantic Grounding Layer for LLMs
//!
//! A meaning-based validation system that grounds LLM outputs in structured
//! semantic frames. Provides query routing, output validation, and
//! deterministic constraint checking for small and large language models.

// Core inference modules
pub mod state_algebra;
pub mod inference_graph;
pub mod edge_inference;
pub mod temporal;
pub mod schema;

// Semantic grounding layer (NEW)
pub mod retrieval;
pub mod frame_memory;
pub mod grounding;
pub mod router;

// Production infrastructure
pub mod observability;
pub mod security;

// Research benchmarks
pub mod benchmarks;

// Cross-domain validation adapters
pub mod domains;

// Frame abstraction for unified cross-domain validation
pub mod frame_abstraction;

// Policy DSL for compliance and access control
pub mod policy;

// Performance optimization - quantization and compression
pub mod quantization;

// Database access control with natural language to SQL
pub mod database;

// Re-export commonly used types
pub use state_algebra::{StateSet, VerbApplicabilityChecker};
pub use inference_graph::{InferenceGraph, VerbNode, EdgeType, InferenceEdge};
pub use edge_inference::{EdgeGenerator, EdgeInferenceConfig, InferredEdge, InferredRelation, build_connected_graph};
pub use temporal::{TemporalGraph, TemporalRelation, CausalReasoner};

// Re-export grounding layer types
pub use retrieval::Embedder;
pub use frame_memory::{FrameMemory, VerbFrame, IndexedFrame};
pub use grounding::{GroundingLayer, GroundingResult, ViolationSeverity, Intent, quick_validate};
pub use router::{QueryRouter, RoutingDecision, ValidationOutcome, RouterBuilder, quick_route};

// Re-export production infrastructure types
pub use observability::{MetricsCollector, RequestContext, Timer};
pub use security::{InputValidator, SecurityMiddleware, ResourceQuotas};

// Re-export benchmark types
pub use benchmarks::{
    BenchmarkDataset, BenchmarkCase, EvaluationResults,
    HallucinationType, ExpectedResult, Domain, Difficulty,
};

// Re-export domain types
pub use domains::{
    DomainAdapter, DomainEntity, DomainFrame, DomainContext,
    DomainValidationResult, DomainViolation,
    CodeDomainAdapter, DatabaseDomainAdapter, MedicalDomainAdapter, LegalDomainAdapter,
    DomainAdapterFactory,
};

// Re-export frame abstraction types
pub use frame_abstraction::{
    AbstractFrame, AbstractEntity, AbstractConstraint, AbstractEffect,
    PropertyValue, ConstraintSeverity,
    FrameAbstractionEngine, CrossDomainFrameRegistry,
};

// Re-export policy types
pub use policy::{
    Policy, PolicyRule, PolicyCondition, PolicyAction,
    PolicySeverity, PolicyMetadata, PolicyEngine,
    PolicyEvaluationResult, PolicyViolation,
    PolicyPackages,
};

// Re-export quantization types
pub use quantization::{
    QuantizedEmbedding, QuantizationFormat, QuantizationParams,
    EmbeddingQuantizer, BloomFilter, QuantizedStateStore,
    CompressionStats,
};

// Re-export database types
pub use database::{
    DatabaseSchema, DatabaseAccessController,
    TableDefinition, ColumnDefinition, TableRelationship,
    NaturalLanguageQuery, ValidatedSqlQuery,
    AccessLevel, QueryConstraints, UserContext,
    SchemaBuilder,
};
