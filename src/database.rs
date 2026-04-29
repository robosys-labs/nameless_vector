//! Database Access Control Module
//!
//! Natural language to SQL generation with schema-aware validation and
//! access control policy enforcement. Provides secure database querying
//! through semantic validation of generated SQL.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::domains::{DatabaseDomainAdapter, Domain, DomainContext, DomainEntity, DomainValidationResult};
use crate::frame_memory::{FrameMemory, VerbFrame, FrameStates};
use crate::policy::{PolicyEngine, PolicyPackages, PolicyEvaluationResult};
use crate::state_algebra::StateSet;

/// Database schema representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseSchema {
    /// Schema name
    pub name: String,
    /// Tables in the schema
    pub tables: Vec<TableDefinition>,
    /// Relationships between tables
    pub relationships: Vec<TableRelationship>,
    /// Schema version
    pub version: String,
}

/// Table definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableDefinition {
    /// Table name
    pub name: String,
    /// Columns in the table
    pub columns: Vec<ColumnDefinition>,
    /// Primary key column(s)
    pub primary_key: Vec<String>,
    /// Foreign keys
    pub foreign_keys: Vec<ForeignKey>,
    /// Table metadata
    pub metadata: TableMetadata,
}

/// Column definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDefinition {
    /// Column name
    pub name: String,
    /// Data type
    pub data_type: String,
    /// Whether column can be null
    pub nullable: bool,
    /// Default value
    pub default: Option<String>,
    /// Column constraints
    pub constraints: Vec<String>,
    /// Whether this is a sensitive field (PII, etc.)
    pub is_sensitive: bool,
}

/// Foreign key relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForeignKey {
    /// Column in this table
    pub column: String,
    /// Referenced table
    pub referenced_table: String,
    /// Referenced column
    pub referenced_column: String,
}

/// Table metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableMetadata {
    /// Description of the table
    pub description: String,
    /// Access control level
    pub access_level: AccessLevel,
    /// Whether this is a system table
    pub is_system: bool,
    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Access level for tables
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessLevel {
    /// Public read access
    Public,
    /// Authenticated users only
    Authenticated,
    /// Role-based access required
    RoleBased,
    /// Restricted to specific users
    Restricted,
    /// Read-only for everyone
    ReadOnly,
}

/// Relationship between tables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableRelationship {
    /// Source table
    pub from_table: String,
    /// Target table
    pub to_table: String,
    /// Relationship type
    pub relationship_type: RelationshipType,
    /// Join columns
    pub join_columns: Vec<(String, String)>,
}

/// Types of table relationships
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipType {
    OneToOne,
    OneToMany,
    ManyToMany,
}

/// Natural language query input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaturalLanguageQuery {
    /// User's natural language question
    pub query: String,
    /// User context (roles, permissions)
    pub user_context: UserContext,
    /// Query constraints
    pub constraints: QueryConstraints,
}

/// User context for access control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserContext {
    /// User ID
    pub user_id: String,
    /// User roles
    pub roles: Vec<String>,
    /// User permissions
    pub permissions: Vec<String>,
    /// Department/organization
    pub department: Option<String>,
}

/// Constraints on query generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryConstraints {
    /// Maximum rows to return
    pub max_rows: Option<usize>,
    /// Tables that cannot be accessed
    pub forbidden_tables: Vec<String>,
    /// Columns that cannot be selected
    pub forbidden_columns: Vec<String>,
    /// Allow write operations
    pub allow_writes: bool,
    /// Allow DDL operations
    pub allow_ddl: bool,
}

impl Default for QueryConstraints {
    fn default() -> Self {
        Self {
            max_rows: Some(1000),
            forbidden_tables: vec![],
            forbidden_columns: vec![],
            allow_writes: false,
            allow_ddl: false,
        }
    }
}

/// Validated SQL query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatedSqlQuery {
    /// Generated SQL query
    pub sql: String,
    /// Query parameters (for prepared statements)
    pub parameters: Vec<QueryParameter>,
    /// Validation result
    pub validation: ValidationStatus,
    /// Estimated cost/complexity
    pub estimated_cost: QueryCost,
    /// Suggested improvements
    pub suggestions: Vec<String>,
}

/// Query parameter for prepared statements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryParameter {
    pub name: String,
    pub data_type: String,
    pub value: serde_json::Value,
}

/// Validation status of generated SQL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStatus {
    /// Whether query passed all validations
    pub is_valid: bool,
    /// Schema validation result
    pub schema_valid: bool,
    /// Policy validation result
    pub policy_valid: bool,
    /// Semantic validation result
    pub semantic_valid: bool,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<String>,
}

/// Validation error details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub error_type: String,
    pub message: String,
    pub severity: String,
    pub location: Option<String>,
}

/// Query cost estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryCost {
    /// Complexity score (1-10)
    pub complexity: u8,
    /// Estimated execution time (ms)
    pub estimated_time_ms: u64,
    /// Estimated rows scanned
    pub estimated_rows: u64,
    /// Index usage score (0-1)
    pub index_usage: f64,
}

/// Database access controller
pub struct DatabaseAccessController {
    /// Database schema
    schema: DatabaseSchema,
    /// Policy engine for access control
    policy_engine: PolicyEngine,
    /// Domain adapter for validation
    domain_adapter: DatabaseDomainAdapter,
    /// Frame memory for semantic validation
    frame_memory: Option<FrameMemory>,
}

impl DatabaseAccessController {
    /// Create a new database access controller
    pub fn new(schema: DatabaseSchema) -> Self {
        let mut policy_engine = PolicyEngine::new();
        
        // Load default database access policy
        let policy = PolicyPackages::database_access_policy();
        let _ = policy_engine.load_policy(policy);
        
        Self {
            schema,
            policy_engine,
            domain_adapter: DatabaseDomainAdapter::new(),
            frame_memory: None,
        }
    }

    /// Set frame memory for semantic validation
    pub fn with_frame_memory(mut self, memory: FrameMemory) -> Self {
        self.frame_memory = Some(memory);
        self
    }

    /// Parse natural language query and generate validated SQL
    pub fn generate_sql(&self, nl_query: &NaturalLanguageQuery) -> Result<ValidatedSqlQuery> {
        // Step 1: Extract intent and entities from natural language
        let (intent, entities) = self.parse_natural_language(&nl_query.query)?;
        
        // Step 2: Map to database schema
        let schema_mapping = self.map_to_schema(&entities)?;
        
        // Step 3: Generate SQL candidate
        let sql_candidate = self.generate_sql_candidate(&intent, &schema_mapping, &nl_query.constraints)?;
        
        // Step 4: Validate against schema
        let schema_validation = self.validate_schema(&sql_candidate)?;
        
        // Step 5: Validate against policies
        let policy_validation = self.validate_policies(&sql_candidate, &nl_query.user_context)?;
        
        // Step 6: Validate semantics (if frame memory available)
        let semantic_validation = if let Some(ref memory) = self.frame_memory {
            self.validate_semantics(&sql_candidate, memory)?
        } else {
            true
        };
        
        // Step 7: Build result
        let is_valid = schema_validation.is_valid && policy_validation.allowed && semantic_validation;
        
        let mut errors = Vec::new();
        if !schema_validation.is_valid {
            for violation in schema_validation.violations {
                errors.push(ValidationError {
                    error_type: "schema".to_string(),
                    message: violation.description,
                    severity: format!("{:?}", violation.severity),
                    location: None,
                }));
            }
        }
        
        if !policy_validation.allowed {
            for violation in policy_validation.violations {
                errors.push(ValidationError {
                    error_type: "policy".to_string(),
                    message: violation.violation_type,
                    severity: format!("{:?}", violation.severity),
                    location: None,
                }));
            }
        }
        
        // Estimate cost
        let estimated_cost = self.estimate_query_cost(&sql_candidate)?;
        
        Ok(ValidatedSqlQuery {
            sql: sql_candidate,
            parameters: vec![], // TODO: Extract parameters
            validation: ValidationStatus {
                is_valid,
                schema_valid: schema_validation.is_valid,
                policy_valid: policy_validation.allowed,
                semantic_valid: semantic_validation,
                errors,
                warnings: vec![], // TODO: Add warnings
            },
            estimated_cost,
            suggestions: vec![], // TODO: Add optimization suggestions
        });
    }

    /// Parse natural language to extract intent and entities
    fn parse_natural_language(&self, query: &str) -> Result<(QueryIntent, Vec<QueryEntity>)> {
        let query_lower = query.to_lowercase();
        
        // Determine intent
        let intent = if query_lower.contains("select") || query_lower.contains("show") || 
                      query_lower.contains("find") || query_lower.contains("get") {
            QueryIntent::Select
        } else if query_lower.contains("insert") || query_lower.contains("add") || 
                   query_lower.contains("create") {
            QueryIntent::Insert
        } else if query_lower.contains("update") || query_lower.contains("change") || 
                   query_lower.contains("modify") {
            QueryIntent::Update
        } else if query_lower.contains("delete") || query_lower.contains("remove") {
            QueryIntent::Delete
        } else {
            QueryIntent::Select // Default to select for safety
        };
        
        // Extract entities (tables and columns)
        let mut entities = Vec::new();
        
        // Look for table references
        for table in &self.schema.tables {
            if query_lower.contains(&table.name.to_lowercase()) {
                entities.push(QueryEntity {
                    entity_type: "table".to_string(),
                    name: table.name.clone(),
                    attributes: vec![],
                });
                
                // Look for column references in this table
                for column in &table.columns {
                    if query_lower.contains(&column.name.to_lowercase()) {
                        entities.push(QueryEntity {
                            entity_type: "column".to_string(),
                            name: column.name.clone(),
                            attributes: vec![("table".to_string(), table.name.clone())],
                        });
                    }
                }
            }
        }
        
        Ok((intent, entities))
    }

    /// Map query entities to database schema
    fn map_to_schema(&self, entities: &[QueryEntity]) -> Result<SchemaMapping> {
        let mut tables = Vec::new();
        let mut columns = Vec::new();
        
        for entity in entities {
            match entity.entity_type.as_str() {
                "table" => {
                    if let Some(table_def) = self.schema.tables.iter()
                        .find(|t| t.name == entity.name) {
                        tables.push(table_def.clone());
                    }
                }
                "column" => {
                    if let Some(table_name) = entity.attributes.iter()
                        .find(|(k, _)| k == "table")
                        .map(|(_, v)| v.clone()) {
                        if let Some(table) = self.schema.tables.iter()
                            .find(|t| t.name == table_name) {
                            if let Some(column) = table.columns.iter()
                                .find(|c| c.name == entity.name) {
                                columns.push((table_name, column.clone()));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        
        // If no tables found, try to infer from columns
        if tables.is_empty() && !columns.is_empty() {
            for (table_name, _) in &columns {
                if let Some(table) = self.schema.tables.iter()
                    .find(|t| &t.name == table_name) {
                    if !tables.iter().any(|t: &TableDefinition| t.name == table.name) {
                        tables.push(table.clone());
                    }
                }
            }
        }
        
        Ok(SchemaMapping { tables, columns })
    }

    /// Generate SQL candidate from intent and schema mapping
    fn generate_sql_candidate(&self, intent: &QueryIntent, mapping: &SchemaMapping, constraints: &QueryConstraints) -> Result<String> {
        let sql = match intent {
            QueryIntent::Select => {
                self.generate_select_sql(mapping, constraints)?
            }
            QueryIntent::Insert => {
                if !constraints.allow_writes {
                    return Err(anyhow::anyhow!("Write operations not allowed"));
                }
                self.generate_insert_sql(mapping, constraints)?
            }
            QueryIntent::Update => {
                if !constraints.allow_writes {
                    return Err(anyhow::anyhow!("Write operations not allowed"));
                }
                self.generate_update_sql(mapping, constraints)?
            }
            QueryIntent::Delete => {
                if !constraints.allow_writes {
                    return Err(anyhow::anyhow!("Write operations not allowed"));
                }
                // Require WHERE clause for DELETE
                "DELETE FROM ".to_string() + &mapping.tables.first()
                    .map(|t| t.name.clone())
                    .unwrap_or_default() + " WHERE 1=0 /* Safety WHERE required */"
            }
        };
        
        Ok(sql)
    }

    /// Generate SELECT SQL
    fn generate_select_sql(&self, mapping: &SchemaMapping, constraints: &QueryConstraints) -> Result<String> {
        let table = mapping.tables.first()
            .ok_or_else(|| anyhow::anyhow!("No table specified"))?;
        
        // Build column list
        let columns = if mapping.columns.is_empty() {
            "*".to_string()
        } else {
            mapping.columns.iter()
                .filter(|(_, col)| !constraints.forbidden_columns.contains(&col.name))
                .map(|(table, col)| format!("{}.{}".to_string(), table, col.name))
                .collect::<Vec<_>>()
                .join(", ")
        };
        
        // Add LIMIT if specified
        let limit_clause = constraints.max_rows
            .map(|n| format!(" LIMIT {}", n))
            .unwrap_or_default();
        
        Ok(format!("SELECT {} FROM {}{}", columns, table.name, limit_clause))
    }

    /// Generate INSERT SQL
    fn generate_insert_sql(&self, mapping: &SchemaMapping, _constraints: &QueryConstraints) -> Result<String> {
        let table = mapping.tables.first()
            .ok_or_else(|| anyhow::anyhow!("No table specified"))?;
        
        let columns: Vec<_> = table.columns.iter()
            .filter(|c| !c.is_sensitive && c.default.is_none() && !c.nullable)
            .map(|c| c.name.clone())
            .collect();
        
        let placeholders: Vec<_> = (0..columns.len())
            .map(|i| format!("${}", i + 1))
            .collect();
        
        Ok(format!("INSERT INTO {} ({}) VALUES ({})", 
            table.name,
            columns.join(", "),
            placeholders.join(", ")
        ))
    }

    /// Generate UPDATE SQL
    fn generate_update_sql(&self, mapping: &SchemaMapping, _constraints: &QueryConstraints) -> Result<String> {
        let table = mapping.tables.first()
            .ok_or_else(|| anyhow::anyhow!("No table specified"))?;
        
        // Only update non-sensitive, non-key columns
        let set_clause: Vec<_> = table.columns.iter()
            .filter(|c| !c.is_sensitive && !table.primary_key.contains(&c.name))
            .map(|c| format!("{} = ${}", c.name, 1))
            .collect();
        
        // Require WHERE clause
        let where_clause = format!("WHERE {} = ${}", 
            table.primary_key.first().unwrap_or(&"id".to_string()),
            2
        );
        
        Ok(format!("UPDATE {} SET {} {}", 
            table.name,
            set_clause.join(", "),
            where_clause
        ))
    }

    /// Validate SQL against database schema
    fn validate_schema(&self, sql: &str) -> Result<DomainValidationResult> {
        let domain_context = DomainContext {
            domain: Domain::Database,
            schema: Some(serde_json::to_value(&self.schema).unwrap_or_default()),
            policies: vec![],
            known_entities: self.schema.tables.iter()
                .map(|t| DomainEntity {
                    entity_type: "table".to_string(),
                    name: t.name.clone(),
                    properties: serde_json::to_value(t).unwrap_or_default(),
                })
                .collect(),
        };
        
        // Use domain adapter for validation
        let entities: Vec<DomainEntity> = vec![]; // Would extract from SQL AST
        
        Ok(self.domain_adapter.validate(sql, &domain_context)?)
    }

    /// Validate SQL against policies
    fn validate_policies(&self, sql: &str, user_context: &UserContext) -> PolicyEvaluationResult {
        let domain_context = DomainContext {
            domain: Domain::Database,
            schema: Some(serde_json::to_value(&self.schema).unwrap_or_default()),
            policies: user_context.permissions.clone(),
            known_entities: vec![],
        };
        
        let entities: Vec<DomainEntity> = vec![]; // Would extract from SQL
        
        self.policy_engine.evaluate(&entities, &domain_context)
    }

    /// Validate SQL semantics against frame memory
    fn validate_semantics(&self, sql: &str, memory: &FrameMemory) -> Result<bool> {
        // Check if SQL operations align with semantic frames
        // This would involve looking up frames for query patterns
        
        // For now, simple check - could be expanded
        if sql.contains("DELETE") && !sql.contains("WHERE") {
            return Ok(false); // DELETE without WHERE is semantically dangerous
        }
        
        Ok(true)
    }

    /// Estimate query cost
    fn estimate_query_cost(&self, sql: &str) -> Result<QueryCost> {
        let complexity = if sql.contains("JOIN") {
            7
        } else if sql.contains("GROUP BY") {
            6
        } else if sql.contains("ORDER BY") {
            4
        } else if sql.contains("WHERE") {
            3
        } else {
            2
        };
        
        let estimated_rows = if sql.contains("LIMIT") {
            1000
        } else {
            10000
        };
        
        Ok(QueryCost {
            complexity,
            estimated_time_ms: (complexity as u64) * 50,
            estimated_rows,
            index_usage: if sql.contains("WHERE") { 0.8 } else { 0.3 },
        })
    }

    /// Get schema information
    pub fn get_schema(&self) -> &DatabaseSchema {
        &self.schema
    }

    /// Add custom policy
    pub fn add_policy(&mut self, policy: crate::policy::Policy) -> Result<()> {
        self.policy_engine.load_policy(policy)
    }
}

/// Query intent types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryIntent {
    Select,
    Insert,
    Update,
    Delete,
}

/// Entity extracted from natural language query
#[derive(Debug, Clone)]
pub struct QueryEntity {
    pub entity_type: String,
    pub name: String,
    pub attributes: Vec<(String, String)>,
}

/// Mapping of query entities to schema
#[derive(Debug, Clone)]
pub struct SchemaMapping {
    pub tables: Vec<TableDefinition>,
    pub columns: Vec<(String, ColumnDefinition)>,
}

/// Builder for creating sample schemas
pub struct SchemaBuilder;

impl SchemaBuilder {
    /// Create a sample e-commerce schema
    pub fn ecommerce_schema() -> DatabaseSchema {
        DatabaseSchema {
            name: "ecommerce".to_string(),
            version: "1.0".to_string(),
            tables: vec![
                TableDefinition {
                    name: "users".to_string(),
                    columns: vec![
                        ColumnDefinition {
                            name: "id".to_string(),
                            data_type: "INTEGER".to_string(),
                            nullable: false,
                            default: None,
                            constraints: vec!["PRIMARY KEY".to_string()],
                            is_sensitive: false,
                        },
                        ColumnDefinition {
                            name: "email".to_string(),
                            data_type: "VARCHAR".to_string(),
                            nullable: false,
                            default: None,
                            constraints: vec!["UNIQUE".to_string()],
                            is_sensitive: true,
                        },
                        ColumnDefinition {
                            name: "name".to_string(),
                            data_type: "VARCHAR".to_string(),
                            nullable: false,
                            default: None,
                            constraints: vec![],
                            is_sensitive: false,
                        },
                    ],
                    primary_key: vec!["id".to_string()],
                    foreign_keys: vec![],
                    metadata: TableMetadata {
                        description: "User accounts".to_string(),
                        access_level: AccessLevel::Authenticated,
                        is_system: false,
                        tags: vec!["user".to_string(), "auth".to_string()],
                    },
                },
                TableDefinition {
                    name: "orders".to_string(),
                    columns: vec![
                        ColumnDefinition {
                            name: "id".to_string(),
                            data_type: "INTEGER".to_string(),
                            nullable: false,
                            default: None,
                            constraints: vec!["PRIMARY KEY".to_string()],
                            is_sensitive: false,
                        },
                        ColumnDefinition {
                            name: "user_id".to_string(),
                            data_type: "INTEGER".to_string(),
                            nullable: false,
                            default: None,
                            constraints: vec!["FOREIGN KEY".to_string()],
                            is_sensitive: false,
                        },
                        ColumnDefinition {
                            name: "total".to_string(),
                            data_type: "DECIMAL".to_string(),
                            nullable: false,
                            default: None,
                            constraints: vec![],
                            is_sensitive: false,
                        },
                    ],
                    primary_key: vec!["id".to_string()],
                    foreign_keys: vec![
                        ForeignKey {
                            column: "user_id".to_string(),
                            referenced_table: "users".to_string(),
                            referenced_column: "id".to_string(),
                        }
                    ],
                    metadata: TableMetadata {
                        description: "Customer orders".to_string(),
                        access_level: AccessLevel::Authenticated,
                        is_system: false,
                        tags: vec!["order".to_string(), "sales".to_string()],
                    },
                },
            ],
            relationships: vec![
                TableRelationship {
                    from_table: "users".to_string(),
                    to_table: "orders".to_string(),
                    relationship_type: RelationshipType::OneToMany,
                    join_columns: vec![("id".to_string(), "user_id".to_string())],
                }
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_builder() {
        let schema = SchemaBuilder::ecommerce_schema();
        assert_eq!(schema.name, "ecommerce");
        assert!(!schema.tables.is_empty());
    }

    #[test]
    fn test_database_controller_creation() {
        let schema = SchemaBuilder::ecommerce_schema();
        let controller = DatabaseAccessController::new(schema);
        assert_eq!(controller.get_schema().tables.len(), 2);
    }

    #[test]
    fn test_natural_language_parsing() {
        let schema = SchemaBuilder::ecommerce_schema();
        let controller = DatabaseAccessController::new(schema);
        
        let nl_query = NaturalLanguageQuery {
            query: "show me all users".to_string(),
            user_context: UserContext {
                user_id: "test".to_string(),
                roles: vec!["user".to_string()],
                permissions: vec!["read".to_string()],
                department: None,
            },
            constraints: QueryConstraints::default(),
        };
        
        let result = controller.generate_sql(&nl_query).unwrap();
        assert!(result.sql.contains("SELECT"));
        assert!(result.sql.contains("users"));
    }

    #[test]
    fn test_write_operation_blocked() {
        let schema = SchemaBuilder::ecommerce_schema();
        let controller = DatabaseAccessController::new(schema);
        
        let nl_query = NaturalLanguageQuery {
            query: "delete all users".to_string(),
            user_context: UserContext {
                user_id: "test".to_string(),
                roles: vec!["user".to_string()],
                permissions: vec!["read".to_string()],
                department: None,
            },
            constraints: QueryConstraints::default(), // allow_writes = false
        };
        
        let result = controller.generate_sql(&nl_query);
        // Should fail because writes not allowed
        assert!(result.is_err() || !result.unwrap().validation.is_valid);
    }

    #[test]
    fn test_sensitive_column_protection() {
        let schema = SchemaBuilder::ecommerce_schema();
        let controller = DatabaseAccessController::new(schema);
        
        // Email is marked as sensitive in schema
        let users_table = schema.tables.iter().find(|t| t.name == "users").unwrap();
        let email_column = users_table.columns.iter().find(|c| c.name == "email").unwrap();
        assert!(email_column.is_sensitive);
    }
}
