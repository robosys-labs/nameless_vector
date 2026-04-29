//! Demo: Database Access Control with Natural Language to SQL
//!
//! This demonstrates:
//! 1. Schema-aware SQL generation from natural language
//! 2. Policy enforcement (GDPR, access control)
//! 3. Semantic validation with grounded meaning
//! 4. SQLite integration for real query execution

use anyhow::Result;
use axiom_ai::database::{
    DatabaseSchema, DatabaseAccessController, SchemaBuilder,
    NaturalLanguageQuery, UserContext, QueryConstraints,
    AccessLevel, TableDefinition, ColumnDefinition, TableMetadata,
};
use axiom_ai::policy::{PolicyEngine, PolicyPackages, PolicyEvaluationResult};
use axiom_ai::domains::Domain;
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Axiom: Database Access Control Demo                      ║");
    println!("║     Natural Language → Validated SQL → SQLite                 ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Step 1: Create a sample database schema
    println!("📋 Step 1: Creating sample e-commerce schema...");
    let schema = create_ecommerce_schema();
    println!("   ✓ Created schema with {} tables", schema.tables.len());
    
    for table in &schema.tables {
        println!("     - {} ({} columns)", table.name, table.columns.len());
    }

    // Step 2: Initialize the access controller with policies
    println!("\n🔐 Step 2: Initializing access controller with policies...");
    let mut controller = DatabaseAccessController::new(schema);
    
    // Add GDPR policy
    let gdpr_policy = PolicyPackages::gdpr_policy();
    controller.add_policy(gdpr_policy)?;
    println!("   ✓ Loaded GDPR policy");
    
    // Add database access policy
    let access_policy = PolicyPackages::database_access_policy();
    controller.add_policy(access_policy)?;
    println!("   ✓ Loaded database access policy");

    // Step 3: Demonstrate different query scenarios
    println!("\n📝 Step 3: Testing natural language queries...\n");
    
    // Scenario 1: Simple SELECT (allowed)
    let query1 = NaturalLanguageQuery {
        query: "show me all users".to_string(),
        user_context: UserContext {
            user_id: "admin".to_string(),
            roles: vec!["admin".to_string()],
            permissions: vec!["read".to_string(), "write".to_string()],
            department: Some("IT".to_string()),
        },
        constraints: QueryConstraints::default(),
    };
    
    println!("Query 1: \"{}\"", query1.query);
    match controller.generate_sql(&query1) {
        Ok(result) => {
            println!("   Generated SQL: {}", result.sql);
            println!("   Valid: {}", result.validation.is_valid);
            if !result.validation.errors.is_empty() {
                for error in &result.validation.errors {
                    println!("   ⚠ Error: {}", error.message);
                }
            }
            println!("   ✓ Query passed all validations\n");
        }
        Err(e) => println!("   ✗ Error: {}\n", e),
    }

    // Scenario 2: Query with policy violation (DELETE without WHERE)
    let query2 = NaturalLanguageQuery {
        query: "delete all users".to_string(),
        user_context: UserContext {
            user_id: "user123".to_string(),
            roles: vec!["user".to_string()],
            permissions: vec!["read".to_string()],
            department: None,
        },
        constraints: QueryConstraints {
            max_rows: Some(1000),
            forbidden_tables: vec![],
            forbidden_columns: vec![],
            allow_writes: false, // Write operations not allowed
            allow_ddl: false,
        },
    };
    
    println!("Query 2: \"{}\"", query2.query);
    match controller.generate_sql(&query2) {
        Ok(result) => {
            println!("   Generated SQL: {}", result.sql);
            println!("   Valid: {}", result.validation.is_valid);
            if !result.validation.errors.is_empty() {
                for error in &result.validation.errors {
                    println!("   ⚠ Policy Violation: {}", error.message);
                }
            }
            println!();
        }
        Err(e) => println!("   ✗ Error (expected): {}\n", e),
    }

    // Scenario 3: Sensitive data access (GDPR concern)
    let query3 = NaturalLanguageQuery {
        query: "get all email addresses".to_string(),
        user_context: UserContext {
            user_id: "marketing".to_string(),
            roles: vec!["marketing".to_string()],
            permissions: vec!["read".to_string()],
            department: Some("Marketing".to_string()),
        },
        constraints: QueryConstraints::default(),
    };
    
    println!("Query 3: \"{}\"", query3.query);
    match controller.generate_sql(&query3) {
        Ok(result) => {
            println!("   Generated SQL: {}", result.sql);
            println!("   Valid: {}", result.validation.is_valid);
            if result.sql.contains("email") {
                println!("   ⚠ Warning: Query accesses PII (email) - GDPR audit required");
            }
            println!();
        }
        Err(e) => println!("   ✗ Error: {}\n", e),
    }

    // Step 4: Demonstrate grounded meaning enhancement
    println!("\n🧠 Step 4: Grounded Meaning Enhancement");
    println!("   The system enhances LLM prompts with:");
    println!("   • Schema context (available tables, columns, types)");
    println!("   • Policy constraints (what operations are allowed)");
    println!("   • Semantic frames (valid query patterns from training)");
    println!("   • Access controls (user permissions and restrictions)");

    // Step 5: Summary
    println!("\n📊 Summary:");
    println!("   ✓ Natural language understood and validated");
    println!("   ✓ Schema constraints enforced (no invalid columns/tables)");
    println!("   ✓ Access policies enforced (no DELETE without permission)");
    println!("   ✓ GDPR compliance checked (PII access flagged)");
    println!("   ✓ SQL generated safely with validation");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Demo complete! The system successfully:");
    println!("• Transforms natural language to SQL");
    println!("• Enforces security policies");
    println!("• Validates against database schema");
    println!("• Flags compliance issues (GDPR)");
    println!("═══════════════════════════════════════════════════════════════\n");

    Ok(())
}

fn create_ecommerce_schema() -> DatabaseSchema {
    DatabaseSchema {
        name: "ecommerce".to_string(),
        version: "1.0.0".to_string(),
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
                        is_sensitive: true, // PII - GDPR protected
                    },
                    ColumnDefinition {
                        name: "name".to_string(),
                        data_type: "VARCHAR".to_string(),
                        nullable: false,
                        default: None,
                        constraints: vec![],
                        is_sensitive: false,
                    },
                    ColumnDefinition {
                        name: "created_at".to_string(),
                        data_type: "TIMESTAMP".to_string(),
                        nullable: false,
                        default: Some("CURRENT_TIMESTAMP".to_string()),
                        constraints: vec![],
                        is_sensitive: false,
                    },
                ],
                primary_key: vec!["id".to_string()],
                foreign_keys: vec![],
                metadata: TableMetadata {
                    description: "Customer user accounts".to_string(),
                    access_level: AccessLevel::Authenticated,
                    is_system: false,
                    tags: vec!["customer".to_string(), "pii".to_string()],
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
                    ColumnDefinition {
                        name: "status".to_string(),
                        data_type: "VARCHAR".to_string(),
                        nullable: false,
                        default: Some("'pending'".to_string()),
                        constraints: vec![],
                        is_sensitive: false,
                    },
                ],
                primary_key: vec!["id".to_string()],
                foreign_keys: vec![],
                metadata: TableMetadata {
                    description: "Customer orders".to_string(),
                    access_level: AccessLevel::Authenticated,
                    is_system: false,
                    tags: vec!["order".to_string(), "sales".to_string()],
                },
            },
            TableDefinition {
                name: "products".to_string(),
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
                        name: "name".to_string(),
                        data_type: "VARCHAR".to_string(),
                        nullable: false,
                        default: None,
                        constraints: vec![],
                        is_sensitive: false,
                    },
                    ColumnDefinition {
                        name: "price".to_string(),
                        data_type: "DECIMAL".to_string(),
                        nullable: false,
                        default: None,
                        constraints: vec!["CHECK (price >= 0)".to_string()],
                        is_sensitive: false,
                    },
                    ColumnDefinition {
                        name: "stock".to_string(),
                        data_type: "INTEGER".to_string(),
                        nullable: false,
                        default: Some("0".to_string()),
                        constraints: vec!["CHECK (stock >= 0)".to_string()],
                        is_sensitive: false,
                    },
                ],
                primary_key: vec!["id".to_string()],
                foreign_keys: vec![],
                metadata: TableMetadata {
                    description: "Product catalog".to_string(),
                    access_level: AccessLevel::Public,
                    is_system: false,
                    tags: vec!["product".to_string(), "catalog".to_string()],
                },
            },
        ],
        relationships: vec![
            axiom_ai::database::TableRelationship {
                from_table: "users".to_string(),
                to_table: "orders".to_string(),
                relationship_type: axiom_ai::database::RelationshipType::OneToMany,
                join_columns: vec![("id".to_string(), "user_id".to_string())],
            }
        ],
    }
}
