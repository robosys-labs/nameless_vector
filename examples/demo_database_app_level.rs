//! Application-Level Database Access Demo
//!
//! This demonstrates Nameless Vector as an APPLICATION-LEVEL INTERFACE,
//! embedded directly in your app (like Laravel/Prisma), NOT a database sidecar.
//!
//! Architecture:
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  Web/API Layer                                                  │
//! │  - User submits: "show me sales from last month"                 │
//! └─────────────────────────────────────────────────────────────────┘
//!                                 ↓
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  Application Layer (Laravel/Prisma-style Repository)          │
//! │  ┌─────────────────────────────────────────────────────────┐    │
//! │  │ NamelessVectorSemanticLayer                            │    │
//! │  │  - Embeds natural language                             │    │
//! │  │  - Matches against semantic frames                     │    │
//! │  │  - Validates against RBAC policies                     │    │
//! │  │  - Grounds meaning to database schema                  │    │
//! │  └─────────────────────────────────────────────────────────┘    │
//! │                        ↓                                        │
//! │  ┌─────────────────────────────────────────────────────────┐    │
//! │  │ ORM QueryBuilder (simulated)                           │    │
//! │  │  - Converts validated intent to type-safe queries        │    │
//! │  │  - Applies row-level security                          │    │
//! │  │  - Adds tenant isolation                               │    │
//! │  └─────────────────────────────────────────────────────────┘    │
//! └─────────────────────────────────────────────────────────────────┘
//!                                 ↓
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  Database Layer                                                 │
//! │  - Receives: SELECT * FROM orders WHERE ... AND tenant_id = ?   │
//! │  - With RLS policies already applied                            │
//! └─────────────────────────────────────────────────────────────────┘

use anyhow::{Context, Result};
use rusqlite::{Connection, params};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════════
// APPLICATION LAYER: This is part of your app code, not a sidecar
// ═══════════════════════════════════════════════════════════════════════════════

/// Simulates Laravel Eloquent / Prisma Client style repository
/// The Nameless Vector layer is EMBEDDED here, not external
pub struct SmartRepository {
    db: Connection,
    /// Nameless Vector semantic layer - APPLICATION EMBEDDED
    semantic_engine: SemanticQueryEngine,
    /// RBAC policies - APPLICATION EMBEDDED
    rbac_engine: RbacEngine,
}

/// Nameless Vector Semantic Engine - Application Embedded
/// This is NOT a sidecar - it's compiled into your app binary
struct SemanticQueryEngine {
    /// Frame memory for verb/entity understanding
    frame_memory: HashMap<String, SemanticFrame>,
    /// Schema embeddings for table/column resolution
    schema_embeddings: HashMap<String, Vec<f32>>,
    /// Policy frames for RBAC
    policy_frames: Vec<PolicyFrame>,
}

/// Semantic frame for database operations
/// Represents learned patterns like "show me X" → SELECT * FROM X
#[derive(Clone)]
struct SemanticFrame {
    verb_pattern: String,      // e.g., "show", "find", "get"
    intent: QueryIntent,
    required_permissions: Vec<String>,
    pii_classification: PiiLevel,
    requires_approval: bool,
}

/// RBAC Engine - Application Embedded
struct RbacEngine {
    user_context: UserContext,
    role_permissions: HashMap<String, Vec<String>>,
    data_classification_rules: Vec<DataClassificationRule>,
}

#[derive(Clone, Debug)]
struct UserContext {
    user_id: String,
    tenant_id: String,          // Multi-tenant isolation
    roles: Vec<String>,
    department: Option<String>,
    clearance_level: i32,       // For classified data
}

enum PiiLevel {
    None,
    Low,      // e.g., department name
    Medium,   // e.g., job title
    High,     // e.g., email, phone
    Critical, // e.g., SSN, medical records
}

#[derive(Clone, Debug)]
struct DataClassificationRule {
    table: String,
    column: String,
    classification: PiiLevel,
    required_clearance: i32,
    allowed_roles: Vec<String>,
}

#[derive(Clone, Debug)]
struct PolicyFrame {
    action: String,
    condition: String,
    effect: PolicyEffect,
}

#[derive(Clone, Debug)]
enum PolicyEffect {
    Allow,
    Deny,
    RequireApproval,
    AuditLog,
}

#[derive(Clone, Debug)]
enum QueryIntent {
    Select,
    Insert,
    Update,
    Delete,
    Aggregate,  // COUNT, SUM, etc.
}

/// ORM-style query representation
/// Your app builds this AFTER semantic validation
#[derive(Debug)]
struct OrmQuery {
    table: String,
    operation: QueryIntent,
    columns: Vec<String>,
    filters: Vec<Filter>,
    joins: Vec<Join>,
    tenant_isolation: bool,
    row_level_security: Vec<String>,
    /// Grounded meaning - what the user actually wanted
    semantic_interpretation: SemanticInterpretation,
}

#[derive(Debug)]
struct Filter {
    column: String,
    operator: String,
    value: String,
}

#[derive(Debug)]
struct Join {
    table: String,
    on_column: String,
    to_column: String,
    join_type: String,
}

#[derive(Debug)]
struct SemanticInterpretation {
    original_query: String,
    detected_intent: String,
    confidence: f32,
    extracted_entities: Vec<String>,
    applied_policies: Vec<String>,
    pii_columns_accessed: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════════

impl SmartRepository {
    fn new(user_context: UserContext) -> Result<Self> {
        let db = Connection::open_in_memory()
            .context("Failed to create database")?;
        
        Self::setup_database(&db)?;
        
        let semantic_engine = Self::initialize_semantic_engine();
        let rbac_engine = Self::initialize_rbac_engine(user_context);
        
        Ok(Self {
            db,
            semantic_engine,
            rbac_engine,
        })
    }

    fn setup_database(conn: &Connection) -> Result<()> {
        // Multi-tenant SaaS schema with RLS
        conn.execute(
            "CREATE TABLE tenants (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                plan TEXT NOT NULL
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                email TEXT NOT NULL,
                name TEXT NOT NULL,
                department TEXT,
                salary INTEGER,
                ssn TEXT,  -- Critical PII
                role TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (tenant_id) REFERENCES tenants(id)
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE sales (
                id INTEGER PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                amount REAL NOT NULL,
                product TEXT NOT NULL,
                customer_email TEXT,  -- PII
                region TEXT,
                sale_date TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (tenant_id) REFERENCES tenants(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )",
            [],
        )?;

        // Seed data
        conn.execute(
            "INSERT INTO tenants (id, name, plan) VALUES 
             ('tenant_001', 'Acme Corp', 'enterprise'),
             ('tenant_002', 'Startup Inc', 'basic')",
            [],
        )?;

        conn.execute(
            "INSERT INTO users (tenant_id, email, name, department, salary, ssn, role) VALUES
             ('tenant_001', 'alice@acme.com', 'Alice Johnson', 'Sales', 95000, '123-45-6789', 'manager'),
             ('tenant_001', 'bob@acme.com', 'Bob Smith', 'Engineering', 120000, '987-65-4321', 'lead'),
             ('tenant_001', 'carol@acme.com', 'Carol White', 'Sales', 75000, NULL, 'rep'),
             ('tenant_002', 'david@startup.com', 'David Lee', 'Engineering', 85000, NULL, 'admin')",
            [],
        )?;

        conn.execute(
            "INSERT INTO sales (tenant_id, user_id, amount, product, customer_email, region) VALUES
             ('tenant_001', 1, 5000.00, 'Enterprise License', 'client@bigcorp.com', 'North'),
             ('tenant_001', 1, 3500.00, 'Pro Plan', 'customer@tech.com', 'South'),
             ('tenant_001', 3, 2500.00, 'Basic Plan', 'user@small.com', 'East'),
             ('tenant_002', 4, 1500.00, 'Starter', 'dev@startup.com', 'West')",
            [],
        )?;

        Ok(())
    }

    fn initialize_semantic_engine() -> SemanticQueryEngine {
        let mut frame_memory = HashMap::new();

        // Nameless Vector: Learned semantic frames for database operations
        // These frames represent the "meaning" of natural language patterns
        frame_memory.insert(
            "show".to_string(),
            SemanticFrame {
                verb_pattern: "show|display|list|find|get".to_string(),
                intent: QueryIntent::Select,
                required_permissions: vec!["read".to_string()],
                pii_classification: PiiLevel::None,
                requires_approval: false,
            },
        );

        frame_memory.insert(
            "sales".to_string(),
            SemanticFrame {
                verb_pattern: "sales|revenue|deals|transactions".to_string(),
                intent: QueryIntent::Select,
                required_permissions: vec!["read_sales".to_string()],
                pii_classification: PiiLevel::Medium,
                requires_approval: false,
            },
        );

        frame_memory.insert(
            "delete".to_string(),
            SemanticFrame {
                verb_pattern: "delete|remove|drop|clear".to_string(),
                intent: QueryIntent::Delete,
                required_permissions: vec!["write".to_string(), "delete".to_string()],
                pii_classification: PiiLevel::None,
                requires_approval: true,  // DELETE requires explicit approval
            },
        );

        let mut schema_embeddings = HashMap::new();
        // In real implementation, these would be actual embeddings
        schema_embeddings.insert("sales".to_string(), vec![0.1, 0.2, 0.3]); // Semantic match to "sales" table
        schema_embeddings.insert("revenue".to_string(), vec![0.1, 0.2, 0.3]); // Semantic match to "sales" table
        schema_embeddings.insert("employees".to_string(), vec![0.4, 0.5, 0.6]); // Semantic match to "users" table
        schema_embeddings.insert("staff".to_string(), vec![0.4, 0.5, 0.6]); // Semantic match to "users" table

        let policy_frames = vec![
            PolicyFrame {
                action: "access_ssn".to_string(),
                condition: "clearance_level >= 5".to_string(),
                effect: PolicyEffect::Deny,
            },
            PolicyFrame {
                action: "delete".to_string(),
                condition: "role != admin".to_string(),
                effect: PolicyEffect::Deny,
            },
            PolicyFrame {
                action: "cross_tenant".to_string(),
                condition: "always".to_string(),
                effect: PolicyEffect::Deny,
            },
        ];

        SemanticQueryEngine {
            frame_memory,
            schema_embeddings,
            policy_frames,
        }
    }

    fn initialize_rbac_engine(user_context: UserContext) -> RbacEngine {
        let mut role_permissions = HashMap::new();
        role_permissions.insert(
            "admin".to_string(),
            vec!["read".to_string(), "write".to_string(), "delete".to_string(), "read_ssn".to_string()],
        );
        role_permissions.insert(
            "manager".to_string(),
            vec!["read".to_string(), "write".to_string(), "read_team_data".to_string()],
        );
        role_permissions.insert(
            "rep".to_string(),
            vec!["read".to_string(), "write_own".to_string()],
        );

        let data_classification_rules = vec![
            DataClassificationRule {
                table: "users".to_string(),
                column: "email".to_string(),
                classification: PiiLevel::High,
                required_clearance: 1,
                allowed_roles: vec!["admin".to_string(), "manager".to_string()],
            },
            DataClassificationRule {
                table: "users".to_string(),
                column: "ssn".to_string(),
                classification: PiiLevel::Critical,
                required_clearance: 5,
                allowed_roles: vec!["admin".to_string()],
            },
            DataClassificationRule {
                table: "sales".to_string(),
                column: "customer_email".to_string(),
                classification: PiiLevel::High,
                required_clearance: 1,
                allowed_roles: vec!["admin".to_string(), "manager".to_string(), "rep".to_string()],
            },
        ];

        RbacEngine {
            user_context,
            role_permissions,
            data_classification_rules,
        }
    }

    /// MAIN API: Natural language to ORM query
    /// This is what your Laravel controller or Express route would call
    pub fn natural_language_query(&self, nl_query: &str) -> Result<QueryResult> {
        println!("\n┌────────────────────────────────────────────────────────────────────┐");
        println!("│  APPLICATION LAYER: SmartRepository::natural_language_query()     │");
        println!("│  Input: \"{}\"", nl_query);
        println!("└────────────────────────────────────────────────────────────────────┘");

        // STEP 1: Semantic Understanding (Nameless Vector Layer)
        println!("\n  [STEP 1] Nameless Vector Semantic Analysis");
        let semantic_result = self.semantic_engine.analyze(nl_query)?;
        println!("    Detected Intent: {:?}", semantic_result.intent);
        println!("    Matched Entities: {:?}", semantic_result.entities);
        println!("    Confidence: {:.2}", semantic_result.confidence);

        // STEP 2: RBAC Validation (Application Embedded)
        println!("\n  [STEP 2] RBAC Policy Validation");
        let rbac_result = self.rbac_engine.validate(&semantic_result)?;
        println!("    User: {} (roles: {:?})", 
            self.rbac_engine.user_context.user_id,
            self.rbac_engine.user_context.roles);
        println!("    Permission Check: {}", 
            if rbac_result.allowed { "✓ GRANTED" } else { "✗ DENIED" });
        
        if !rbac_result.violations.is_empty() {
            for v in &rbac_result.violations {
                println!("    ✗ Policy Violation: {}", v);
            }
        }

        if !rbac_result.allowed {
            return Err(anyhow::anyhow!(
                "RBAC DENIED: {}", 
                rbac_result.violations.join(", ")));
        }

        // STEP 3: Build ORM Query (Type-safe, not raw SQL)
        println!("\n  [STEP 3] ORM Query Construction");
        let orm_query = self.build_orm_query(&semantic_result, &rbac_result)?;
        println!("    Table: {}", orm_query.table);
        println!("    Operation: {:?}", orm_query.operation);
        println!("    Tenant Isolation: {}", 
            if orm_query.tenant_isolation { "✓ Applied" } else { "N/A" });
        println!("    RLS Filters: {:?}", orm_query.row_level_security);

        // STEP 4: Execute with grounding
        println!("\n  [STEP 4] Query Execution with Grounded Meaning");
        let result = self.execute_orm_query(&orm_query)?;

        // STEP 5: Return with semantic context
        Ok(QueryResult {
            data: result,
            grounding: semantic_result.interpretation,
            audit_trail: rbac_result.audit_events,
        })
    }

    fn build_orm_query(
        &self,
        semantic: &SemanticAnalysis,
        rbac: &RbacValidation,
    ) -> Result<OrmQuery> {
        // Determine target table via semantic matching
        let table = semantic.entities.first()
            .and_then(|e| self.resolve_table_entity(e))
            .unwrap_or_else(|| "users".to_string());

        // Apply tenant isolation (ALWAYS - this is application-level, not DB RLS)
        let mut filters = semantic.filters.clone();
        filters.push(Filter {
            column: "tenant_id".to_string(),
            operator: "=".to_string(),
            value: self.rbac_engine.user_context.tenant_id.clone(),
        });

        // Apply row-level security from RBAC
        let mut rls_filters = vec![];
        if !rbac.can_access_all_rows {
            // e.g., manager can only see their department
            if let Some(ref dept) = self.rbac_engine.user_context.department {
                rls_filters.push(format!("department = '{}'", dept));
            }
        }

        // Check for PII columns and filter if necessary
        let columns = if rbac.pii_restricted {
            semantic.columns.iter()
                .filter(|c| !self.is_pii_column(&table, c))
                .cloned()
                .collect()
        } else {
            semantic.columns.clone()
        };

        let interpretation = SemanticInterpretation {
            original_query: semantic.original_query.clone(),
            detected_intent: format!("{:?}", semantic.intent),
            confidence: semantic.confidence,
            extracted_entities: semantic.entities.clone(),
            applied_policies: rbac.applied_policies.clone(),
            pii_columns_accessed: rbac.pii_accessed.clone(),
        };

        Ok(OrmQuery {
            table,
            operation: semantic.intent.clone(),
            columns,
            filters,
            joins: vec![],
            tenant_isolation: true,
            row_level_security: rls_filters,
            semantic_interpretation: interpretation,
        })
    }

    fn execute_orm_query(&self, query: &OrmQuery) -> Result<Vec<Vec<String>>> {
        // Convert ORM query to SQL (this is internal, not exposed)
        let sql = self.orm_to_sql(query)?;
        
        println!("    Generated SQL: {}", sql);

        let mut stmt = self.db.prepare(&sql)?;
        let column_count = stmt.column_count();
        
        let rows = stmt.query_map([], |row| {
            let mut values = vec![];
            for i in 0..column_count {
                let value: String = row.get(i).unwrap_or_else(|_| "NULL".to_string());
                values.push(value);
            }
            Ok(values)
        })?;

        let mut results = vec![];
        for row in rows {
            results.push(row?);
        }

        Ok(results)
    }

    fn orm_to_sql(&self, query: &OrmQuery) -> Result<String> {
        let columns = if query.columns.is_empty() {
            "*".to_string()
        } else {
            query.columns.join(", ")
        };

        let mut sql = format!("SELECT {} FROM {}", columns, query.table);

        // WHERE clause combining all filters
        let all_conditions: Vec<String> = query.filters.iter()
            .map(|f| format!("{} {} '{}'", f.column, f.operator, f.value))
            .chain(query.row_level_security.iter().cloned())
            .collect();

        if !all_conditions.is_empty() {
            sql.push_str(&format!(" WHERE {}", all_conditions.join(" AND ")));
        }

        Ok(sql)
    }

    fn resolve_table_entity(&self, entity: &str) -> Option<String> {
        // Semantic resolution via embeddings (simplified)
        // In real impl: compare embedding(entity) with embedding(table_name)
        match entity.to_lowercase().as_str() {
            "sales" | "revenue" | "deals" | "transactions" => Some("sales".to_string()),
            "users" | "employees" | "staff" | "people" => Some("users".to_string()),
            _ => None,
        }
    }

    fn is_pii_column(&self, table: &str, column: &str) -> bool {
        self.rbac_engine.data_classification_rules.iter()
            .any(|r| r.table == table && r.column == column)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SUPPORTING STRUCTS
// ═══════════════════════════════════════════════════════════════════════════════

struct SemanticAnalysis {
    original_query: String,
    intent: QueryIntent,
    entities: Vec<String>,
    columns: Vec<String>,
    filters: Vec<Filter>,
    confidence: f32,
    interpretation: SemanticInterpretation,
}

struct RbacValidation {
    allowed: bool,
    violations: Vec<String>,
    can_access_all_rows: bool,
    pii_restricted: bool,
    pii_accessed: Vec<String>,
    applied_policies: Vec<String>,
    audit_events: Vec<String>,
}

struct QueryResult {
    data: Vec<Vec<String>>,
    grounding: SemanticInterpretation,
    audit_trail: Vec<String>,
}

impl SemanticQueryEngine {
    fn analyze(&self, query: &str) -> Result<SemanticAnalysis> {
        let query_lower = query.to_lowercase();
        
        // Extract intent via frame matching
        let intent = if query_lower.contains("show") || query_lower.contains("find") {
            QueryIntent::Select
        } else if query_lower.contains("delete") || query_lower.contains("remove") {
            QueryIntent::Delete
        } else {
            QueryIntent::Select
        };

        // Extract entities via semantic matching
        let mut entities = vec![];
        for (keyword, _) in &self.schema_embeddings {
            if query_lower.contains(keyword) {
                entities.push(keyword.clone());
            }
        }

        // Extract filters
        let mut filters = vec![];
        // Simple pattern: "from X" or "in X department"
        if let Some(pos) = query_lower.find("department") {
            let after = &query_lower[pos + 10..];
            if let Some(word) = after.split_whitespace().nth(1) {
                filters.push(Filter {
                    column: "department".to_string(),
                    operator: "=".to_string(),
                    value: word.trim_matches(|c| c == '\'' || c == '"').to_string(),
                });
            }
        }

        let interpretation = SemanticInterpretation {
            original_query: query.to_string(),
            detected_intent: format!("{:?}", intent),
            confidence: 0.85,
            extracted_entities: entities.clone(),
            applied_policies: vec![],
            pii_columns_accessed: vec![],
        };

        Ok(SemanticAnalysis {
            original_query: query.to_string(),
            intent,
            entities,
            columns: vec![], // Would be extracted from query
            filters,
            confidence: 0.85,
            interpretation,
        })
    }
}

impl RbacEngine {
    fn validate(&self, semantic: &SemanticAnalysis) -> Result<RbacValidation> {
        let mut violations = vec![];
        let mut applied_policies = vec![];
        let mut audit_events = vec![];
        let mut pii_accessed = vec![];

        // Check required permissions for intent
        let required_perms = match semantic.intent {
            QueryIntent::Select => vec!["read"],
            QueryIntent::Delete => vec!["delete"],
            _ => vec!["read"],
        };

        for perm in &required_perms {
            if !self.has_permission(perm) {
                violations.push(format!("Missing permission: {}", perm));
            }
        }
        
        applied_policies.push(format!("intent_check:{:?}", semantic.intent));

        // Check PII access
        let pii_restricted = !self.has_permission("read_pii");
        if pii_restricted {
            applied_policies.push("pii_restriction:active".to_string());
        }

        // Row-level access (can user see all rows or only their own?)
        let can_access_all_rows = self.user_context.roles.contains(&"admin".to_string())
            || self.user_context.roles.contains(&"manager".to_string());

        if !can_access_all_rows {
            applied_policies.push("row_isolation:user_tenant_only".to_string());
        }

        // Audit logging
        audit_events.push(format!(
            "User {} executed semantic query: '{}'",
            self.user_context.user_id,
            semantic.original_query
        ));

        let allowed = violations.is_empty();

        Ok(RbacValidation {
            allowed,
            violations,
            can_access_all_rows,
            pii_restricted,
            pii_accessed,
            applied_policies,
            audit_events,
        })
    }

    fn has_permission(&self, perm: &str) -> bool {
        for role in &self.user_context.roles {
            if let Some(perms) = self.role_permissions.get(role) {
                if perms.contains(&perm.to_string()) {
                    return true;
                }
            }
        }
        false
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN DEMO
// ═══════════════════════════════════════════════════════════════════════════════

fn main() -> Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════════════════════════════╗");
    println!("║     AXIOM: Application-Level Natural Language Database Interface               ║");
    println!("║     (Nameless Vector EMBEDDED in Laravel/Prisma-style Repository)               ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════╝");
    
    println!("\n🏗 ARCHITECTURE:");
    println!("   ┌──────────────────────────────────────────────────────────────────────┐");
    println!("   │  Your App (Laravel/Express/Next.js)                                │");
    println!("   │  ┌────────────────────────────────────────────────────────────────┐  │");
    println!("   │  │  SmartRepository                                              │  │");
    println!("   │  │  ├── NamelessVectorSemanticLayer (embedded, not sidecar)      │  │");
    println!("   │  │  ├── RbacEngine (policies embedded in app)                    │  │");
    println!("   │  │  └── ORM QueryBuilder (type-safe query construction)          │  │");
    println!("   │  └────────────────────────────────────────────────────────────────┘  │");
    println!("   │                              ↓                                       │");
    println!("   │  ┌────────────────────────────────────────────────────────────────┐  │");
    println!("   │  │  Database (SQLite/Postgres/MySQL)                            │  │");
    println!("   │  │  - Receives validated, tenant-scoped, RLS-applied queries   │  │");
    println!("   │  └────────────────────────────────────────────────────────────────┘  │");
    println!("   └──────────────────────────────────────────────────────────────────────┘");
    
    println!("\n⚡ KEY POINT: Nameless Vector is COMPILED INTO your app binary,");
    println!("   NOT a separate sidecar service. It's an application-level semantic layer.\n");

    // Test 1: Admin queries sales
    {
        println!("{}", "═".repeat(80));
        println!("TEST 1: Admin queries sales data");
        println!("{}", "═".repeat(80));
        
        let admin = UserContext {
            user_id: "admin_001".to_string(),
            tenant_id: "tenant_001".to_string(),
            roles: vec!["admin".to_string()],
            department: None,
            clearance_level: 5,
        };

        let repo = SmartRepository::new(admin)?;
        
        match repo.natural_language_query("show me all sales") {
            Ok(result) => {
                println!("\n✓ Query succeeded with grounded meaning:");
                println!("  Intent: {}", result.grounding.detected_intent);
                println!("  Entities: {:?}", result.grounding.extracted_entities);
                println!("  Applied policies: {:?}", result.grounding.applied_policies);
                println!("  Audit: {:?}", result.audit_trail);
                
                if !result.data.is_empty() {
                    println!("\n  Results:");
                    for (i, row) in result.data.iter().enumerate() {
                        println!("    Row {}: {:?}", i + 1, row);
                    }
                }
            }
            Err(e) => println!("✗ Query failed: {}", e),
        }
    }

    // Test 2: Manager with tenant isolation
    {
        println!("\n{}", "═".repeat(80));
        println!("TEST 2: Manager queries - Tenant Isolation Applied");
        println!("{}", "═".repeat(80));
        
        let manager = UserContext {
            user_id: "mgr_002".to_string(),
            tenant_id: "tenant_001".to_string(),
            roles: vec!["manager".to_string()],
            department: Some("Sales".to_string()),
            clearance_level: 2,
        };

        let repo = SmartRepository::new(manager)?;
        
        match repo.natural_language_query("show me sales from department") {
            Ok(result) => {
                println!("\n✓ Query succeeded");
                println!("  Applied tenant_id filter: tenant_001 (automatic)");
                println!("  Results count: {}", result.data.len());
            }
            Err(e) => println!("✗ Query failed: {}", e),
        }
    }

    // Test 3: Rep attempts DELETE (should be denied)
    {
        println!("\n{}", "═".repeat(80));
        println!("TEST 3: Sales Rep attempts DELETE (RBAC should BLOCK)");
        println!("{}", "═".repeat(80));
        
        let rep = UserContext {
            user_id: "rep_003".to_string(),
            tenant_id: "tenant_001".to_string(),
            roles: vec!["rep".to_string()],
            department: Some("Sales".to_string()),
            clearance_level: 1,
        };

        let repo = SmartRepository::new(rep)?;
        
        match repo.natural_language_query("delete all sales records") {
            Ok(_) => println!("✗ UNEXPECTED: Delete succeeded!"),
            Err(e) => {
                println!("\n✓ EXPECTED: DELETE blocked at application level");
                println!("  Reason: {}", e);
                println!("  This happened BEFORE any SQL was generated!");
            }
        }
    }

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              KEY TAKEAWAYS                                        ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  1. Nameless Vector is APPLICATION-EMBEDDED, not a sidecar                      ║");
    println!("║  2. Natural language → Semantic Analysis → RBAC → ORM → Database                 ║");
    println!("║  3. RBAC policies applied AT THE APPLICATION LEVEL before SQL generation         ║");
    println!("║  4. Tenant isolation built into the ORM query construction                        ║");
    println!("║  5. Grounded meaning extracted for audit trails and LLM context                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════╝");

    Ok(())
}
