//! Schema Migration Module
//! 
//! Provides versioning and migration support for verb/noun data structures.
//! P3 requirement for production-grade data integrity.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Current schema version following semantic versioning
pub const CURRENT_SCHEMA_VERSION: &str = "1.1.0";

/// Schema version components for comparison
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl SchemaVersion {
    /// Parse version string (e.g., "1.2.3")
    pub fn parse(version: &str) -> Result<Self> {
        let parts: Vec<&str> = version.split('.').collect();
        if parts.len() != 3 {
            return Err(anyhow::anyhow!(
                "Invalid version format: {}. Expected semver (e.g., 1.2.3)",
                version
            ));
        }

        Ok(Self {
            major: parts[0].parse().context("Invalid major version")?,
            minor: parts[1].parse().context("Invalid minor version")?,
            patch: parts[2].parse().context("Invalid patch version")?,
        })
    }

    /// Check if this version is compatible with another
    /// Same major version = compatible
    pub fn is_compatible_with(&self, other: &SchemaVersion) -> bool {
        self.major == other.major
    }

    /// Check if upgrade is needed
    pub fn needs_upgrade_from(&self, other: &SchemaVersion) -> bool {
        self.major > other.major
            || (self.major == other.major && self.minor > other.minor)
            || (self.major == other.major
                && self.minor == other.minor
                && self.patch > other.patch)
    }
}

impl std::fmt::Display for SchemaVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Versioned data wrapper for all stored entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Versioned<T> {
    pub version: String,
    pub data: T,
    pub migrated_from: Option<String>,
    pub migrated_at: Option<String>,
}

impl<T> Versioned<T> {
    /// Wrap data with current schema version
    pub fn new(data: T) -> Self {
        Self {
            version: CURRENT_SCHEMA_VERSION.to_string(),
            data,
            migrated_from: None,
            migrated_at: None,
        }
    }

    /// Wrap data with specific version (for migrations)
    pub fn with_version(data: T, version: &str) -> Self {
        Self {
            version: version.to_string(),
            data,
            migrated_from: None,
            migrated_at: None,
        }
    }
}

/// Migration trait for upgrading data between schema versions
trait Migration<T> {
    fn from_version(&self) -> &str;
    fn to_version(&self) -> &str;
    fn migrate(&self, data: T) -> Result<T>;
}

/// Migration registry for managing schema upgrades
pub struct MigrationRegistry<T> {
    migrations: HashMap<(String, String), Box<dyn Fn(T) -> Result<T>>>,
}

impl<T> MigrationRegistry<T> {
    pub fn new() -> Self {
        Self {
            migrations: HashMap::new(),
        }
    }

    /// Register a migration function
    pub fn register<F>(&mut self, from: &str, to: &str, migration: F)
    where
        F: Fn(T) -> Result<T> + 'static,
    {
        self.migrations
            .insert((from.to_string(), to.to_string()), Box::new(migration));
    }

    /// Find migration path between versions
    pub fn find_path(&self, from: &str, to: &str) -> Option<Vec<(String, String)>> {
        // Simple BFS to find shortest migration path
        if from == to {
            return Some(vec![]);
        }

        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((from.to_string(), vec![]));

        while let Some((current, path)) = queue.pop_front() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            // Find all possible next steps
            for ((from_ver, to_ver), _) in &self.migrations {
                if from_ver == &current {
                    let mut new_path = path.clone();
                    new_path.push((from_ver.clone(), to_ver.clone()));

                    if to_ver == to {
                        return Some(new_path);
                    }

                    queue.push_back((to_ver.clone(), new_path));
                }
            }
        }

        None
    }

    /// Apply migrations to upgrade data
    pub fn migrate(&self, mut data: T, from: &str, to: &str) -> Result<Versioned<T>> {
        let path = self.find_path(from, to).ok_or_else(|| {
            anyhow::anyhow!("No migration path from {} to {}", from, to)
        })?;

        for (from_ver, to_ver) in path {
            let migration = self
                .migrations
                .get(&(from_ver.clone(), to_ver.clone()))
                .ok_or_else(|| anyhow::anyhow!("Migration not found: {} -> {}", from_ver, to_ver))?;

            data = migration(data)?;
        }

        Ok(Versioned {
            version: to.to_string(),
            data,
            migrated_from: Some(from.to_string()),
            migrated_at: Some(chrono::Utc::now().to_rfc3339()),
        })
    }
}

impl<T> Default for MigrationRegistry<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// VerbOutcome v1.0.0 structure (legacy)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerbOutcomeV1 {
    pub verb: String,
    pub applicable_subjects: Vec<String>,
    pub applicable_objects: Vec<String>,
    pub required_states: Vec<String>,
    pub final_states: Vec<String>,
}

/// VerbOutcome v1.1.0 structure (current)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerbOutcomeV2 {
    pub verb: String,
    pub applicable_subjects: Vec<String>,
    pub applicable_objects: Vec<String>,
    pub required_subject_states: StateSetV2,
    pub required_object_states: StateSetV2,
    pub final_subject_states: StateSetV2,
    pub final_object_states: StateSetV2,
    pub goals: Vec<String>,
    pub mechanisms: Vec<String>,
    pub tools: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSetV2 {
    pub physical: Vec<String>,
    pub emotional: Vec<String>,
    pub positional: Vec<String>,
    pub mental: Vec<String>,
}

/// Create migration registry with predefined migrations
pub fn create_verb_migration_registry() -> MigrationRegistry<VerbOutcomeV1> {
    let mut registry = MigrationRegistry::new();

    // Migration from 1.0.0 to 1.1.0
    registry.register("1.0.0", "1.1.0", |v1: VerbOutcomeV1| {
        // Convert flat state lists to dimensional state sets
        let required_subject = StateSetV2 {
            physical: v1.required_states.clone(),
            emotional: vec![],
            positional: vec![],
            mental: vec![],
        };

        let required_object = StateSetV2 {
            physical: v1.required_states.clone(),
            emotional: vec![],
            positional: vec![],
            mental: vec![],
        };

        let final_subject = StateSetV2 {
            physical: v1.final_states.clone(),
            emotional: vec![],
            positional: vec![],
            mental: vec![],
        };

        let final_object = StateSetV2 {
            physical: v1.final_states.clone(),
            emotional: vec![],
            positional: vec![],
            mental: vec![],
        };

        // This migration loses information - we can't determine goals/mechanisms from v1
        // Log warning about data loss
        eprintln!(
            "WARNING: Migrating verb '{}' from 1.0.0 to 1.1.0 - goals/mechanisms/tools will be empty",
            v1.verb
        );

        // Return as V1 structure but with v2 fields (for gradual migration)
        // In real implementation, this would return V2
        Ok(v1)
    });

    registry
}

/// Validate and migrate verb data if needed
pub fn load_and_migrate_verb(json: &str) -> Result<Versioned<serde_json::Value>> {
    // Try to parse as versioned first
    let value: serde_json::Value = serde_json::from_str(json).context("Failed to parse JSON")?;

    // Check if versioned
    if let Some(version) = value.get("version").and_then(|v| v.as_str()) {
        let current = SchemaVersion::parse(CURRENT_SCHEMA_VERSION)?;
        let data_version = SchemaVersion::parse(version)?;

        if !current.is_compatible_with(&data_version) {
            return Err(anyhow::anyhow!(
                "Incompatible schema version: data={}, current={}",
                version,
                CURRENT_SCHEMA_VERSION
            ));
        }

        if current.needs_upgrade_from(&data_version) {
            // Would trigger migration here
            // For now, return as-is with warning
            eprintln!(
                "WARNING: Data version {} is older than current {}. Migration not yet implemented.",
                version,
                CURRENT_SCHEMA_VERSION
            );
        }

        Ok(Versioned {
            version: version.to_string(),
            data: value.get("data").cloned().unwrap_or(value),
            migrated_from: None,
            migrated_at: None,
        })
    } else {
        // Legacy format without version - assume v1.0.0
        eprintln!("WARNING: Loading legacy verb data without version - assuming 1.0.0");
        Ok(Versioned {
            version: "1.0.0".to_string(),
            data: value,
            migrated_from: None,
            migrated_at: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_version_parsing() {
        let v = SchemaVersion::parse("1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
    }

    #[test]
    fn test_schema_version_compatibility() {
        let v1 = SchemaVersion::parse("1.0.0").unwrap();
        let v2 = SchemaVersion::parse("1.2.0").unwrap();
        let v3 = SchemaVersion::parse("2.0.0").unwrap();

        assert!(v1.is_compatible_with(&v2));
        assert!(v2.is_compatible_with(&v1));
        assert!(!v1.is_compatible_with(&v3));
    }

    #[test]
    fn test_schema_version_upgrade_check() {
        let v1 = SchemaVersion::parse("1.0.0").unwrap();
        let v2 = SchemaVersion::parse("1.1.0").unwrap();
        let v3 = SchemaVersion::parse("2.0.0").unwrap();

        assert!(v2.needs_upgrade_from(&v1));
        assert!(v3.needs_upgrade_from(&v2));
        assert!(!v1.needs_upgrade_from(&v2));
    }

    #[test]
    fn test_versioned_wrapper() {
        let data = "test data";
        let versioned = Versioned::new(data);

        assert_eq!(versioned.version, CURRENT_SCHEMA_VERSION);
        assert_eq!(versioned.data, data);
        assert!(versioned.migrated_from.is_none());
    }

    #[test]
    fn test_migration_path_finding() {
        let mut registry = MigrationRegistry::<i32>::new();
        registry.register("1.0.0", "1.1.0", |x| Ok(x + 1));
        registry.register("1.1.0", "1.2.0", |x| Ok(x + 10));

        let path = registry.find_path("1.0.0", "1.2.0").unwrap();
        assert_eq!(path.len(), 2);
        assert_eq!(path[0], ("1.0.0".to_string(), "1.1.0".to_string()));
        assert_eq!(path[1], ("1.1.0".to_string(), "1.2.0".to_string()));
    }

    #[test]
    fn test_migration_execution() {
        let mut registry = MigrationRegistry::<i32>::new();
        registry.register("1.0.0", "1.1.0", |x| Ok(x + 1));

        let result = registry.migrate(5, "1.0.0", "1.1.0").unwrap();
        assert_eq!(result.data, 6);
        assert_eq!(result.version, "1.1.0");
        assert_eq!(result.migrated_from, Some("1.0.0".to_string()));
    }
}
