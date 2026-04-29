//! Observability Module
//! 
//! Provides structured logging, metrics, health checks, and distributed tracing.
//! P2 requirement for production-grade observability.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, instrument, span, warn, Span, Level};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, fmt, EnvFilter};

/// Initialize the tracing subscriber for structured logging
pub fn init_tracing() {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(fmt::layer()
            .with_target(true)
            .with_thread_ids(true)
            .with_line_number(true)
            .json()
        )
        .with(env_filter)
        .init();

    info!(target: "observability", "Tracing initialized");
}

/// Metrics collector for inference operations
#[derive(Debug)]
pub struct MetricsCollector {
    /// Total number of queries processed
    queries_total: AtomicU64,
    /// Number of successful queries
    queries_success: AtomicU64,
    /// Number of failed queries
    queries_failed: AtomicU64,
    /// Total query latency in microseconds
    query_latency_us: AtomicU64,
    /// Cache hit count
    cache_hits: AtomicU64,
    /// Cache miss count
    cache_misses: AtomicU64,
    /// GPU memory usage in bytes
    gpu_memory_bytes: AtomicU64,
    /// Active request count
    active_requests: AtomicU64,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self {
            queries_total: AtomicU64::new(0),
            queries_success: AtomicU64::new(0),
            queries_failed: AtomicU64::new(0),
            query_latency_us: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            gpu_memory_bytes: AtomicU64::new(0),
            active_requests: AtomicU64::new(0),
        }
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a query start, returns a guard that records end on drop
    pub fn start_query(&self) -> QueryGuard {
        self.active_requests.fetch_add(1, Ordering::Relaxed);
        let start_time = Instant::now();
        
        QueryGuard {
            collector: self,
            start_time,
        }
    }

    /// Record a successful query completion
    pub fn record_success(&self, latency: std::time::Duration) {
        self.queries_total.fetch_add(1, Ordering::Relaxed);
        self.queries_success.fetch_add(1, Ordering::Relaxed);
        self.query_latency_us.fetch_add(
            latency.as_micros() as u64,
            Ordering::Relaxed
        );
    }

    /// Record a failed query
    pub fn record_failure(&self) {
        self.queries_total.fetch_add(1, Ordering::Relaxed);
        self.queries_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record cache hit
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record cache miss
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Update GPU memory usage
    pub fn set_gpu_memory(&self, bytes: u64) {
        self.gpu_memory_bytes.store(bytes, Ordering::Relaxed);
    }

    /// Decrement active requests
    fn decrement_active(&self) {
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get current metrics snapshot
    pub fn snapshot(&self) -> MetricsSnapshot {
        let total = self.queries_total.load(Ordering::Relaxed);
        let success = self.queries_success.load(Ordering::Relaxed);
        let failed = self.queries_failed.load(Ordering::Relaxed);
        let latency_total = self.query_latency_us.load(Ordering::Relaxed);
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        
        let avg_latency_us = if total > 0 {
            latency_total / total
        } else {
            0
        };
        
        let cache_hit_rate = if hits + misses > 0 {
            (hits as f64 / (hits + misses) as f64) * 100.0
        } else {
            0.0
        };

        MetricsSnapshot {
            queries_total: total,
            queries_success: success,
            queries_failed: failed,
            queries_active: self.active_requests.load(Ordering::Relaxed),
            avg_latency_us,
            cache_hit_rate,
            gpu_memory_bytes: self.gpu_memory_bytes.load(Ordering::Relaxed),
        }
    }
}

/// Guard struct that automatically records query end on drop
pub struct QueryGuard<'a> {
    collector: &'a MetricsCollector,
    start_time: Instant,
}

impl<'a> Drop for QueryGuard<'a> {
    fn drop(&mut self) {
        self.collector.decrement_active();
    }
}

impl<'a> QueryGuard<'a> {
    /// Record successful completion
    pub fn success(self) {
        let latency = self.start_time.elapsed();
        self.collector.record_success(latency);
    }

    /// Record failure
    pub fn failure(self) {
        self.collector.record_failure();
    }
}

/// Metrics snapshot for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub queries_total: u64,
    pub queries_success: u64,
    pub queries_failed: u64,
    pub queries_active: u64,
    pub avg_latency_us: u64,
    pub cache_hit_rate: f64,
    pub gpu_memory_bytes: u64,
}

/// Health check status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: HealthState,
    pub version: String,
    pub timestamp: String,
    pub checks: HashMap<String, CheckResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthState {
    Healthy,
    Degraded,
    Unhealthy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    pub status: CheckState,
    pub message: String,
    pub latency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CheckState {
    Pass,
    Fail,
    Warn,
}

/// Health checker for system components
pub struct HealthChecker {
    checks: HashMap<String, Box<dyn Fn() -> CheckResult + Send + Sync>>,
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            checks: HashMap::new(),
        }
    }

    /// Register a health check
    pub fn register_check<F>(&mut self, name: &str, check: F)
    where
        F: Fn() -> CheckResult + Send + Sync + 'static,
    {
        self.checks.insert(name.to_string(), Box::new(check));
    }

    /// Run all health checks
    #[instrument(skip(self))]
    pub fn check_all(&self) -> HealthStatus {
        let mut results = HashMap::new();
        let mut overall = HealthState::Healthy;

        for (name, check_fn) in &self.checks {
            let start = Instant::now();
            let result = check_fn();
            let latency = start.elapsed().as_millis() as u64;

            let check_result = CheckResult {
                status: result.status.clone(),
                message: result.message.clone(),
                latency_ms: latency,
            };

            match result.status {
                CheckState::Pass => {}
                CheckState::Warn if matches!(overall, HealthState::Healthy) => {
                    overall = HealthState::Degraded;
                }
                CheckState::Fail => {
                    overall = HealthState::Unhealthy;
                }
                _ => {}
            }

            results.insert(name.clone(), check_result);
        }

        HealthStatus {
            status: overall,
            version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            checks: results,
        }
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Request context for distributed tracing
#[derive(Debug, Clone)]
pub struct RequestContext {
    pub request_id: String,
    pub span: Span,
    pub start_time: Instant,
}

impl RequestContext {
    /// Create a new request context with unique ID
    pub fn new(operation: &str) -> Self {
        let request_id = format!("{}-{:016x}", operation, rand::random::<u64>());
        let span = span!(Level::INFO, "request", request_id = %request_id, operation = %operation);
        
        Self {
            request_id,
            span,
            start_time: Instant::now(),
        }
    }

    /// Log with request context
    pub fn log_info(&self, message: &str) {
        let _enter = self.span.enter();
        info!(request_id = %self.request_id, "{}", message);
    }

    pub fn log_error(&self, message: &str) {
        let _enter = self.span.enter();
        error!(request_id = %self.request_id, "{}", message);
    }

    pub fn log_warn(&self, message: &str) {
        let _enter = self.span.enter();
        warn!(request_id = %self.request_id, "{}", message);
    }

    pub fn log_debug(&self, message: &str) {
        let _enter = self.span.enter();
        debug!(request_id = %self.request_id, "{}", message);
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }
}

/// Performance timer for measuring operation latency
pub struct Timer {
    start: Instant,
    name: String,
}

impl Timer {
    pub fn new(name: &str) -> Self {
        Self {
            start: Instant::now(),
            name: name.to_string(),
        }
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        let elapsed = self.elapsed_ms();
        debug!(operation = %self.name, latency_ms = elapsed, "Operation completed");
    }
}

/// Standardized error logging with context
#[macro_export]
macro_rules! log_error {
    ($ctx:expr, $msg:expr, $err:expr) => {
        error!(
            request_id = %$ctx.request_id,
            error = %$err,
            "{}: {}",
            $msg,
            $err
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector() {
        let metrics = MetricsCollector::new();
        
        // Record some operations
        let guard = metrics.start_query();
        guard.success();
        
        let guard2 = metrics.start_query();
        guard2.failure();
        
        metrics.record_cache_hit();
        metrics.record_cache_miss();
        
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.queries_total, 2);
        assert_eq!(snapshot.queries_success, 1);
        assert_eq!(snapshot.queries_failed, 1);
        assert_eq!(snapshot.cache_hit_rate, 50.0);
    }

    #[test]
    fn test_health_checker() {
        let mut checker = HealthChecker::new();
        
        checker.register_check("test", || CheckResult {
            status: CheckState::Pass,
            message: "Test passed".to_string(),
            latency_ms: 1,
        });

        let status = checker.check_all();
        assert!(matches!(status.status, HealthState::Healthy));
        assert!(status.checks.contains_key("test"));
    }

    #[test]
    fn test_request_context() {
        let ctx = RequestContext::new("test_op");
        assert!(!ctx.request_id.is_empty());
        assert!(ctx.request_id.starts_with("test_op-"));
    }
}
