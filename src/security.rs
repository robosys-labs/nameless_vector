//! Security Module
//! 
//! Provides input validation, rate limiting, resource quotas, and sanitization.
//! P2 requirement for production-grade security.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use anyhow::{Context, Result};
use regex::Regex;

/// Input validation rules
pub struct InputValidator {
    /// Maximum allowed input length
    max_length: usize,
    /// Allowed character pattern (whitelist)
    allowed_chars: Regex,
    /// Blocked patterns (blacklist)
    blocked_patterns: Vec<Regex>,
    /// Maximum nested structure depth (for JSON)
    max_depth: usize,
}

impl InputValidator {
    /// Create validator with production-safe defaults
    pub fn new() -> Self {
        Self {
            max_length: 10000,
            // Allow alphanumeric, spaces, basic punctuation, and common symbols
            allowed_chars: Regex::new(r#"^[\w\s\-.,:;!?()'"\/\\@#$%&*+=<>\[\]{}|^~`]+$"#).unwrap(),
            blocked_patterns: vec![
                // Block SQL injection patterns
                Regex::new(r#"(?i)(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|--|;|/\*)"#).unwrap(),
                // Block command injection
                Regex::new(r#"(?i)(\b(?:rm|ls|cat|echo|bash|sh|cmd|powershell)\b|[`|$(){}[\]])"#).unwrap(),
                // Block path traversal
                Regex::new(r#"\.\./|\.\.\\|%2e%2e"#).unwrap(),
                // Block common XSS patterns
                Regex::new(r#"(?i)(<script|javascript:|on\w+\s*=)"#).unwrap(),
            ],
            max_depth: 10,
        }
    }

    /// Validate a query string
    pub fn validate_query(&self, input: &str) -> Result<()> {
        // Check length
        if input.len() > self.max_length {
            return Err(anyhow::anyhow!(
                "Input exceeds maximum length of {} characters",
                self.max_length
            ));
        }

        // Check empty
        if input.trim().is_empty() {
            return Err(anyhow::anyhow!("Input cannot be empty"));
        }

        // Check allowed characters
        if !self.allowed_chars.is_match(input) {
            return Err(anyhow::anyhow!(
                "Input contains invalid characters. Only alphanumeric, spaces, and basic punctuation allowed."
            ));
        }

        // Check blocked patterns
        for pattern in &self.blocked_patterns {
            if pattern.is_match(input) {
                return Err(anyhow::anyhow!(
                    "Input contains potentially malicious patterns"
                ));
            }
        }

        Ok(())
    }

    /// Sanitize input by removing/replacing dangerous characters
    pub fn sanitize(&self, input: &str) -> String {
        input
            .chars()
            .filter(|c| self.allowed_chars.is_match(&c.to_string()))
            .take(self.max_length)
            .collect()
    }

    /// Validate verb name (stricter rules)
    pub fn validate_verb_name(&self, verb: &str) -> Result<()> {
        if verb.len() > 50 {
            return Err(anyhow::anyhow!("Verb name too long (max 50 chars)"));
        }

        // Verbs should only contain alphabetic characters and hyphens
        let verb_pattern = Regex::new(r#"^[a-zA-Z][a-zA-Z\-]*$"#).unwrap();
        if !verb_pattern.is_match(verb) {
            return Err(anyhow::anyhow!(
                "Verb must start with letter and contain only letters and hyphens"
            ));
        }

        Ok(())
    }

    /// Validate state name
    pub fn validate_state_name(&self, state: &str) -> Result<()> {
        if state.len() > 100 {
            return Err(anyhow::anyhow!("State name too long (max 100 chars)"));
        }

        // States should be lowercase with underscores
        let state_pattern = Regex::new(r#"^[a-z][a-z_0-9]*$"#).unwrap();
        if !state_pattern.is_match(state) {
            return Err(anyhow::anyhow!(
                "State must be lowercase with underscores only"
            ));
        }

        Ok(())
    }

    /// Set custom max length
    pub fn with_max_length(mut self, length: usize) -> Self {
        self.max_length = length;
        self
    }
}

impl Default for InputValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Rate limiter using token bucket algorithm
pub struct RateLimiter {
    /// Requests per second allowed
    requests_per_second: u64,
    /// Burst capacity
    burst_capacity: u64,
    /// Per-IP tracking
    buckets: Mutex<HashMap<IpAddr, TokenBucket>>,
    /// Global bucket for all requests
    global_bucket: Mutex<TokenBucket>,
}

struct TokenBucket {
    tokens: f64,
    last_update: Instant,
    capacity: u64,
    rate: f64,
}

impl TokenBucket {
    fn new(capacity: u64, rate: f64) -> Self {
        Self {
            tokens: capacity as f64,
            last_update: Instant::now(),
            capacity,
            rate,
        }
    }

    /// Try to consume tokens, returns true if allowed
    fn try_consume(&mut self, tokens: u64) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        
        // Add tokens based on elapsed time
        self.tokens = (self.tokens + elapsed * self.rate).min(self.capacity as f64);
        self.last_update = now;

        // Check if we have enough tokens
        if self.tokens >= tokens as f64 {
            self.tokens -= tokens as f64;
            true
        } else {
            false
        }
    }
}

impl RateLimiter {
    /// Create rate limiter with default settings (100 req/s, burst of 150)
    pub fn new() -> Self {
        Self {
            requests_per_second: 100,
            burst_capacity: 150,
            buckets: Mutex::new(HashMap::new()),
            global_bucket: Mutex::new(TokenBucket::new(1000, 200.0)),
        }
    }

    /// Check if request is allowed from specific IP
    pub fn check_rate(&self, ip: IpAddr) -> Result<()> {
        // Check per-IP limit first (cheaper, less likely to fail)
        let ip_allowed = {
            let mut buckets = self.buckets.lock().unwrap_or_else(|poisoned| {
                // If mutex is poisoned, clear it and continue
                poisoned.into_inner()
            });
            let bucket = buckets.entry(ip).or_insert_with(|| {
                TokenBucket::new(self.burst_capacity, self.requests_per_second as f64)
            });
            bucket.try_consume(1)
        };

        if !ip_allowed {
            return Err(anyhow::anyhow!(
                "Rate limit exceeded for your IP. Max {} requests per second.",
                self.requests_per_second
            ));
        }

        // Then check global limit
        let global_allowed = {
            let mut global = self.global_bucket.lock().unwrap_or_else(|poisoned| {
                poisoned.into_inner()
            });
            global.try_consume(1)
        };

        if !global_allowed {
            // BUG FIX: Return the IP bucket token since we're rejecting
            let mut buckets = self.buckets.lock().unwrap_or_else(|poisoned| {
                poisoned.into_inner()
            });
            if let Some(bucket) = buckets.get_mut(&ip) {
                // Add token back (capped at capacity)
                bucket.tokens = (bucket.tokens + 1.0).min(bucket.capacity as f64);
            }
            return Err(anyhow::anyhow!(
                "Global rate limit exceeded. Please try again later."
            ));
        }

        Ok(())
    }

    /// Set custom rate limits
    pub fn with_limits(mut self, rps: u64, burst: u64) -> Self {
        self.requests_per_second = rps;
        self.burst_capacity = burst;
        self
    }

    /// Cleanup old entries (call periodically)
    pub fn cleanup(&self) {
        let mut buckets = match self.buckets.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                // Log and recover from poisoned mutex
                eprintln!("WARNING: Rate limiter mutex poisoned, recovering");
                poisoned.into_inner()
            }
        };
        let now = Instant::now();
        buckets.retain(|_, bucket| {
            // Keep if bucket was accessed in last 5 minutes
            now.duration_since(bucket.last_update).as_secs() < 300
        });
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

/// Resource quota manager for GPU/memory limits
pub struct ResourceQuotas {
    /// Maximum concurrent requests
    max_concurrent: usize,
    /// Current active requests
    active_count: AtomicU64,
    /// Maximum tokens per request
    max_tokens_per_request: usize,
    /// Maximum total tokens in flight
    max_total_tokens: usize,
    /// Current tokens in flight
    current_tokens: AtomicU64,
    /// Maximum GPU memory usage (bytes)
    max_gpu_memory_bytes: u64,
}

impl ResourceQuotas {
    /// Create with production-safe defaults
    pub fn new() -> Self {
        Self {
            max_concurrent: 10,
            active_count: AtomicU64::new(0),
            max_tokens_per_request: 800,
            max_total_tokens: 4000,
            current_tokens: AtomicU64::new(0),
            max_gpu_memory_bytes: 4 * 1024 * 1024 * 1024,
        }
    }

    /// Try to acquire resources for a request
    /// Uses CAS loop to prevent race conditions
    pub fn try_acquire(&self, requested_tokens: usize) -> Result<ResourceGuard> {
        // Check per-request token limit first (cheap check)
        if requested_tokens > self.max_tokens_per_request {
            return Err(anyhow::anyhow!(
                "Requested tokens {} exceeds maximum {}",
                requested_tokens,
                self.max_tokens_per_request
            ));
        }

        // Try to acquire concurrent slot using CAS loop
        loop {
            let active = self.active_count.load(Ordering::Acquire);
            if active >= self.max_concurrent as u64 {
                return Err(anyhow::anyhow!(
                    "Too many concurrent requests. Max {} allowed.",
                    self.max_concurrent
                ));
            }

            // Check token budget before acquiring
            let current_tokens = self.current_tokens.load(Ordering::Acquire);
            if current_tokens + requested_tokens as u64 > self.max_total_tokens as u64 {
                return Err(anyhow::anyhow!(
                    "Token budget exceeded. Try again with fewer tokens or wait."
                ));
            }

            // Attempt to increment - CAS prevents race condition
            match self.active_count.compare_exchange(
                active,
                active + 1,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // Successfully acquired slot, now add tokens
                    self.current_tokens.fetch_add(requested_tokens as u64, Ordering::Release);
                    return Ok(ResourceGuard {
                        quotas: self,
                        tokens: requested_tokens,
                    });
                }
                Err(_) => {
                    // CAS failed, another thread modified the count - retry
                    continue;
                }
            }
        }
    }

    /// Set custom limits
    pub fn with_limits(
        mut self,
        max_concurrent: usize,
        max_tokens: usize,
        max_gpu_memory: u64,
    ) -> Self {
        self.max_concurrent = max_concurrent;
        self.max_tokens_per_request = max_tokens;
        self.max_total_tokens = max_tokens * max_concurrent;
        self.max_gpu_memory_bytes = max_gpu_memory;
        self
    }

    /// Get current utilization
    pub fn utilization(&self) -> ResourceUtilization {
        ResourceUtilization {
            active_requests: self.active_count.load(Ordering::Relaxed),
            max_concurrent: self.max_concurrent as u64,
            tokens_in_flight: self.current_tokens.load(Ordering::Relaxed),
            max_tokens: self.max_total_tokens as u64,
            gpu_memory_used: 0,
            gpu_memory_max: self.max_gpu_memory_bytes,
        }
    }
}

impl Default for ResourceQuotas {
    fn default() -> Self {
        Self::new()
    }
}

/// Guard that releases resources on drop
pub struct ResourceGuard<'a> {
    quotas: &'a ResourceQuotas,
    tokens: usize,
}

impl<'a> Drop for ResourceGuard<'a> {
    fn drop(&mut self) {
        self.quotas.active_count.fetch_sub(1, Ordering::Relaxed);
        self.quotas.current_tokens.fetch_sub(self.tokens as u64, Ordering::Relaxed);
    }
}

/// Resource utilization snapshot
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    pub active_requests: u64,
    pub max_concurrent: u64,
    pub tokens_in_flight: u64,
    pub max_tokens: u64,
    pub gpu_memory_used: u64,
    pub gpu_memory_max: u64,
}

impl ResourceUtilization {
    /// Calculate load percentage
    pub fn load_percentage(&self) -> f64 {
        let req_load = self.active_requests as f64 / self.max_concurrent as f64;
        let token_load = self.tokens_in_flight as f64 / self.max_tokens as f64;
        req_load.max(token_load) * 100.0
    }

    pub fn is_overloaded(&self) -> bool {
        self.load_percentage() > 90.0
    }
}

/// API key validator for authenticated endpoints
pub struct ApiKeyValidator {
    valid_keys: Mutex<HashMap<String, ApiKeyInfo>>,
}

#[derive(Debug)]
struct ApiKeyInfo {
    name: String,
    created_at: Instant,
    requests_count: AtomicU64,
}

impl ApiKeyValidator {
    pub fn new() -> Self {
        Self {
            valid_keys: Mutex::new(HashMap::new()),
        }
    }

    /// Register a new API key
    pub fn register_key(&self, key: &str, name: &str) {
        let mut keys = match self.valid_keys.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        keys.insert(
            key.to_string(),
            ApiKeyInfo {
                name: name.to_string(),
                created_at: Instant::now(),
                requests_count: AtomicU64::new(0),
            },
        );
    }

    /// Validate an API key
    pub fn validate(&self, key: &str) -> Result<()> {
        let keys = match self.valid_keys.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        
        if let Some(info) = keys.get(key) {
            info.requests_count.fetch_add(1, Ordering::Relaxed);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Invalid API key"))
        }
    }

    /// Revoke an API key
    pub fn revoke_key(&self, key: &str) {
        let mut keys = match self.valid_keys.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        keys.remove(key);
    }
}

impl Default for ApiKeyValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Security middleware combining all protections
pub struct SecurityMiddleware {
    pub input_validator: InputValidator,
    pub rate_limiter: RateLimiter,
    pub resource_quotas: ResourceQuotas,
    pub api_key_validator: ApiKeyValidator,
}

impl SecurityMiddleware {
    pub fn new() -> Self {
        Self {
            input_validator: InputValidator::new(),
            rate_limiter: RateLimiter::new(),
            resource_quotas: ResourceQuotas::new(),
            api_key_validator: ApiKeyValidator::new(),
        }
    }

    /// Full request validation pipeline
    pub fn validate_request(
        &self,
        ip: IpAddr,
        api_key: Option<&str>,
        query: &str,
        requested_tokens: usize,
    ) -> Result<ResourceGuard> {
        // Validate API key if provided
        if let Some(key) = api_key {
            self.api_key_validator.validate(key)?;
        }

        // Check rate limit
        self.rate_limiter.check_rate(ip)?;

        // Validate input
        self.input_validator.validate_query(query)?;

        // Check resource quotas
        let guard = self.resource_quotas.try_acquire(requested_tokens)?;

        Ok(guard)
    }
}

impl Default for SecurityMiddleware {
    fn default() -> Self {
        Self::new()
    }
}
