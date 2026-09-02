//! Configuration types for OpenAHI Rust SDK

use serde::{Serialize, Deserialize};
use std::path::PathBuf;

/// SDK configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdkConfig {
    /// Runtime configuration path
    pub runtime_config: Option<PathBuf>,
    
    /// Default model name
    pub default_model: String,
    
    /// Default model version
    pub default_model_version: String,
    
    /// API endpoint (for remote runtime)
    pub api_endpoint: Option<String>,
    
    /// Timeout in seconds
    pub timeout: u64,
    
    /// Retry count
    pub retry_count: u32,
    
    /// Log level
    pub log_level: String,
    
    /// Debug mode
    pub debug: bool,
}

impl Default for SdkConfig {
    fn default() -> Self {
        Self {
            runtime_config: None,
            default_model: "composter".to_string(),
            default_model_version: "1.00.0".to_string(),
            api_endpoint: None,
            timeout: 30,
            retry_count: 3,
            log_level: "info".to_string(),
            debug: false,
        }
    }
}

impl SdkConfig {
    /// Create a new SDK configuration
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set default model
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }
    
    /// Set default model version
    pub fn with_default_model_version(mut self, version: impl Into<String>) -> Self {
        self.default_model_version = version.into();
        self
    }
    
    /// Set API endpoint
    pub fn with_api_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.api_endpoint = Some(endpoint.into());
        self
    }
    
    /// Load from file
    pub fn from_file(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }
    
    /// Save to file
    pub fn to_file(&self, path: impl AsRef<std::path::Path>) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

/// Generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    
    /// Temperature for sampling
    pub temperature: f32,
    
    /// Top-k sampling
    pub top_k: Option<usize>,
    
    /// Top-p (nucleus) sampling
    pub top_p: Option<f32>,
    
    /// Repetition penalty
    pub repetition_penalty: Option<f32>,
    
    /// Stop sequences
    pub stop_sequences: Option<Vec<String>>,
    
    /// Echo input
    pub echo: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            stop_sequences: None,
            echo: false,
        }
    }
}

impl GenerationConfig {
    /// Create a new generation configuration
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set maximum tokens
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }
    
    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }
    
    /// Set top-k
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }
    
    /// Set stop sequences
    pub fn with_stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(stop_sequences);
        self
    }
    
    /// Convert to runtime inference config
    pub fn to_runtime_config(&self) -> openahi_runtime::InferenceConfig {
        let mut config = openahi_runtime::InferenceConfig::default();
        config.max_tokens = self.max_tokens;
        config.temperature = self.temperature;
        config.top_k = self.top_k;
        config
    }
}

/// Model configuration for SDK
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name
    pub name: String,
    
    /// Model version
    pub version: String,
    
    /// Model type
    pub model_type: String,
    
    /// Vocabulary size
    pub vocab_size: usize,
    
    /// Context length
    pub context_length: usize,
    
    /// Embedding dimension
    pub embedding_dim: usize,
    
    /// Number of layers
    pub num_layers: usize,
    
    /// Number of heads
    pub num_heads: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: "composter".to_string(),
            version: "1.00.0".to_string(),
            model_type: "transformer".to_string(),
            vocab_size: 8192,
            context_length: 512,
            embedding_dim: 512,
            num_layers: 6,
            num_heads: 8,
        }
    }
}

impl ModelConfig {
    /// Create a new model configuration
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            ..Default::default()
        }
    }
}
