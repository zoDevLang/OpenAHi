//! Configuration types for OpenAHI Runtime

use serde::{Serialize, Deserialize};
use std::path::PathBuf;
use std::collections::HashMap;

/// Runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Path to the model directory
    pub model_dir: PathBuf,
    
    /// Path to the cache directory
    pub cache_dir: PathBuf,
    
    /// Maximum number of loaded models
    pub max_loaded_models: usize,
    
    /// Maximum memory usage in bytes (0 = unlimited)
    pub max_memory: u64,
    
    /// Number of inference threads
    pub num_threads: usize,
    
    /// Enable GPU acceleration (if available)
    pub enable_gpu: bool,
    
    /// Log level
    pub log_level: String,
    
    /// Debug mode
    pub debug: bool,
    
    /// API configuration
    pub api: ApiConfig,
    
    /// Security configuration
    pub security: SecurityConfig,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from("./models"),
            cache_dir: PathBuf::from("./cache"),
            max_loaded_models: 10,
            max_memory: 0, // Unlimited
            num_threads: num_cpus::get().unwrap_or(4),
            enable_gpu: true,
            log_level: "info".to_string(),
            debug: false,
            api: ApiConfig::default(),
            security: SecurityConfig::default(),
        }
    }
}

impl RuntimeConfig {
    /// Create a new runtime configuration
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the model directory
    pub fn with_model_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_dir = path.into();
        self
    }
    
    /// Set the cache directory
    pub fn with_cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = path.into();
        self
    }
    
    /// Load configuration from a JSON file
    pub fn from_file(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }
    
    /// Save configuration to a JSON file
    pub fn to_file(&self, path: impl AsRef<std::path::Path>) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

/// API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// Enable HTTP API
    pub enable_http: bool,
    
    /// HTTP server address
    pub http_address: String,
    
    /// HTTP server port
    pub http_port: u16,
    
    /// Enable WebSocket API
    pub enable_websocket: bool,
    
    /// WebSocket server address
    pub websocket_address: String,
    
    /// WebSocket server port
    pub websocket_port: u16,
    
    /// CORS origins (empty = allow all)
    pub cors_origins: Vec<String>,
    
    /// Maximum request size in bytes
    pub max_request_size: u64,
    
    /// Request timeout in seconds
    pub request_timeout: u64,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            enable_http: true,
            http_address: "127.0.0.1".to_string(),
            http_port: 8080,
            enable_websocket: false,
            websocket_address: "127.0.0.1".to_string(),
            websocket_port: 8081,
            cors_origins: vec![],
            max_request_size: 10 * 1024 * 1024, // 10 MB
            request_timeout: 30, // 30 seconds
        }
    }
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable checksum verification
    pub verify_checksums: bool,
    
    /// Enable sandboxing
    pub enable_sandbox: bool,
    
    /// Allowed model sources
    pub allowed_sources: Vec<String>,
    
    /// Blocked model names
    pub blocked_models: Vec<String>,
    
    /// Maximum file size for model downloads
    pub max_download_size: u64,
    
    /// Enable HTTPS only
    pub https_only: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            verify_checksums: true,
            enable_sandbox: false,
            allowed_sources: vec![],
            blocked_models: vec![],
            max_download_size: 1024 * 1024 * 1024, // 1 GB
            https_only: true,
        }
    }
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name
    pub name: String,
    
    /// Model version
    pub version: String,
    
    /// Model type (e.g., "composter")
    pub model_type: String,
    
    /// Path to model file
    pub path: PathBuf,
    
    /// Model architecture configuration
    pub architecture: ArchitectureConfig,
    
    /// Tokenizer configuration
    pub tokenizer: TokenizerConfig,
    
    /// Inference configuration
    pub inference: InferenceConfig,
    
    /// Model metadata
    pub metadata: ModelMetadata,
}

impl ModelConfig {
    /// Create a new model configuration
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            model_type: "composter".to_string(),
            path: PathBuf::new(),
            architecture: ArchitectureConfig::default(),
            tokenizer: TokenizerConfig::default(),
            inference: InferenceConfig::default(),
            metadata: ModelMetadata::default(),
        }
    }
}

/// Architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    
    /// Context length
    pub context_length: usize,
    
    /// Embedding dimension
    pub embedding_dim: usize,
    
    /// Number of layers
    pub num_layers: usize,
    
    /// Number of attention heads
    pub num_heads: usize,
    
    /// Dropout rate
    pub dropout: f32,
    
    /// Head dimension (derived)
    #[serde(skip)]
    pub head_dim: usize,
}

impl Default for ArchitectureConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8192,
            context_length: 512,
            embedding_dim: 512,
            num_layers: 6,
            num_heads: 8,
            dropout: 0.1,
            head_dim: 512 / 8, // embedding_dim / num_heads
        }
    }
}

impl ArchitectureConfig {
    /// Validate the configuration
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.embedding_dim % self.num_heads != 0 {
            anyhow::bail!(
                "embedding_dim ({}) must be divisible by num_heads ({})",
                self.embedding_dim,
                self.num_heads
            );
        }
        Ok(())
    }
    
    /// Calculate head dimension
    pub fn calculate_head_dim(&mut self) {
        self.head_dim = self.embedding_dim / self.num_heads;
    }
}

/// Tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Tokenizer type
    pub tokenizer_type: String,
    
    /// Vocabulary file path
    pub vocab_path: Option<PathBuf>,
    
    /// Merges file path (for BPE)
    pub merges_path: Option<PathBuf>,
    
    /// Special tokens
    pub special_tokens: HashMap<String, u32>,
    
    /// Maximum sequence length
    pub max_length: usize,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        let mut special_tokens = HashMap::new();
        special_tokens.insert("[PAD]".to_string(), 0);
        special_tokens.insert("[BOS]".to_string(), 1);
        special_tokens.insert("[EOS]".to_string(), 2);
        special_tokens.insert("[UNK]".to_string(), 3);
        
        Self {
            tokenizer_type: "bpe".to_string(),
            vocab_path: None,
            merges_path: None,
            special_tokens,
            max_length: 512,
        }
    }
}

/// Inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
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
    
    /// Number of return sequences
    pub num_return_sequences: usize,
    
    /// End-of-sequence token ID
    pub eos_token_id: Option<u32>,
    
    /// Pad token ID
    pub pad_token_id: u32,
    
    /// Use GPU if available
    pub use_gpu: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            num_return_sequences: 1,
            eos_token_id: Some(2), // [EOS] token
            pad_token_id: 0, // [PAD] token
            use_gpu: true,
        }
    }
}

impl InferenceConfig {
    /// Create a new inference configuration
    pub fn new() -> Self {
        Self::default()
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
    
    /// Set maximum tokens
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    
    /// Model version
    pub version: String,
    
    /// Model description
    pub description: String,
    
    /// Creator
    pub creator: String,
    
    /// License
    pub license: String,
    
    /// Parameter count
    pub parameter_count: u64,
    
    /// Context length
    pub context_length: usize,
    
    /// Creation date
    pub created_at: String,
    
    /// Last updated date
    pub updated_at: String,
    
    /// Checksum
    pub checksum: String,
    
    /// Tags
    pub tags: Vec<String>,
    
    /// Dependencies
    pub dependencies: Vec<String>,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            name: "composter".to_string(),
            version: "1.00.0".to_string(),
            description: "Composter 1.00.0 - The first OpenAHI model".to_string(),
            creator: "ZoDev".to_string(),
            license: "Apache-2.0".to_string(),
            parameter_count: 0,
            context_length: 512,
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
            checksum: String::new(),
            tags: vec!["transformer".to_string(), "language-model".to_string()],
            dependencies: vec![],
        }
    }
}

impl ModelMetadata {
    /// Create metadata for Composter 1.00.0
    pub fn composter_1_00_0() -> Self {
        Self {
            name: "composter".to_string(),
            version: "1.00.0".to_string(),
            description: "Composter 1.00.0 - A small transformer language model for OpenAHI".to_string(),
            creator: "ZoDev".to_string(),
            license: "Apache-2.0".to_string(),
            parameter_count: 0, // Will be calculated
            context_length: 512,
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
            checksum: String::new(),
            tags: vec![
                "transformer".to_string(),
                "language-model".to_string(),
                "openahi".to_string(),
                "prototype".to_string(),
            ],
            dependencies: vec![],
        }
    }
}
