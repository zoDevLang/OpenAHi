//! Error types for OpenAHI Runtime

use std::fmt;
use std::path::PathBuf;

/// Result type for OpenAHI runtime operations
pub type RuntimeResult<T> = anyhow::Result<T, RuntimeError>;

/// Error type for OpenAHI runtime
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// Model not found
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    /// Model version not found
    #[error("Model version not found: {0}@{1}")]
    ModelVersionNotFound(String, String),
    
    /// Invalid model configuration
    #[error("Invalid model configuration: {0}")]
    InvalidModelConfig(String),
    
    /// Failed to load model
    #[error("Failed to load model: {0}")]
    ModelLoadError(String),
    
    /// Failed to save model
    #[error("Failed to save model: {0}")]
    ModelSaveError(String),
    
    /// Inference error
    #[error("Inference error: {0}")]
    InferenceError(String),
    
    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    
    /// Checksum verification failed
    #[error("Checksum verification failed for: {0}")]
    ChecksumError(PathBuf),
    
    /// Model file corrupted
    #[error("Model file corrupted: {0}")]
    CorruptedModelFile(String),
    
    /// Incompatible model version
    #[error("Incompatible model version: expected {0}, got {1}")]
    IncompatibleVersion(String, String),
    
    /// Resource error (out of memory, etc.)
    #[error("Resource error: {0}")]
    ResourceError(String),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    /// Not implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

impl RuntimeError {
    /// Create a new runtime error with a message
    pub fn new(message: impl Into<String>) -> Self {
        RuntimeError::RuntimeError(message.into())
    }
    
    /// Check if error is a model not found error
    pub fn is_model_not_found(&self) -> bool {
        matches!(self, RuntimeError::ModelNotFound(_))
    }
    
    /// Check if error is a version compatibility error
    pub fn is_version_error(&self) -> bool {
        matches!(self, RuntimeError::IncompatibleVersion(_, _))
    }
}

/// Custom error type for more detailed errors
#[derive(Debug, Clone)]
pub struct RuntimeErrorDetails {
    pub error: RuntimeError,
    pub context: Option<String>,
    pub timestamp: std::time::SystemTime,
}

impl fmt::Display for RuntimeErrorDetails {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RuntimeError: {}", self.error)?;
        if let Some(ctx) = &self.context {
            write!(f, " (context: {})", ctx)?;
        }
        Ok(())
    }
}

impl std::error::Error for RuntimeErrorDetails {}

impl RuntimeErrorDetails {
    pub fn new(error: RuntimeError) -> Self {
        Self {
            error,
            context: None,
            timestamp: std::time::SystemTime::now(),
        }
    }
    
    pub fn with_context(error: RuntimeError, context: impl Into<String>) -> Self {
        Self {
            error,
            context: Some(context.into()),
            timestamp: std::time::SystemTime::now(),
        }
    }
}
