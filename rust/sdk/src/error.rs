//! Error types for OpenAHI Rust SDK

use std::fmt;

use openahi_runtime::RuntimeError;

/// Result type for OpenAHI SDK operations
pub type SdkResult<T> = anyhow::Result<T, SdkError>;

/// Error type for OpenAHI SDK
#[derive(Debug, thiserror::Error)]
pub enum SdkError {
    /// Runtime error
    #[error("Runtime error: {0}")]
    Runtime(#[from] RuntimeError),
    
    /// Connection error
    #[error("Connection error: {0}")]
    ConnectionError(String),
    
    /// Model not found
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    
    /// Generation error
    #[error("Generation error: {0}")]
    GenerationError(String),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// JSON error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

impl SdkError {
    /// Create a new SDK error
    pub fn new(message: impl Into<String>) -> Self {
        SdkError::RuntimeError(message.into())
    }
    
    /// Check if error is a model not found error
    pub fn is_model_not_found(&self) -> bool {
        matches!(self, SdkError::ModelNotFound(_))
    }
    
    /// Check if error is a connection error
    pub fn is_connection_error(&self) -> bool {
        matches!(self, SdkError::ConnectionError(_))
    }
}

/// SDK error details
#[derive(Debug, Clone)]
pub struct SdkErrorDetails {
    pub error: SdkError,
    pub context: Option<String>,
    pub timestamp: std::time::SystemTime,
}

impl fmt::Display for SdkErrorDetails {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SdkError: {}", self.error)?;
        if let Some(ctx) = &self.context {
            write!(f, " (context: {})", ctx)?;
        }
        Ok(())
    }
}

impl std::error::Error for SdkErrorDetails {}
