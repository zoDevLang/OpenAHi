//! Model types for OpenAHI Rust SDK

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model name
    pub name: String,
    
    /// Model version
    pub version: String,
    
    /// Model type
    pub model_type: String,
    
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
    
    /// Vocabulary size
    pub vocab_size: usize,
    
    /// Created at
    pub created_at: String,
    
    /// Updated at
    pub updated_at: String,
    
    /// Checksum
    pub checksum: String,
    
    /// Tags
    pub tags: Vec<String>,
}

impl Default for ModelInfo {
    fn default() -> Self {
        Self {
            name: "composter".to_string(),
            version: "1.00.0".to_string(),
            model_type: "transformer".to_string(),
            description: "Composter 1.00.0 - The first OpenAHI model".to_string(),
            creator: "ZoDev".to_string(),
            license: "Apache-2.0".to_string(),
            parameter_count: 0,
            context_length: 512,
            vocab_size: 8192,
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
            checksum: String::new(),
            tags: vec![
                "transformer".to_string(),
                "language-model".to_string(),
                "openahi".to_string(),
            ],
        }
    }
}

impl ModelInfo {
    /// Create model info for Composter 1.00.0
    pub fn composter_1_00_0() -> Self {
        Self {
            name: "composter".to_string(),
            version: "1.00.0".to_string(),
            model_type: "transformer".to_string(),
            description: "Composter 1.00.0 - A small transformer language model for OpenAHI".to_string(),
            creator: "ZoDev".to_string(),
            license: "Apache-2.0".to_string(),
            parameter_count: 0,
            context_length: 512,
            vocab_size: 8192,
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
            checksum: String::new(),
            tags: vec![
                "transformer".to_string(),
                "language-model".to_string(),
                "openahi".to_string(),
                "prototype".to_string(),
            ],
        }
    }
}

/// Model list response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelList {
    /// Available models
    pub models: Vec<ModelInfo>,
    
    /// Total count
    pub total: usize,
}

impl ModelList {
    /// Create a new model list
    pub fn new(models: Vec<ModelInfo>) -> Self {
        Self {
            models,
            total: models.len(),
        }
    }
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    
    /// Model version
    pub version: String,
    
    /// Architecture
    pub architecture: HashMap<String, String>,
    
    /// Files
    pub files: HashMap<String, String>,
    
    /// Dependencies
    pub dependencies: Vec<String>,
}

impl ModelMetadata {
    /// Create metadata for Composter 1.00.0
    pub fn composter_1_00_0() -> Self {
        let mut architecture = HashMap::new();
        architecture.insert("type".to_string(), "transformer".to_string());
        architecture.insert("vocab_size".to_string(), "8192".to_string());
        architecture.insert("context_length".to_string(), "512".to_string());
        architecture.insert("embedding_dim".to_string(), "512".to_string());
        architecture.insert("num_layers".to_string(), "6".to_string());
        architecture.insert("num_heads".to_string(), "8".to_string());
        
        let mut files = HashMap::new();
        files.insert("model".to_string(), "composter_1.00.0.pt".to_string());
        files.insert("config".to_string(), "config.json".to_string());
        files.insert("vocab".to_string(), "vocab.json".to_string());
        
        Self {
            name: "composter".to_string(),
            version: "1.00.0".to_string(),
            architecture,
            files,
            dependencies: vec![],
        }
    }
}

/// Generation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    /// Generated text
    pub text: String,
    
    /// Finish reason
    pub finish_reason: FinishReason,
    
    /// Token count
    pub token_count: usize,
    
    /// Generation time in milliseconds
    pub generation_time: u64,
    
    /// Model name
    pub model: String,
    
    /// Model version
    pub version: String,
}

/// Finish reason for generation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FinishReason {
    /// Generation completed normally
    #[serde(rename = "complete")]
    Complete,
    /// Stopped due to EOS token
    #[serde(rename = "end_of_sequence")]
    EndOfSequence,
    /// Stopped due to max tokens
    #[serde(rename = "max_tokens")]
    MaxTokens,
    /// Stopped due to stop sequence
    #[serde(rename = "stop_sequence")]
    StopSequence,
    /// Stopped due to error
    #[serde(rename = "error")]
    Error,
}

impl Default for FinishReason {
    fn default() -> Self {
        Self::Complete
    }
}

/// Batch generation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchGenerationResult {
    /// Results for each prompt
    pub results: Vec<GenerationResult>,
    
    /// Total generation time
    pub total_time: u64,
    
    /// Number of successful generations
    pub success_count: usize,
    
    /// Number of failed generations
    pub failure_count: usize,
}

impl BatchGenerationResult {
    /// Create a new batch result
    pub fn new(results: Vec<GenerationResult>, total_time: u64) -> Self {
        let success_count = results.iter().filter(|r| !r.text.is_empty()).count();
        let failure_count = results.len() - success_count;
        
        Self {
            results,
            total_time,
            success_count,
            failure_count,
        }
    }
}
