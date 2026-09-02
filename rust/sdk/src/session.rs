//! Generation session for OpenAHI Rust SDK

use std::sync::Arc;
use std::time::Instant;

use openahi_runtime::OpenAHIRuntime;

use crate::config::GenerationConfig;
use crate::error::{SdkError, SdkResult};
use crate::model::{GenerationResult, FinishReason};

/// Generation session
///
/// Manages a generation session with state
#[derive(Debug, Clone)]
pub struct GenerationSession {
    /// Runtime
    runtime: Arc<OpenAHIRuntime>,
    
    /// Model name
    model: String,
    
    /// Model version
    version: String,
    
    /// Generation configuration
    config: GenerationConfig,
    
    /// Session ID
    session_id: String,
    
    /// Start time
    start_time: Instant,
    
    /// Request count
    request_count: u64,
    
    /// Total tokens generated
    total_tokens: u64,
}

impl GenerationSession {
    /// Create a new generation session
    pub fn new(
        runtime: Arc<OpenAHIRuntime>,
        model: String,
        version: String,
        config: GenerationConfig,
    ) -> SdkResult<Self> {
        Ok(Self {
            runtime,
            model,
            version,
            config,
            session_id: uuid::Uuid::new_v4().to_string(),
            start_time: Instant::now(),
            request_count: 0,
            total_tokens: 0,
        })
    }
    
    /// Get session ID
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
    
    /// Generate text
    pub fn generate(&mut self, prompt: &str) -> SdkResult<GenerationResult> {
        let runtime_config = self.config.to_runtime_config();
        
        let start = Instant::now();
        
        let result = self.runtime.generate(
            &self.model,
            &self.version,
            prompt,
            Some(runtime_config),
        )?;
        
        let elapsed = start.elapsed().as_millis() as u64;
        
        // Update statistics
        self.request_count += 1;
        self.total_tokens += prompt.len() as u64 / 4;
        
        Ok(GenerationResult {
            text: result,
            finish_reason: FinishReason::Complete,
            token_count: prompt.len() / 4,
            generation_time: elapsed,
            model: self.model.clone(),
            version: self.version.clone(),
        })
    }
    
    /// Batch generate
    pub fn batch_generate(&mut self, prompts: Vec<String>) -> SdkResult<Vec<GenerationResult>> {
        let runtime_config = self.config.to_runtime_config();
        
        let mut results = Vec::new();
        
        for prompt in prompts {
            let start = Instant::now();
            
            match self.runtime.generate(
                &self.model,
                &self.version,
                &prompt,
                Some(runtime_config.clone()),
            ) {
                Ok(text) => {
                    let elapsed = start.elapsed().as_millis() as u64;
                    
                    results.push(GenerationResult {
                        text,
                        finish_reason: FinishReason::Complete,
                        token_count: prompt.len() / 4,
                        generation_time: elapsed,
                        model: self.model.clone(),
                        version: self.version.clone(),
                    });
                    
                    // Update statistics
                    self.request_count += 1;
                    self.total_tokens += prompt.len() as u64 / 4;
                }
                Err(e) => {
                    results.push(GenerationResult {
                        text: format!("Error: {}", e),
                        finish_reason: FinishReason::Error,
                        token_count: 0,
                        generation_time: 0,
                        model: self.model.clone(),
                        version: self.version.clone(),
                    });
                }
            }
        }
        
        Ok(results)
    }
    
    /// Get session statistics
    pub fn stats(&self) -> SessionStats {
        SessionStats {
            session_id: self.session_id.clone(),
            model: self.model.clone(),
            version: self.version.clone(),
            request_count: self.request_count,
            total_tokens: self.total_tokens,
            uptime: self.start_time.elapsed().as_millis() as u64,
        }
    }
    
    /// Close the session
    pub fn close(&mut self) -> SdkResult<()> {
        // Clean up resources
        Ok(())
    }
}

/// Session statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStats {
    /// Session ID
    pub session_id: String,
    
    /// Model name
    pub model: String,
    
    /// Model version
    pub version: String,
    
    /// Request count
    pub request_count: u64,
    
    /// Total tokens
    pub total_tokens: u64,
    
    /// Uptime in milliseconds
    pub uptime: u64,
}
