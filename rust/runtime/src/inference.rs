//! Inference module for OpenAHI Runtime

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::config::InferenceConfig;
use crate::error::{RuntimeError, RuntimeResult};
use crate::model::Model;

/// Inference session
///
/// Manages a single inference session with a model
#[derive(Debug)]
pub struct InferenceSession {
    /// Model being used
    pub model: Arc<Mutex<Model>>,
    
    /// Inference configuration
    pub config: InferenceConfig,
    
    /// Session ID
    pub session_id: String,
    
    /// Start time
    pub start_time: Instant,
    
    /// Last activity time
    pub last_activity: Instant,
    
    /// Request count
    pub request_count: u64,
    
    /// Total tokens generated
    pub total_tokens: u64,
    
    /// Total inference time
    pub total_time: Duration,
}

impl InferenceSession {
    /// Create a new inference session
    pub fn new(model: Arc<Mutex<Model>>, config: InferenceConfig) -> Self {
        Self {
            model,
            config,
            session_id: uuid::Uuid::new_v4().to_string(),
            start_time: Instant::now(),
            last_activity: Instant::now(),
            request_count: 0,
            total_tokens: 0,
            total_time: Duration::from_secs(0),
        }
    }
    
    /// Generate text
    pub fn generate(&mut self, prompt: &str) -> RuntimeResult<String> {
        let start = Instant::now();
        
        let result = {
            let model = self.model.lock().unwrap();
            model.generate(prompt, &self.config)
        };
        
        let elapsed = start.elapsed();
        
        // Update statistics
        self.request_count += 1;
        self.last_activity = Instant::now();
        self.total_time += elapsed;
        
        // Estimate tokens (simplified)
        self.total_tokens += prompt.len() as u64 / 4; // Rough estimate
        
        result
    }
    
    /// Batch generate
    pub fn batch_generate(&mut self, prompts: &[String]) -> RuntimeResult<Vec<String>> {
        let start = Instant::now();
        
        let mut results = Vec::new();
        for prompt in prompts {
            results.push(self.generate(prompt)?);
        }
        
        let elapsed = start.elapsed();
        self.total_time += elapsed;
        self.request_count += prompts.len() as u64;
        self.last_activity = Instant::now();
        
        Ok(results)
    }
    
    /// Get session statistics
    pub fn stats(&self) -> InferenceStats {
        InferenceStats {
            session_id: self.session_id.clone(),
            request_count: self.request_count,
            total_tokens: self.total_tokens,
            total_time: self.total_time,
            avg_time_per_request: if self.request_count > 0 {
                self.total_time / self.request_count as u32
            } else {
                Duration::from_secs(0)
            },
            uptime: self.start_time.elapsed(),
        }
    }
    
    /// Close the session
    pub fn close(&mut self) -> RuntimeResult<()> {
        // Clean up resources
        Ok(())
    }
}

/// Inference statistics
#[derive(Debug, Clone)]
pub struct InferenceStats {
    pub session_id: String,
    pub request_count: u64,
    pub total_tokens: u64,
    pub total_time: Duration,
    pub avg_time_per_request: Duration,
    pub uptime: Duration,
}

/// Inference manager
///
/// Manages multiple inference sessions
#[derive(Debug, Default)]
pub struct InferenceManager {
    /// Active sessions
    pub sessions: Mutex<Vec<InferenceSession>>,
    
    /// Maximum concurrent sessions
    pub max_sessions: usize,
    
    /// Total requests
    pub total_requests: u64,
    
    /// Total tokens generated
    pub total_tokens: u64,
}

impl InferenceManager {
    /// Create a new inference manager
    pub fn new(max_sessions: usize) -> Self {
        Self {
            sessions: Mutex::new(Vec::new()),
            max_sessions,
            total_requests: 0,
            total_tokens: 0,
        }
    }
    
    /// Create a new session
    pub fn create_session(
        &self,
        model: Arc<Mutex<Model>>,
        config: InferenceConfig,
    ) -> RuntimeResult<InferenceSession> {
        let mut sessions = self.sessions.lock().unwrap();
        
        if sessions.len() >= self.max_sessions {
            return Err(RuntimeError::ResourceError(
                format!("Maximum sessions ({}) reached", self.max_sessions)
            ));
        }
        
        let session = InferenceSession::new(model, config);
        sessions.push(session.clone());
        
        Ok(session)
    }
    
    /// Close a session
    pub fn close_session(&self, session_id: &str) -> RuntimeResult<()> {
        let mut sessions = self.sessions.lock().unwrap();
        sessions.retain(|s| s.session_id != session_id);
        Ok(())
    }
    
    /// Get session by ID
    pub fn get_session(&self, session_id: &str) -> Option<InferenceSession> {
        let sessions = self.sessions.lock().unwrap();
        sessions.iter().find(|s| s.session_id == session_id).cloned()
    }
    
    /// List all sessions
    pub fn list_sessions(&self) -> Vec<InferenceStats> {
        let sessions = self.sessions.lock().unwrap();
        sessions.iter().map(|s| s.stats()).collect()
    }
    
    /// Clean up expired sessions
    pub fn cleanup_expired(&self, timeout: Duration) -> usize {
        let mut sessions = self.sessions.lock().unwrap();
        let now = Instant::now();
        
        let count = sessions.len();
        sessions.retain(|s| now.duration_since(s.last_activity) < timeout);
        count - sessions.len()
    }
}

/// Inference result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    /// Generated text
    pub text: String,
    
    /// Finish reason
    pub finish_reason: FinishReason,
    
    /// Number of tokens generated
    pub token_count: usize,
    
    /// Inference time
    pub inference_time: Duration,
}

/// Finish reason for inference
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FinishReason {
    /// Generation completed normally
    Complete,
    /// Stopped due to EOS token
    EndOfSequence,
    /// Stopped due to max tokens
    MaxTokens,
    /// Stopped due to error
    Error(String),
}

/// Generation options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationOptions {
    /// Maximum number of tokens to generate
    pub max_tokens: Option<usize>,
    
    /// Temperature for sampling
    pub temperature: Option<f32>,
    
    /// Top-k sampling
    pub top_k: Option<usize>,
    
    /// Top-p (nucleus) sampling
    pub top_p: Option<f32>,
    
    /// Repetition penalty
    pub repetition_penalty: Option<f32>,
    
    /// Stop sequences
    pub stop_sequences: Option<Vec<String>>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: None,
            temperature: None,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            stop_sequences: None,
        }
    }
}

impl GenerationOptions {
    /// Merge with inference config
    pub fn merge_with_config(&self, config: &InferenceConfig) -> InferenceConfig {
        let mut merged = config.clone();
        
        if let Some(max_tokens) = self.max_tokens {
            merged.max_tokens = max_tokens;
        }
        if let Some(temperature) = self.temperature {
            merged.temperature = temperature;
        }
        if let Some(top_k) = self.top_k {
            merged.top_k = Some(top_k);
        }
        
        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;
    use crate::config::{ModelConfig, InferenceConfig};
    
    #[test]
    fn test_generation_options() {
        let options = GenerationOptions::default();
        let config = InferenceConfig::default();
        let merged = options.merge_with_config(&config);
        
        assert_eq!(merged.max_tokens, config.max_tokens);
    }
}
