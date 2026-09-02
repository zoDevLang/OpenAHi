//! Client for OpenAHI Rust SDK

use std::sync::Arc;
use std::time::Instant;

use openahi_runtime::{OpenAHIRuntime, RuntimeConfig, InferenceConfig};

use crate::config::{SdkConfig, GenerationConfig};
use crate::error::{SdkError, SdkResult};
use crate::model::{ModelInfo, ModelList, GenerationResult, BatchGenerationResult, FinishReason};
use crate::session::GenerationSession;

/// OpenAHI client
///
/// Main client for interacting with OpenAHI runtime
#[derive(Debug, Clone)]
pub struct OpenAHIClient {
    /// Runtime
    pub runtime: Arc<OpenAHIRuntime>,
    
    /// SDK configuration
    pub config: SdkConfig,
}

impl OpenAHIClient {
    /// Create a new OpenAHI client with default configuration
    pub fn new() -> SdkResult<Self> {
        Self::with_config(SdkConfig::default())
    }
    
    /// Create a new OpenAHI client with custom configuration
    pub fn with_config(config: SdkConfig) -> SdkResult<Self> {
        // Create runtime config
        let mut runtime_config = RuntimeConfig::default();
        
        // Apply SDK config to runtime config
        if let Some(ref path) = config.runtime_config {
            if path.exists() {
                runtime_config = RuntimeConfig::from_file(path)?;
            }
        }
        
        // Create runtime
        let runtime = Arc::new(OpenAHIRuntime::with_config(runtime_config)?);
        
        Ok(Self { runtime, config })
    }
    
    /// Create a new OpenAHI client with runtime configuration
    pub fn with_runtime_config(runtime_config: RuntimeConfig) -> SdkResult<Self> {
        let runtime = Arc::new(OpenAHIRuntime::with_config(runtime_config)?);
        let config = SdkConfig::default();
        
        Ok(Self { runtime, config })
    }
    
    /// Get runtime information
    pub fn info(&self) -> SdkResult<openahi_runtime::RuntimeInfo> {
        Ok(self.runtime.info())
    }
    
    /// List available models
    pub fn list_models(&self) -> SdkResult<ModelList> {
        let models = self.runtime.list_models();
        
        let model_infos = models.into_iter().map(|m| {
            ModelInfo {
                name: m.name,
                version: m.version,
                model_type: "transformer".to_string(),
                description: format!("{} model", m.name),
                creator: "ZoDev".to_string(),
                license: "Apache-2.0".to_string(),
                parameter_count: m.parameter_count,
                context_length: m.context_length,
                vocab_size: 0, // Will be updated
                created_at: "".to_string(),
                updated_at: "".to_string(),
                checksum: "".to_string(),
                tags: vec![],
            }
        }).collect();
        
        Ok(ModelList::new(model_infos))
    }
    
    /// Load a model
    pub fn load_model(&self, name: &str, version: &str) -> SdkResult<()> {
        self.runtime.load_model(name, version)?;
        Ok(())
    }
    
    /// Unload a model
    pub fn unload_model(&self, name: &str, version: &str) -> SdkResult<()> {
        self.runtime.unload_model(name, version)?;
        Ok(())
    }
    
    /// Generate text
    pub fn generate(&self, prompt: &str, config: Option<GenerationConfig>) -> SdkResult<GenerationResult> {
        let gen_config = config.unwrap_or_default();
        let runtime_config = gen_config.to_runtime_config();
        
        let start = Instant::now();
        
        let result = self.runtime.generate(
            &self.config.default_model,
            &self.config.default_model_version,
            prompt,
            Some(runtime_config),
        )?;
        
        let elapsed = start.elapsed().as_millis() as u64;
        
        Ok(GenerationResult {
            text: result,
            finish_reason: FinishReason::Complete,
            token_count: prompt.len() / 4, // Rough estimate
            generation_time: elapsed,
            model: self.config.default_model.clone(),
            version: self.config.default_model_version.clone(),
        })
    }
    
    /// Generate with specific model
    pub fn generate_with_model(
        &self,
        model: &str,
        version: &str,
        prompt: &str,
        config: Option<GenerationConfig>,
    ) -> SdkResult<GenerationResult> {
        let gen_config = config.unwrap_or_default();
        let runtime_config = gen_config.to_runtime_config();
        
        let start = Instant::now();
        
        let result = self.runtime.generate(
            model,
            version,
            prompt,
            Some(runtime_config),
        )?;
        
        let elapsed = start.elapsed().as_millis() as u64;
        
        Ok(GenerationResult {
            text: result,
            finish_reason: FinishReason::Complete,
            token_count: prompt.len() / 4,
            generation_time: elapsed,
            model: model.to_string(),
            version: version.to_string(),
        })
    }
    
    /// Batch generate
    pub fn batch_generate(
        &self,
        prompts: Vec<String>,
        config: Option<GenerationConfig>,
    ) -> SdkResult<BatchGenerationResult> {
        let gen_config = config.unwrap_or_default();
        let runtime_config = gen_config.to_runtime_config();
        
        let start = Instant::now();
        
        let mut results = Vec::new();
        for prompt in prompts {
            match self.runtime.generate(
                &self.config.default_model,
                &self.config.default_model_version,
                &prompt,
                Some(runtime_config.clone()),
            ) {
                Ok(text) => {
                    results.push(GenerationResult {
                        text,
                        finish_reason: FinishReason::Complete,
                        token_count: prompt.len() / 4,
                        generation_time: 0,
                        model: self.config.default_model.clone(),
                        version: self.config.default_model_version.clone(),
                    });
                }
                Err(e) => {
                    results.push(GenerationResult {
                        text: format!("Error: {}", e),
                        finish_reason: FinishReason::Error,
                        token_count: 0,
                        generation_time: 0,
                        model: self.config.default_model.clone(),
                        version: self.config.default_model_version.clone(),
                    });
                }
            }
        }
        
        let elapsed = start.elapsed().as_millis() as u64;
        
        Ok(BatchGenerationResult::new(results, elapsed))
    }
    
    /// Create a generation session
    pub fn create_session(&self, config: Option<GenerationConfig>) -> SdkResult<GenerationSession> {
        let gen_config = config.unwrap_or_default();
        
        GenerationSession::new(
            self.runtime.clone(),
            self.config.default_model.clone(),
            self.config.default_model_version.clone(),
            gen_config,
        )
    }
    
    /// Shutdown the client
    pub fn shutdown(&self) -> SdkResult<()> {
        self.runtime.shutdown()?;
        Ok(())
    }
}

impl Default for OpenAHIClient {
    fn default() -> Self {
        Self::new().expect("Failed to create OpenAHI client")
    }
}
