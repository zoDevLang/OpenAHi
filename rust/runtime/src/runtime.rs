//! OpenAHI Runtime
//!
//! The main runtime class that manages model loading, inference orchestration,
//! and resource management.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime};

use crate::config::{RuntimeConfig, ModelConfig, InferenceConfig};
use crate::error::{RuntimeError, RuntimeResult};
use crate::model::{Model, ModelInfo, ModelRegistry, ModelStatus};

/// OpenAHI Runtime
///
/// The main runtime class that provides:
/// - Model loading and management
/// - Inference orchestration
/// - Resource management
/// - Configuration management
#[derive(Debug)]
pub struct OpenAHIRuntime {
    /// Runtime configuration
    pub config: RuntimeConfig,
    
    /// Model registry
    pub registry: ModelRegistry,
    
    /// Loaded models
    pub loaded_models: RwLock<HashMap<String, Arc<Mutex<Model>>>>,
    
    /// Resource manager
    pub resource_manager: ResourceManager,
    
    /// Model cache
    pub cache: ModelCache,
}

impl OpenAHIRuntime {
    /// Create a new OpenAHI runtime with default configuration
    pub fn new() -> RuntimeResult<Self> {
        Self::with_config(RuntimeConfig::default())
    }
    
    /// Create a new OpenAHI runtime with custom configuration
    pub fn with_config(config: RuntimeConfig) -> RuntimeResult<Self> {
        // Create directories if they don't exist
        std::fs::create_dir_all(&config.model_dir)?;
        std::fs::create_dir_all(&config.cache_dir)?;
        
        Ok(Self {
            config,
            registry: ModelRegistry::new(),
            loaded_models: RwLock::new(HashMap::new()),
            resource_manager: ResourceManager::new(),
            cache: ModelCache::new(config.cache_dir.clone()),
        })
    }
    
    /// Load a model by name and version
    pub fn load_model(&self, name: &str, version: &str) -> RuntimeResult<Arc<Mutex<Model>>> {
        let key = format!("{}@{}", name, version);
        
        // Check if already loaded
        {
            let models = self.loaded_models.read().unwrap();
            if let Some(model) = models.get(&key) {
                return Ok(model.clone());
            }
        }
        
        // Check cache
        if let Some(model) = self.cache.get(&key)? {
            let mut models = self.loaded_models.write().unwrap();
            let model_arc = Arc::new(Mutex::new(model));
            models.insert(key.clone(), model_arc.clone());
            return Ok(model_arc);
        }
        
        // Create new model
        let model_config = ModelConfig::new(name.to_string(), version.to_string());
        let mut model = Model::new(model_config);
        
        // Load the model
        model.load()?;
        
        // Register in registry
        self.registry.register(model.info().into());
        
        // Add to cache
        self.cache.put(key.clone(), model.clone())?;
        
        // Add to loaded models
        let model_arc = Arc::new(Mutex::new(model));
        {
            let mut models = self.loaded_models.write().unwrap();
            models.insert(key, model_arc.clone());
        }
        
        Ok(model_arc)
    }
    
    /// Get a loaded model
    pub fn get_model(&self, name: &str, version: &str) -> RuntimeResult<Arc<Mutex<Model>>> {
        let key = format!("{}@{}", name, version);
        
        let models = self.loaded_models.read().unwrap();
        models.get(&key)
            .cloned()
            .ok_or_else(|| RuntimeError::ModelNotFound(key))
    }
    
    /// Unload a model
    pub fn unload_model(&self, name: &str, version: &str) -> RuntimeResult<()> {
        let key = format!("{}@{}", name, version);
        
        let mut models = self.loaded_models.write().unwrap();
        if let Some(model) = models.remove(&key) {
            let mut model_guard = model.lock().unwrap();
            model_guard.unload();
            Ok(())
        } else {
            Err(RuntimeError::ModelNotFound(key))
        }
    }
    
    /// List all loaded models
    pub fn list_models(&self) -> Vec<ModelInfo> {
        let models = self.loaded_models.read().unwrap();
        models.values()
            .map(|m| {
                let guard = m.lock().unwrap();
                guard.info()
            })
            .collect()
    }
    
    /// Generate text using a model
    pub fn generate(
        &self,
        model_name: &str,
        model_version: &str,
        prompt: &str,
        inference_config: Option<InferenceConfig>,
    ) -> RuntimeResult<String> {
        let model = self.get_model(model_name, model_version)?;
        let config = inference_config.unwrap_or_default();
        
        let mut model_guard = model.lock().unwrap();
        model_guard.generate(prompt, &config)
    }
    
    /// Get runtime information
    pub fn info(&self) -> RuntimeInfo {
        RuntimeInfo {
            version: crate::VERSION.to_string(),
            project: crate::PROJECT.to_string(),
            creator: crate::CREATOR.to_string(),
            loaded_models: self.list_models().len(),
            max_loaded_models: self.config.max_loaded_models,
            memory_usage: self.resource_manager.memory_usage(),
            uptime: self.resource_manager.uptime(),
        }
    }
    
    /// Shutdown the runtime
    pub fn shutdown(&self) -> RuntimeResult<()> {
        // Unload all models
        let mut models = self.loaded_models.write().unwrap();
        for (_, model) in models.drain() {
            let mut model_guard = model.lock().unwrap();
            model_guard.unload();
        }
        
        // Clear cache
        self.cache.clear()?;
        
        Ok(())
    }
}

/// Runtime information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeInfo {
    pub version: String,
    pub project: String,
    pub creator: String,
    pub loaded_models: usize,
    pub max_loaded_models: usize,
    pub memory_usage: u64,
    pub uptime: Duration,
}

/// Resource manager
#[derive(Debug, Default)]
pub struct ResourceManager {
    /// Start time
    start_time: SystemTime,
    
    /// Memory usage tracking
    memory_usage: u64,
}

impl ResourceManager {
    /// Create a new resource manager
    pub fn new() -> Self {
        Self {
            start_time: SystemTime::now(),
            memory_usage: 0,
        }
    }
    
    /// Get uptime
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed().unwrap_or(Duration::from_secs(0))
    }
    
    /// Get memory usage
    pub fn memory_usage(&self) -> u64 {
        self.memory_usage
    }
    
    /// Update memory usage
    pub fn update_memory_usage(&mut self, delta: i64) {
        if delta >= 0 {
            self.memory_usage += delta as u64;
        } else {
            self.memory_usage = self.memory_usage.saturating_sub((-delta) as u64);
        }
    }
}

/// Model cache
#[derive(Debug)]
pub struct ModelCache {
    /// Cache directory
    cache_dir: PathBuf,
    
    /// Cached models
    cached_models: RwLock<HashMap<String, Model>>,
}

impl ModelCache {
    /// Create a new model cache
    pub fn new(cache_dir: PathBuf) -> Self {
        std::fs::create_dir_all(&cache_dir).unwrap_or_default();
        
        Self {
            cache_dir,
            cached_models: RwLock::new(HashMap::new()),
        }
    }
    
    /// Get a model from cache
    pub fn get(&self, key: &str) -> RuntimeResult<Option<Model>> {
        let cache = self.cached_models.read().unwrap();
        Ok(cache.get(key).cloned())
    }
    
    /// Put a model in cache
    pub fn put(&self, key: String, model: Model) -> RuntimeResult<()> {
        let mut cache = self.cached_models.write().unwrap();
        cache.insert(key, model);
        Ok(())
    }
    
    /// Remove a model from cache
    pub fn remove(&self, key: &str) -> RuntimeResult<()> {
        let mut cache = self.cached_models.write().unwrap();
        cache.remove(key);
        Ok(())
    }
    
    /// Clear the cache
    pub fn clear(&self) -> RuntimeResult<()> {
        let mut cache = self.cached_models.write().unwrap();
        cache.clear();
        Ok(())
    }
    
    /// Get cache directory
    pub fn cache_dir(&self) -> &PathBuf {
        &self.cache_dir
    }
}

/// Default implementation for ModelInfo
impl From<ModelInfo> for Model {
    fn from(info: ModelInfo) -> Self {
        let mut config = ModelConfig::new(info.name.clone(), info.version.clone());
        config.architecture.context_length = info.context_length;
        
        Model {
            config,
            metadata: ModelMetadata::composter_1_00_0(),
            status: info.status,
            weights: None,
            tokenizer: None,
            inference_engine: None,
            memory_usage: info.memory_usage,
            load_time: info.load_time,
            last_used: SystemTime::now(),
            usage_count: info.usage_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_runtime_creation() {
        let runtime = OpenAHIRuntime::new().unwrap();
        assert_eq!(runtime.info().project, "OpenAHI");
    }
    
    #[test]
    fn test_runtime_info() {
        let runtime = OpenAHIRuntime::new().unwrap();
        let info = runtime.info();
        assert_eq!(info.version, "0.1.0");
        assert_eq!(info.creator, "ZoDev");
    }
}
