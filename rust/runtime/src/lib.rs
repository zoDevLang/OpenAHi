//! OpenAHI Runtime
//!
//! The OpenAHI runtime provides model loading, inference orchestration,
//! and resource management for the OpenAHI ecosystem.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod config;
pub mod error;
pub mod model;
pub mod runtime;
pub mod inference;

// Re-export main types
pub use config::{RuntimeConfig, ModelConfig, InferenceConfig};
pub use error::{RuntimeError, RuntimeResult};
pub use model::{Model, ModelMetadata, ModelStatus};
pub use runtime::OpenAHIRuntime;
pub use inference::InferenceSession;

/// Version of the OpenAHI runtime
pub const VERSION: &str = "0.1.0";

/// Project name
pub const PROJECT: &str = "OpenAHI";

/// Creator
pub const CREATOR: &str = "ZoDev";

/// Default model name
pub const DEFAULT_MODEL: &str = "composter";

/// Default model version
pub const DEFAULT_MODEL_VERSION: &str = "1.00.0";

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version() {
        assert_eq!(VERSION, "0.1.0");
    }
    
    #[test]
    fn test_project() {
        assert_eq!(PROJECT, "OpenAHI");
    }
}
