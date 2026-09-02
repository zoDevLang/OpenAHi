//! OpenAHI Rust SDK
//!
//! Native Rust SDK for OpenAHI applications.
//!
//! Provides a clean API for:
//! - Loading models
//! - Configuring inference
//! - Generating output
//! - Handling errors

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod client;
pub mod config;
pub mod error;
pub mod model;
pub mod session;

// Re-export main types
pub use client::OpenAHIClient;
pub use config::{SdkConfig, GenerationConfig};
pub use error::{SdkError, SdkResult};
pub use model::{ModelInfo, ModelList};
pub use session::GenerationSession;

/// Version of the OpenAHI Rust SDK
pub const VERSION: &str = "0.1.0";

/// Project name
pub const PROJECT: &str = "OpenAHI";

/// Creator
pub const CREATOR: &str = "ZoDev";

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
