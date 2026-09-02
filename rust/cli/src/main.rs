//! OpenAHI CLI
//!
//! Command-line interface for the OpenAHI ecosystem.
//!
//! Provides commands for:
//! - Model management (install, list, remove)
//! - Inference (run, generate)
//! - Information (info, models)
//! - Evaluation

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ArgGroup};
use log::{info, error, debug};

use openahi_runtime::{OpenAHIRuntime, RuntimeConfig, InferenceConfig, VERSION, PROJECT, CREATOR, DEFAULT_MODEL, DEFAULT_MODEL_VERSION};

/// OpenAHI - Open Artificial Hyper Intelligence
///
/// A model ecosystem for running and managing AI models locally.
#[derive(Parser, Debug)]
#[command(name = "openahi")]
#[command(author = "ZoDev")]
#[command(version = VERSION)]
#[command(about = "OpenAHI - Open Artificial Hyper Intelligence")]
#[command(long_about = "A model ecosystem for running and managing AI models locally.\n\nOpenAHI is a model ecosystem, not a chatbot.\nComposter is the first OpenAHI model.")]
#[command(group(
    ArgGroup::new("commands")
        .required(true)
        .args(["info", "models", "install", "run", "evaluate", "generate", "serve"]),
))]
struct Cli {
    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
    
    /// Enable debug output
    #[arg(short, long, global = true)]
    debug: bool,
    
    /// Path to configuration file
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,
    
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Display information about OpenAHI
    Info,
    
    /// List available models
    Models {
        /// Show all versions
        #[arg(short, long)]
        all: bool,
        
        /// Show detailed information
        #[arg(short, long)]
        detailed: bool,
    },
    
    /// Install a model
    Install {
        /// Model name (e.g., composter)
        model: String,
        
        /// Model version (e.g., 1.00.0)
        #[arg(short, long, default_value = DEFAULT_MODEL_VERSION)]
        version: String,
        
        /// Force reinstall
        #[arg(short, long)]
        force: bool,
    },
    
    /// Run a model (interactive mode)
    Run {
        /// Model name
        #[arg(short, long, default_value = DEFAULT_MODEL)]
        model: String,
        
        /// Model version
        #[arg(short, long, default_value = DEFAULT_MODEL_VERSION)]
        version: String,
        
        /// Temperature for sampling
        #[arg(short, long, default_value = "1.0")]
        temperature: f32,
        
        /// Maximum tokens to generate
        #[arg(short, long, default_value = "100")]
        max_tokens: usize,
    },
    
    /// Generate text from a prompt
    Generate {
        /// Input prompt
        prompt: String,
        
        /// Model name
        #[arg(short, long, default_value = DEFAULT_MODEL)]
        model: String,
        
        /// Model version
        #[arg(short, long, default_value = DEFAULT_MODEL_VERSION)]
        version: String,
        
        /// Temperature for sampling
        #[arg(short, long, default_value = "1.0")]
        temperature: f32,
        
        /// Maximum tokens to generate
        #[arg(short, long, default_value = "100")]
        max_tokens: usize,
        
        /// Top-k sampling
        #[arg(short, long)]
        top_k: Option<usize>,
        
        /// Output file (optional)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    
    /// Evaluate a model
    Evaluate {
        /// Model name
        #[arg(short, long, default_value = DEFAULT_MODEL)]
        model: String,
        
        /// Model version
        #[arg(short, long, default_value = DEFAULT_MODEL_VERSION)]
        version: String,
        
        /// Evaluation dataset (optional)
        #[arg(short, long)]
        dataset: Option<PathBuf>,
    },
    
    /// Start the OpenAHI server
    Serve {
        /// Server address
        #[arg(short, long, default_value = "127.0.0.1")]
        address: String,
        
        /// Server port
        #[arg(short, long, default_value = "8080")]
        port: u16,
        
        /// Enable CORS
        #[arg(long)]
        enable_cors: bool,
    },
}

fn main() -> Result<()> {
    // Parse CLI arguments
    let cli = Cli::parse();
    
    // Initialize logging
    init_logging(&cli);
    
    info!("OpenAHI CLI v{}", VERSION);
    debug!("Command: {:?}", cli.command);
    
    // Create runtime
    let mut runtime_config = RuntimeConfig::default();
    
    // Load custom config if specified
    if let Some(config_path) = &cli.config {
        if config_path.exists() {
            runtime_config = RuntimeConfig::from_file(config_path)
                .context("Failed to load configuration file")?;
        } else {
            error!("Configuration file not found: {}", config_path.display());
        }
    }
    
    let runtime = Arc::new(OpenAHIRuntime::with_config(runtime_config)?);
    
    // Execute command
    match &cli.command {
        Commands::Info => {
            cmd_info(&runtime);
        }
        Commands::Models { all, detailed } => {
            cmd_models(&runtime, *all, *detailed);
        }
        Commands::Install { model, version, force } => {
            cmd_install(&runtime, model, version, *force)?;
        }
        Commands::Run { model, version, temperature, max_tokens } => {
            cmd_run(&runtime, model, version, *temperature, *max_tokens)?;
        }
        Commands::Generate { prompt, model, version, temperature, max_tokens, top_k, output } => {
            cmd_generate(&runtime, prompt, model, version, *temperature, *max_tokens, *top_k, output)?;
        }
        Commands::Evaluate { model, version, dataset } => {
            cmd_evaluate(&runtime, model, version, dataset)?;
        }
        Commands::Serve { address, port, enable_cors } => {
            cmd_serve(&runtime, address, *port, *enable_cors)?;
        }
    }
    
    Ok(())
}

fn init_logging(cli: &Cli) {
    let mut builder = env_logger::Builder::new();
    
    if cli.debug {
        builder.filter_level(log::LevelFilter::Debug);
    } else if cli.verbose {
        builder.filter_level(log::LevelFilter::Info);
    } else {
        builder.filter_level(log::LevelFilter::Warn);
    }
    
    builder.init();
}

fn cmd_info(runtime: &Arc<OpenAHIRuntime>) {
    let info = runtime.info();
    
    println!("OpenAHI - Open Artificial Hyper Intelligence");
    println!();
    println!("Version: {}", info.version);
    println!("Project: {}", info.project);
    println!("Creator: {}", info.creator);
    println!();
    println!("Runtime:");
    println!("  Loaded models: {}", info.loaded_models);
    println!("  Max loaded models: {}", info.max_loaded_models);
    println!("  Memory usage: {} bytes", info.memory_usage);
    println!("  Uptime: {:?}", info.uptime);
    println!();
    println!("OpenAHI is a model ecosystem, not a chatbot.");
    println!("Composter is the first OpenAHI model.");
}

fn cmd_models(runtime: &Arc<OpenAHIRuntime>, all: bool, detailed: bool) {
    let models = runtime.list_models();
    
    if models.is_empty() {
        println!("No models loaded.");
        return;
    }
    
    if detailed {
        for model in &models {
            println!("Model: {}@{}", model.name, model.version);
            println!("  Status: {:?}", model.status);
            println!("  Parameters: {}", model.parameter_count);
            println!("  Context length: {}", model.context_length);
            println!("  Memory usage: {} bytes", model.memory_usage);
            println!("  Load time: {} ms", model.load_time);
            println!("  Usage count: {}", model.usage_count);
            println!();
        }
    } else {
        for model in &models {
            println!("{}@{}", model.name, model.version);
        }
    }
}

fn cmd_install(runtime: &Arc<OpenAHIRuntime>, model: &str, version: &str, force: bool) -> Result<()> {
    info!("Installing model: {}@{}", model, version);
    
    // Check if already installed
    if !force {
        let models = runtime.list_models();
        for m in &models {
            if m.name == model && m.version == version {
                println!("Model {}@{}" is already installed.", model, version);
                return Ok(());
            }
        }
    }
    
    // Load the model
    info!("Loading model: {}@{}", model, version);
    let _ = runtime.load_model(model, version)?;
    
    println!("Successfully installed {}@{}", model, version);
    
    Ok(())
}

fn cmd_run(
    runtime: &Arc<OpenAHIRuntime>,
    model: &str,
    version: &str,
    temperature: f32,
    max_tokens: usize,
) -> Result<()> {
    info!("Running model: {}@{}", model, version);
    
    // Load model if not already loaded
    let _ = runtime.load_model(model, version)?;
    
    let inference_config = InferenceConfig::default()
        .with_temperature(temperature)
        .with_max_tokens(max_tokens);
    
    println!("OpenAHI Interactive Mode");
    println!("Model: {}@{}", model, version);
    println!("Type 'quit' or 'exit' to end the session.");
    println!();
    
    use std::io::{self, Write, BufRead, BufReader};
    
    let stdin = io::stdin();
    let reader = BufReader::new(stdin);
    
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        
        if line.is_empty() {
            continue;
        }
        
        if line.eq_ignore_ascii_case("quit") || line.eq_ignore_ascii_case("exit") {
            break;
        }
        
        // Generate response
        let start = std::time::Instant::now();
        let result = runtime.generate(model, version, line, Some(inference_config.clone()));
        let elapsed = start.elapsed();
        
        match result {
            Ok(output) => {
                println!("\n{}", output);
                println!("\nTime: {:?}", elapsed);
            }
            Err(e) => {
                error!("Generation error: {}", e);
                println!("Error: {}", e);
            }
        }
        
        print!("\n> ");
        io::stdout().flush()?;
    }
    
    println!("Goodbye!");
    
    Ok(())
}

fn cmd_generate(
    runtime: &Arc<OpenAHIRuntime>,
    prompt: &str,
    model: &str,
    version: &str,
    temperature: f32,
    max_tokens: usize,
    top_k: Option<usize>,
    output: &Option<PathBuf>,
) -> Result<()> {
    info!("Generating text with model: {}@{}", model, version);
    
    // Load model if not already loaded
    let _ = runtime.load_model(model, version)?;
    
    let mut inference_config = InferenceConfig::default()
        .with_temperature(temperature)
        .with_max_tokens(max_tokens);
    
    if let Some(k) = top_k {
        inference_config = inference_config.with_top_k(k);
    }
    
    let start = std::time::Instant::now();
    let result = runtime.generate(model, version, prompt, Some(inference_config));
    let elapsed = start.elapsed();
    
    match result {
        Ok(output) => {
            if let Some(output_path) = output {
                std::fs::write(output_path, &output)?;
                println!("Generated text saved to: {}", output_path.display());
            } else {
                println!("{}", output);
            }
            println!("Time: {:?}", elapsed);
        }
        Err(e) => {
            error!("Generation error: {}", e);
            return Err(e);
        }
    }
    
    Ok(())
}

fn cmd_evaluate(
    runtime: &Arc<OpenAHIRuntime>,
    model: &str,
    version: &str,
    dataset: &Option<PathBuf>,
) -> Result<()> {
    info!("Evaluating model: {}@{}", model, version);
    
    // Load model if not already loaded
    let _ = runtime.load_model(model, version)?;
    
    if let Some(dataset_path) = dataset {
        println!("Evaluating on dataset: {}", dataset_path.display());
        // In a real implementation, this would load the dataset and run evaluation
        println!("Evaluation not yet implemented for custom datasets.");
    } else {
        println!("Running basic evaluation...");
        // Run a simple test
        let result = runtime.generate(model, version, "The quick brown fox", None);
        match result {
            Ok(output) => {
                println!("Test generation: {}", output);
                println!("Evaluation complete.");
            }
            Err(e) => {
                error!("Evaluation error: {}", e);
                return Err(e);
            }
        }
    }
    
    Ok(())
}

fn cmd_serve(
    runtime: &Arc<OpenAHIRuntime>,
    address: &str,
    port: u16,
    enable_cors: bool,
) -> Result<()> {
    info!("Starting OpenAHI server on {}:{}", address, port);
    
    // In a real implementation, this would start an HTTP server
    // For now, we'll just print a message
    
    println!("OpenAHI Server");
    println!("Address: {}", address);
    println!("Port: {}", port);
    println!("CORS enabled: {}", enable_cors);
    println!();
    println!("Server not yet implemented. Use the CLI for now.");
    
    Ok(())
}
