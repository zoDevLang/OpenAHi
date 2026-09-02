/**
 * Type definitions for OpenAHI TypeScript SDK
 */

/**
 * Configuration for OpenAHI client
 */
export interface OpenAHIConfig {
    /**
     * Model to use (format: "name@version" or just "name" for latest)
     */
    model?: string;
    
    /**
     * Base URL for OpenAHI runtime API
     * @default "http://localhost:8080"
     */
    baseUrl?: string;
    
    /**
     * Timeout in milliseconds
     * @default 30000
     */
    timeout?: number;
    
    /**
     * Maximum retries
     * @default 3
     */
    maxRetries?: number;
    
    /**
     * API key for authentication (optional)
     */
    apiKey?: string;
    
    /**
     * Debug mode
     * @default false
     */
    debug?: boolean;
}

/**
 * Configuration for text generation
 */
export interface GenerationConfig {
    /**
     * Maximum number of tokens to generate
     * @default 100
     */
    maxTokens?: number;
    
    /**
     * Temperature for sampling (0.0 to 2.0)
     * @default 1.0
     */
    temperature?: number;
    
    /**
     * Top-k sampling (number of top tokens to consider)
     */
    topK?: number;
    
    /**
     * Top-p (nucleus) sampling (0.0 to 1.0)
     */
    topP?: number;
    
    /**
     * Repetition penalty
     */
    repetitionPenalty?: number;
    
    /**
     * Stop sequences (generation stops when any of these is encountered)
     */
    stopSequences?: string[];
    
    /**
     * Whether to echo the input in the output
     * @default false
     */
    echo?: boolean;
    
    /**
     * Number of return sequences
     * @default 1
     */
    numReturnSequences?: number;
}

/**
 * Information about a model
 */
export interface ModelInfo {
    /**
     * Model name
     */
    name: string;
    
    /**
     * Model version
     */
    version: string;
    
    /**
     * Model type
     */
    modelType: string;
    
    /**
     * Model description
     */
    description: string;
    
    /**
     * Creator
     */
    creator: string;
    
    /**
     * License
     */
    license: string;
    
    /**
     * Number of parameters
     */
    parameterCount: number;
    
    /**
     * Context length (maximum sequence length)
     */
    contextLength: number;
    
    /**
     * Vocabulary size
     */
    vocabSize: number;
    
    /**
     * Creation date
     */
    createdAt?: string;
    
    /**
     * Last update date
     */
    updatedAt?: string;
    
    /**
     * Checksum
     */
    checksum?: string;
    
    /**
     * Tags
     */
    tags?: string[];
}

/**
 * Result of a generation request
 */
export interface GenerationResult {
    /**
     * Generated text
     */
    text: string;
    
    /**
     * Reason generation finished
     */
    finishReason: FinishReason;
    
    /**
     * Number of tokens generated
     */
    tokenCount: number;
    
    /**
     * Generation time in milliseconds
     */
    generationTime?: number;
    
    /**
     * Model used
     */
    model?: string;
    
    /**
     * Model version
     */
    version?: string;
}

/**
 * Reason generation finished
 */
export type FinishReason = 
    | 'complete'           // Generation completed normally
    | 'end_of_sequence'    // Stopped due to EOS token
    | 'max_tokens'         // Stopped due to max tokens
    | 'stop_sequence'      // Stopped due to stop sequence
    | 'error';             // Stopped due to error

/**
 * Response from listModels
 */
export interface ModelListResponse {
    models: ModelInfo[];
    total: number;
}

/**
 * Runtime information
 */
export interface RuntimeInfo {
    version: string;
    project: string;
    creator: string;
    loadedModels: number;
    maxLoadedModels: number;
    memoryUsage: number;
    uptime: number;
}

/**
 * Health check response
 */
export interface HealthCheckResponse {
    status: 'healthy' | 'unhealthy' | 'degraded';
    version: string;
    timestamp: string;
    checks: HealthCheck[];
}

/**
 * Individual health check
 */
export interface HealthCheck {
    name: string;
    status: 'passed' | 'failed' | 'skipped';
    message?: string;
}

/**
 * API error response
 */
export interface ApiError {
    error: string;
    message: string;
    code?: number;
    details?: any;
}

/**
 * Batch generation result
 */
export interface BatchGenerationResult {
    results: GenerationResult[];
    totalTime: number;
    successCount: number;
    failureCount: number;
}

/**
 * Session configuration
 */
export interface SessionConfig extends GenerationConfig {
    /**
     * Session ID (auto-generated if not provided)
     */
    sessionId?: string;
}

/**
 * Session information
 */
export interface SessionInfo {
    sessionId: string;
    model: string;
    version: string;
    createdAt: string;
    requestCount: number;
    totalTokens: number;
}
