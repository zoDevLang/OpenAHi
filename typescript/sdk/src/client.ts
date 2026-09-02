/**
 * OpenAHI TypeScript SDK Client
 * 
 * Main client class for communicating with OpenAHI runtime
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios';
import { OpenAHIConfig, GenerationConfig, ModelInfo, GenerationResult, FinishReason, ModelListResponse, RuntimeInfo, HealthCheckResponse, BatchGenerationResult } from './types';
import { OpenAHIError } from './error';

/**
 * Default configuration
 */
const DEFAULT_CONFIG: Required<OpenAHIConfig> = {
    model: 'composter@1.00.0',
    baseUrl: 'http://localhost:8080',
    timeout: 30000,
    maxRetries: 3,
    apiKey: undefined,
    debug: false,
};

/**
 * OpenAHI client
 * 
 * Provides a TypeScript interface for communicating with a local OpenAHI runtime.
 * 
 * @example
 * ```typescript
 * const ahi = new OpenAHI({
 *     model: 'composter@1.00.0'
 * });
 * 
 * const result = await ahi.generate('Hello, world!');
 * console.log(result);
 * ```
 */
export class OpenAHI {
    /**
     * Configuration
     */
    public readonly config: Required<OpenAHIConfig>;
    
    /**
     * Axios instance for HTTP requests
     */
    private readonly axios: AxiosInstance;
    
    /**
     * Request counter for generating IDs
     */
    private requestCounter: number = 0;
    
    /**
     * Create a new OpenAHI client
     * @param config Configuration options
     */
    constructor(config: OpenAHIConfig = {}) {
        this.config = { ...DEFAULT_CONFIG, ...config };
        
        // Create Axios instance
        this.axios = axios.create({
            baseURL: this.config.baseUrl,
            timeout: this.config.timeout,
            headers: {
                'Content-Type': 'application/json',
                ...(this.config.apiKey ? { 'Authorization': `Bearer ${this.config.apiKey}` } : {}),
            },
        });
        
        // Add request interceptor for logging
        if (this.config.debug) {
            this.axios.interceptors.request.use((config) => {
                console.log(`[OpenAHI] Request: ${config.method?.toUpperCase()} ${config.url}`);
                return config;
            });
            
            this.axios.interceptors.response.use(
                (response) => {
                    console.log(`[OpenAHI] Response: ${response.status} ${response.statusText}`);
                    return response;
                },
                (error) => {
                    console.error(`[OpenAHI] Error: ${error.message}`);
                    return Promise.reject(error);
                }
            );
        }
    }
    
    /**
     * Generate a unique request ID
     */
    private generateRequestId(): string {
        return `req_${Date.now()}_${++this.requestCounter}`;
    }
    
    /**
     * Execute a request with retries
     */
    private async requestWithRetry<T>(
        config: AxiosRequestConfig,
        retries: number = this.config.maxRetries
    ): Promise<AxiosResponse<T>> {
        try {
            return await this.axios.request<T>(config);
        } catch (error) {
            if (retries <= 0) {
                throw this.handleAxiosError(error as AxiosError);
            }
            
            // Retry after a delay
            await new Promise((resolve) => setTimeout(resolve, 1000));
            return this.requestWithRetry(config, retries - 1);
        }
    }
    
    /**
     * Handle Axios errors
     */
    private handleAxiosError(error: AxiosError): Error {
        if (!error.response) {
            // Network error
            return OpenAHIError.connectionError(
                error.message || 'Network error',
                { url: error.config?.url }
            );
        }
        
        const status = error.response.status;
        const data = error.response.data as any;
        
        switch (status) {
            case 400:
                return OpenAHIError.validationError(
                    data?.message || 'Bad request',
                    data
                );
            case 401:
                return OpenAHIError.authenticationError(
                    data?.message || 'Authentication failed'
                );
            case 404:
                return OpenAHIError.modelNotFound(
                    data?.message || 'Not found'
                );
            case 429:
                return OpenAHIError.rateLimitError(
                    data?.message || 'Rate limit exceeded'
                );
            case 500:
            case 502:
            case 503:
            case 504:
                return OpenAHIError.serverError(
                    data?.message || 'Server error'
                );
            default:
                return OpenAHIError.serverError(
                    data?.message || error.message || 'Unknown error'
                );
        }
    }
    
    /**
     * Check if the runtime is available
     */
    async checkRuntime(): Promise<HealthCheckResponse> {
        try {
            const response = await this.requestWithRetry<HealthCheckResponse>({
                method: 'GET',
                url: '/health',
            });
            return response.data;
        } catch (error) {
            throw OpenAHIError.connectionError(
                'Failed to connect to OpenAHI runtime',
                { error: (error as Error).message }
            );
        }
    }
    
    /**
     * Get runtime information
     */
    async getRuntimeInfo(): Promise<RuntimeInfo> {
        const response = await this.requestWithRetry<RuntimeInfo>({
            method: 'GET',
            url: '/api/info',
        });
        return response.data;
    }
    
    /**
     * List available models
     */
    async listModels(): Promise<ModelListResponse> {
        const response = await this.requestWithRetry<ModelListResponse>({
            method: 'GET',
            url: '/api/models',
        });
        return response.data;
    }
    
    /**
     * Get information about a specific model
     */
    async getModelInfo(name: string, version?: string): Promise<ModelInfo> {
        const url = version 
            ? `/api/models/${encodeURIComponent(name)}/${encodeURIComponent(version)}`
            : `/api/models/${encodeURIComponent(name)}`;
        
        const response = await this.requestWithRetry<ModelInfo>({
            method: 'GET',
            url,
        });
        return response.data;
    }
    
    /**
     * Load a model
     */
    async loadModel(name: string, version?: string): Promise<void> {
        const url = version 
            ? `/api/models/${encodeURIComponent(name)}/${encodeURIComponent(version)}/load`
            : `/api/models/${encodeURIComponent(name)}/load`;
        
        await this.requestWithRetry<void>({
            method: 'POST',
            url,
        });
    }
    
    /**
     * Unload a model
     */
    async unloadModel(name: string, version?: string): Promise<void> {
        const url = version 
            ? `/api/models/${encodeURIComponent(name)}/${encodeURIComponent(version)}/unload`
            : `/api/models/${encodeURIComponent(name)}/unload`;
        
        await this.requestWithRetry<void>({
            method: 'POST',
            url,
        });
    }
    
    /**
     * Generate text
     * 
     * @param prompt Input prompt
     * @param config Generation configuration
     * @returns Generation result
     */
    async generate(
        prompt: string,
        config?: GenerationConfig
    ): Promise<GenerationResult> {
        const requestId = this.generateRequestId();
        const fullConfig: GenerationConfig = {
            maxTokens: 100,
            temperature: 1.0,
            echo: false,
            ...config,
        };
        
        try {
            const startTime = Date.now();
            
            const response = await this.requestWithRetry<GenerationResult>({
                method: 'POST',
                url: '/api/generate',
                data: {
                    model: this.config.model,
                    prompt,
                    ...fullConfig,
                },
            });
            
            const result = response.data;
            result.generationTime = Date.now() - startTime;
            result.model = this.config.model;
            result.version = this.config.model.split('@')[1] || '1.00.0';
            
            return result;
        } catch (error) {
            throw OpenAHIError.generationError(
                `Failed to generate text: ${(error as Error).message}`,
                { requestId, prompt }
            );
        }
    }
    
    /**
     * Generate text with a specific model
     * 
     * @param model Model name and version (e.g., "composter@1.00.0")
     * @param prompt Input prompt
     * @param config Generation configuration
     * @returns Generation result
     */
    async generateWithModel(
        model: string,
        prompt: string,
        config?: GenerationConfig
    ): Promise<GenerationResult> {
        const requestId = this.generateRequestId();
        const fullConfig: GenerationConfig = {
            maxTokens: 100,
            temperature: 1.0,
            echo: false,
            ...config,
        };
        
        try {
            const startTime = Date.now();
            
            const response = await this.requestWithRetry<GenerationResult>({
                method: 'POST',
                url: '/api/generate',
                data: {
                    model,
                    prompt,
                    ...fullConfig,
                },
            });
            
            const result = response.data;
            result.generationTime = Date.now() - startTime;
            result.model = model.split('@')[0];
            result.version = model.split('@')[1] || '1.00.0';
            
            return result;
        } catch (error) {
            throw OpenAHIError.generationError(
                `Failed to generate text: ${(error as Error).message}`,
                { requestId, model, prompt }
            );
        }
    }
    
    /**
     * Batch generate text
     * 
     * @param prompts Array of prompts
     * @param config Generation configuration
     * @returns Batch generation result
     */
    async batchGenerate(
        prompts: string[],
        config?: GenerationConfig
    ): Promise<BatchGenerationResult> {
        const fullConfig: GenerationConfig = {
            maxTokens: 100,
            temperature: 1.0,
            echo: false,
            ...config,
        };
        
        try {
            const startTime = Date.now();
            
            const response = await this.requestWithRetry<BatchGenerationResult>({
                method: 'POST',
                url: '/api/generate/batch',
                data: {
                    model: this.config.model,
                    prompts,
                    ...fullConfig,
                },
            });
            
            const result = response.data;
            result.totalTime = Date.now() - startTime;
            
            return result;
        } catch (error) {
            throw OpenAHIError.generationError(
                `Failed to batch generate text: ${(error as Error).message}`,
                { prompts: prompts.slice(0, 5) } // Limit logged prompts
            );
        }
    }
    
    /**
     * Stream generation (not yet implemented in runtime)
     * 
     * This method will be available when the runtime supports streaming.
     */
    async *streamGenerate(
        prompt: string,
        config?: GenerationConfig
    ): AsyncGenerator<string> {
        // TODO: Implement when runtime supports streaming
        throw OpenAHIError.notImplementedError(
            'Streaming generation is not yet implemented'
        );
    }
    
    /**
     * Get completion (alias for generate)
     */
    async getCompletion(prompt: string, config?: GenerationConfig): Promise<GenerationResult> {
        return this.generate(prompt, { ...config, echo: true });
    }
    
    /**
     * Close the client
     */
    async close(): Promise<void> {
        // Clean up resources
    }
}

/**
 * Not implemented error (static method)
 */
namespace OpenAHI {
    export function notImplementedError(message: string): OpenAHIError {
        return new OpenAHIError(message, 'NOT_IMPLEMENTED');
    }
}
