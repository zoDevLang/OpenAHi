/**
 * Utility functions for OpenAHI TypeScript SDK
 */

import { ModelInfo, ModelListResponse, RuntimeInfo } from './types';
import { OpenAHI } from './client';

/**
 * Default OpenAHI client instance
 */
let defaultClient: OpenAHI | null = null;

/**
 * Get the default OpenAHI client
 */
export function getDefaultClient(): OpenAHI {
    if (!defaultClient) {
        defaultClient = new OpenAHI();
    }
    return defaultClient;
}

/**
 * Set the default OpenAHI client
 */
export function setDefaultClient(client: OpenAHI): void {
    defaultClient = client;
}

/**
 * Reset the default client
 */
export function resetDefaultClient(): void {
    defaultClient = null;
}

/**
 * List all available models
 */
export async function listModels(baseUrl?: string): Promise<ModelListResponse> {
    const client = baseUrl ? new OpenAHI({ baseUrl }) : getDefaultClient();
    return client.listModels();
}

/**
 * Get information about a specific model
 */
export async function getModelInfo(name: string, version?: string, baseUrl?: string): Promise<ModelInfo> {
    const client = baseUrl ? new OpenAHI({ baseUrl }) : getDefaultClient();
    return client.getModelInfo(name, version);
}

/**
 * Check if the runtime is available
 */
export async function checkRuntime(baseUrl?: string): Promise<boolean> {
    const client = baseUrl ? new OpenAHI({ baseUrl }) : getDefaultClient();
    try {
        const health = await client.checkRuntime();
        return health.status === 'healthy';
    } catch {
        return false;
    }
}

/**
 * Get runtime information
 */
export async function getRuntimeInfo(baseUrl?: string): Promise<RuntimeInfo | null> {
    const client = baseUrl ? new OpenAHI({ baseUrl }) : getDefaultClient();
    try {
        return await client.getRuntimeInfo();
    } catch {
        return null;
    }
}

/**
 * Generate text with the default client
 */
export async function generate(
    prompt: string,
    config?: any,
    baseUrl?: string
): Promise<string> {
    const client = baseUrl ? new OpenAHI({ baseUrl }) : getDefaultClient();
    const result = await client.generate(prompt, config);
    return result.text;
}

/**
 * Create a simple generation function
 * 
 * @example
 * ```typescript
 * const generate = createGenerator({ model: 'composter@1.00.0' });
 * const result = await generate('Hello');
 * ```
 */
export function createGenerator(config?: any, baseUrl?: string): (prompt: string) => Promise<string> {
    const client = baseUrl ? new OpenAHI({ baseUrl, ...config }) : new OpenAHI(config);
    
    return async (prompt: string) => {
        const result = await client.generate(prompt);
        return result.text;
    };
}

/**
 * Create a batch generator
 */
export function createBatchGenerator(config?: any, baseUrl?: string): (prompts: string[]) => Promise<string[]> {
    const client = baseUrl ? new OpenAHI({ baseUrl, ...config }) : new OpenAHI(config);
    
    return async (prompts: string[]) => {
        const result = await client.batchGenerate(prompts);
        return result.results.map(r => r.text);
    };
}

/**
 * Sleep utility
 */
export function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Retry utility
 */
export async function retry<T>(
    fn: () => Promise<T>,
    options: { retries?: number; delay?: number; onRetry?: (error: Error, attempt: number) => void } = {}
): Promise<T> {
    const { retries = 3, delay = 1000, onRetry } = options;
    
    let lastError: Error | undefined;
    
    for (let attempt = 1; attempt <= retries; attempt++) {
        try {
            return await fn();
        } catch (error) {
            lastError = error as Error;
            
            if (attempt < retries) {
                if (onRetry) {
                    onRetry(lastError, attempt);
                }
                await sleep(delay);
            }
        }
    }
    
    throw lastError;
}

/**
 * Timeout utility
 */
export function withTimeout<T>(
    promise: Promise<T>,
    ms: number,
    message: string = 'Request timed out'
): Promise<T> {
    return Promise.race([
        promise,
        new Promise<T>((_, reject) => {
            setTimeout(() => reject(new Error(message)), ms);
        }),
    ]);
}
