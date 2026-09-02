/**
 * OpenAHI TypeScript SDK
 * 
 * A TypeScript SDK that allows applications to communicate with a locally running
 * OpenAHI runtime.
 * 
 * Example:
 * ```typescript
 * import { OpenAHI } from "@openahi/sdk";
 * 
 * const ahi = new OpenAHI({
 *     model: "composter@1.00.0"
 * });
 * 
 * const result = await ahi.generate("Hello");
 * 
 * console.log(result);
 * ```
 * 
 * Note: This SDK communicates with the local OpenAHI runtime, not an external AI provider.
 */

// Export main classes and types
export { OpenAHI } from './client';
export { OpenAHIConfig, GenerationConfig, ModelInfo, GenerationResult, FinishReason } from './types';
export { OpenAHIError } from './error';

// Export utility functions
export { listModels, getModelInfo, checkRuntime } from './utils';
