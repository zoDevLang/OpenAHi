/**
 * Error types for OpenAHI TypeScript SDK
 */

/**
 * Custom error class for OpenAHI SDK
 */
export class OpenAHIError extends Error {
    /**
     * Error code
     */
    public readonly code: string;
    
    /**
     * HTTP status code (if applicable)
     */
    public readonly statusCode?: number;
    
    /**
     * Additional details
     */
    public readonly details?: any;
    
    /**
     * Create a new OpenAHI error
     * @param message Error message
     * @param code Error code
     * @param statusCode HTTP status code
     * @param details Additional details
     */
    constructor(
        message: string,
        code: string = 'OPENHI_ERROR',
        statusCode?: number,
        details?: any
    ) {
        super(message);
        
        this.name = 'OpenAHIError';
        this.code = code;
        this.statusCode = statusCode;
        this.details = details;
        
        // Maintain proper stack trace
        if (Error.captureStackTrace) {
            Error.captureStackTrace(this, OpenAHIError);
        }
    }
    
    /**
     * Create a connection error
     */
    static connectionError(message: string, details?: any): OpenAHIError {
        return new OpenAHIError(
            message,
            'CONNECTION_ERROR',
            undefined,
            details
        );
    }
    
    /**
     * Create a timeout error
     */
    static timeoutError(message: string = 'Request timed out'): OpenAHIError {
        return new OpenAHIError(message, 'TIMEOUT_ERROR', 408);
    }
    
    /**
     * Create a model not found error
     */
    static modelNotFound(model: string): OpenAHIError {
        return new OpenAHIError(
            `Model not found: ${model}`,
            'MODEL_NOT_FOUND',
            404
        );
    }
    
    /**
     * Create a generation error
     */
    static generationError(message: string, details?: any): OpenAHIError {
        return new OpenAHIError(
            message,
            'GENERATION_ERROR',
            undefined,
            details
        );
    }
    
    /**
     * Create a validation error
     */
    static validationError(message: string, details?: any): OpenAHIError {
        return new OpenAHIError(
            message,
            'VALIDATION_ERROR',
            400,
            details
        );
    }
    
    /**
     * Create an authentication error
     */
    static authenticationError(message: string = 'Authentication failed'): OpenAHIError {
        return new OpenAHIError(message, 'AUTHENTICATION_ERROR', 401);
    }
    
    /**
     * Create a rate limit error
     */
    static rateLimitError(message: string = 'Rate limit exceeded'): OpenAHIError {
        return new OpenAHIError(message, 'RATE_LIMIT_ERROR', 429);
    }
    
    /**
     * Create a server error
     */
    static serverError(message: string = 'Internal server error'): OpenAHIError {
        return new OpenAHIError(message, 'SERVER_ERROR', 500);
    }
    
    /**
     * Check if error is a connection error
     */
    isConnectionError(): boolean {
        return this.code === 'CONNECTION_ERROR';
    }
    
    /**
     * Check if error is a timeout error
     */
    isTimeoutError(): boolean {
        return this.code === 'TIMEOUT_ERROR';
    }
    
    /**
     * Check if error is a model not found error
     */
    isModelNotFound(): boolean {
        return this.code === 'MODEL_NOT_FOUND';
    }
    
    /**
     * Check if error is a generation error
     */
    isGenerationError(): boolean {
        return this.code === 'GENERATION_ERROR';
    }
    
    /**
     * Check if error is a validation error
     */
    isValidationError(): boolean {
        return this.code === 'VALIDATION_ERROR';
    }
    
    /**
     * Convert to plain object
     */
    toJSON(): any {
        return {
            name: this.name,
            message: this.message,
            code: this.code,
            statusCode: this.statusCode,
            details: this.details,
        };
    }
}

/**
 * Check if an error is an OpenAHI error
 */
export function isOpenAHIError(error: any): error is OpenAHIError {
    return error instanceof OpenAHIError || 
           (error && typeof error === 'object' && error.name === 'OpenAHIError');
}
