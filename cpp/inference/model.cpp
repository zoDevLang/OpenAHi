#include "openahi/inference/model.h"
#include "openahi/inference/tensor.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>

namespace openahi {
namespace inference {

// Helper function to load tensor from file
Tensor load_tensor(std::ifstream& file) {
    // Read shape
    size_t num_dims;
    file.read(reinterpret_cast<char*>(&num_dims), sizeof(num_dims));
    
    Shape shape;
    for (size_t i = 0; i < num_dims; ++i) {
        size_t dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        shape.dims.push_back(dim);
    }
    
    // Read data type
    int dtype_int;
    file.read(reinterpret_cast<char*>(&dtype_int), sizeof(dtype_int));
    DataType dtype = static_cast<DataType>(dtype_int);
    
    // Read data
    Tensor tensor(shape, dtype);
    size_t size = shape.size();
    
    switch (dtype) {
        case DataType::FLOAT32: {
            float* data = tensor.data<float>();
            file.read(reinterpret_cast<char*>(data), size * sizeof(float));
            break;
        }
        case DataType::FLOAT64: {
            double* data = tensor.data<double>();
            file.read(reinterpret_cast<char*>(data), size * sizeof(double));
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type in load_tensor");
    }
    
    return tensor;
}

// Helper function to save tensor to file
void save_tensor(const Tensor& tensor, std::ofstream& file) {
    // Write shape
    size_t num_dims = tensor.shape().num_dims();
    file.write(reinterpret_cast<const char*>(&num_dims), sizeof(num_dims));
    
    for (size_t dim : tensor.shape().dims) {
        file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }
    
    // Write data type
    int dtype_int = static_cast<int>(tensor.dtype());
    file.write(reinterpret_cast<const char*>(&dtype_int), sizeof(dtype_int));
    
    // Write data
    size_t size = tensor.size();
    switch (tensor.dtype()) {
        case DataType::FLOAT32: {
            const float* data = tensor.data<float>();
            file.write(reinterpret_cast<const char*>(data), size * sizeof(float));
            break;
        }
        case DataType::FLOAT64: {
            const double* data = tensor.data<double>();
            file.write(reinterpret_cast<const char*>(data), size * sizeof(double));
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type in save_tensor");
    }
}

// Multi-head attention implementation
MultiHeadAttention::MultiHeadAttention(const ComposterConfig& config) 
    : config_(config), 
      q_weight_(Shape({config.embedding_dim, config.embedding_dim}), DataType::FLOAT32),
      k_weight_(Shape({config.embedding_dim, config.embedding_dim}), DataType::FLOAT32),
      v_weight_(Shape({config.embedding_dim, config.embedding_dim}), DataType::FLOAT32),
      out_weight_(Shape({config.embedding_dim, config.embedding_dim}), DataType::FLOAT32),
      q_bias_(Shape({config.embedding_dim}), DataType::FLOAT32),
      k_bias_(Shape({config.embedding_dim}), DataType::FLOAT32),
      v_bias_(Shape({config.embedding_dim}), DataType::FLOAT32),
      out_bias_(Shape({config.embedding_dim}), DataType::FLOAT32),
      scale_(1.0f / sqrtf(config.head_dim)) {
    
    // Initialize weights with small random values
    q_weight_ = random_normal(Shape({config.embedding_dim, config.embedding_dim}), 0.0f, 0.02f);
    k_weight_ = random_normal(Shape({config.embedding_dim, config.embedding_dim}), 0.0f, 0.02f);
    v_weight_ = random_normal(Shape({config.embedding_dim, config.embedding_dim}), 0.0f, 0.02f);
    out_weight_ = random_normal(Shape({config.embedding_dim, config.embedding_dim}), 0.0f, 0.02f);
    
    q_bias_.zero();
    k_bias_.zero();
    v_bias_.zero();
    out_bias_.zero();
}

Tensor MultiHeadAttention::forward(const Tensor& x) {
    // x shape: (batch_size, seq_len, embedding_dim)
    size_t batch_size = x.shape()[0];
    size_t seq_len = x.shape()[1];
    
    // Project queries, keys, values
    // q = x @ q_weight + q_bias
    Tensor q = matmul(x, q_weight_);
    q = add(q, q_bias_.reshape(Shape({1, 1, config_.embedding_dim})));
    
    Tensor k = matmul(x, k_weight_);
    k = add(k, k_bias_.reshape(Shape({1, 1, config_.embedding_dim})));
    
    Tensor v = matmul(x, v_weight_);
    v = add(v, v_bias_.reshape(Shape({1, 1, config_.embedding_dim})));
    
    // Reshape for multi-head attention
    // (batch_size, seq_len, embedding_dim) -> (batch_size, num_heads, seq_len, head_dim)
    Shape q_shape({batch_size, config_.num_heads, seq_len, config_.head_dim});
    Tensor q_reshaped = q.reshape(q_shape);
    Tensor k_reshaped = k.reshape(q_shape);
    Tensor v_reshaped = v.reshape(q_shape);
    
    // Compute attention scores
    // attn_scores = (q @ k.transpose()) * scale
    Tensor k_transposed = k_reshaped.transpose();
    
    // Reshape for matmul: (batch_size * num_heads, seq_len, head_dim) @ (batch_size * num_heads, head_dim, seq_len)
    Shape q_2d_shape({batch_size * config_.num_heads, seq_len, config_.head_dim});
    Shape k_t_2d_shape({batch_size * config_.num_heads, config_.head_dim, seq_len});
    
    // For simplicity, we'll use a flattened approach
    // This is a simplified implementation; a full implementation would use proper reshaping
    Tensor attn_scores = matmul(q, k.transpose());
    attn_scores = scalar_multiply(attn_scores, scale_);
    
    // Apply causal mask (autoregressive)
    Tensor causal_mask = zeros(Shape({seq_len, seq_len}), DataType::FLOAT32);
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            if (j > i) {
                // Mask future tokens
                causal_mask.at<float>(i * seq_len + j) = -1e9f;
            }
        }
    }
    
    // Add mask to attention scores
    // Broadcast mask to all batches and heads
    Tensor mask_broadcasted = causal_mask.reshape(Shape({1, 1, seq_len, seq_len}));
    // For now, just add the mask to each batch/head
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < config_.num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < seq_len; ++j) {
                    size_t idx = b * config_.num_heads * seq_len * seq_len + 
                                h * seq_len * seq_len + 
                                i * seq_len + j;
                    if (j > i) {
                        attn_scores.at<float>(idx) += -1e9f;
                    }
                }
            }
        }
    }
    
    // Softmax
    Tensor attn_probs = softmax(attn_scores, -1);
    
    // Apply attention to values
    // Reshape v for matmul
    Tensor output = matmul(attn_probs, v);
    
    // Reshape back
    // (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, embedding_dim)
    Shape output_shape({batch_size, seq_len, config_.embedding_dim});
    output = output.reshape(output_shape);
    
    // Output projection
    output = matmul(output, out_weight_);
    output = add(output, out_bias_.reshape(Shape({1, 1, config_.embedding_dim})));
    
    return output;
}

void MultiHeadAttention::load_weights(const std::string& prefix, std::ifstream& file) {
    q_weight_ = load_tensor(file);
    k_weight_ = load_tensor(file);
    v_weight_ = load_tensor(file);
    out_weight_ = load_tensor(file);
    q_bias_ = load_tensor(file);
    k_bias_ = load_tensor(file);
    v_bias_ = load_tensor(file);
    out_bias_ = load_tensor(file);
}

void MultiHeadAttention::save_weights(const std::string& prefix, std::ofstream& file) const {
    save_tensor(q_weight_, file);
    save_tensor(k_weight_, file);
    save_tensor(v_weight_, file);
    save_tensor(out_weight_, file);
    save_tensor(q_bias_, file);
    save_tensor(k_bias_, file);
    save_tensor(v_bias_, file);
    save_tensor(out_bias_, file);
}

// Feed-forward network implementation
FeedForward::FeedForward(const ComposterConfig& config) 
    : config_(config), hidden_dim_(config.embedding_dim * 4),
      fc1_weight_(Shape({config.embedding_dim, hidden_dim_}), DataType::FLOAT32),
      fc1_bias_(Shape({hidden_dim_}), DataType::FLOAT32),
      fc2_weight_(Shape({hidden_dim_, config.embedding_dim}), DataType::FLOAT32),
      fc2_bias_(Shape({config.embedding_dim}), DataType::FLOAT32) {
    
    fc1_weight_ = random_normal(Shape({config.embedding_dim, hidden_dim_}), 0.0f, 0.02f);
    fc2_weight_ = random_normal(Shape({hidden_dim_, config.embedding_dim}), 0.0f, 0.02f);
    fc1_bias_.zero();
    fc2_bias_.zero();
}

Tensor FeedForward::forward(const Tensor& x) {
    // x shape: (batch_size, seq_len, embedding_dim)
    
    // First layer
    Tensor h = matmul(x, fc1_weight_);
    h = add(h, fc1_bias_.reshape(Shape({1, 1, hidden_dim_})));
    h = gelu(h);
    
    // Second layer
    Tensor output = matmul(h, fc2_weight_);
    output = add(output, fc2_bias_.reshape(Shape({1, 1, config_.embedding_dim})));
    
    return output;
}

void FeedForward::load_weights(const std::string& prefix, std::ifstream& file) {
    fc1_weight_ = load_tensor(file);
    fc1_bias_ = load_tensor(file);
    fc2_weight_ = load_tensor(file);
    fc2_bias_ = load_tensor(file);
}

void FeedForward::save_weights(const std::string& prefix, std::ofstream& file) const {
    save_tensor(fc1_weight_, file);
    save_tensor(fc1_bias_, file);
    save_tensor(fc2_weight_, file);
    save_tensor(fc2_bias_, file);
}

// Transformer block implementation
TransformerBlock::TransformerBlock(const ComposterConfig& config) 
    : config_(config), attention_(config), ffn_(config),
      ln1_gamma_(Shape({config.embedding_dim}), DataType::FLOAT32),
      ln1_beta_(Shape({config.embedding_dim}), DataType::FLOAT32),
      ln2_gamma_(Shape({config.embedding_dim}), DataType::FLOAT32),
      ln2_beta_(Shape({config.embedding_dim}), DataType::FLOAT32) {
    
    ln1_gamma_ = ones(Shape({config.embedding_dim}), DataType::FLOAT32);
    ln1_beta_ = zeros(Shape({config.embedding_dim}), DataType::FLOAT32);
    ln2_gamma_ = ones(Shape({config.embedding_dim}), DataType::FLOAT32);
    ln2_beta_ = zeros(Shape({config.embedding_dim}), DataType::FLOAT32);
}

Tensor TransformerBlock::forward(const Tensor& x) {
    // x shape: (batch_size, seq_len, embedding_dim)
    
    // Self-attention with residual connection
    Tensor attn_output = attention_.forward(x);
    Tensor x1 = add(x, attn_output);
    
    // Layer norm
    Tensor ln1_output = layer_norm(x1, ln1_gamma_, ln1_beta_);
    
    // Feed-forward with residual connection
    Tensor ffn_output = ffn_.forward(ln1_output);
    Tensor x2 = add(ln1_output, ffn_output);
    
    // Layer norm
    Tensor output = layer_norm(x2, ln2_gamma_, ln2_beta_);
    
    return output;
}

void TransformerBlock::load_weights(const std::string& prefix, std::ifstream& file) {
    attention_.load_weights(prefix + "attention.", file);
    ffn_.load_weights(prefix + "ffn.", file);
    ln1_gamma_ = load_tensor(file);
    ln1_beta_ = load_tensor(file);
    ln2_gamma_ = load_tensor(file);
    ln2_beta_ = load_tensor(file);
}

void TransformerBlock::save_weights(const std::string& prefix, std::ofstream& file) const {
    attention_.save_weights(prefix + "attention.", file);
    ffn_.save_weights(prefix + "ffn.", file);
    save_tensor(ln1_gamma_, file);
    save_tensor(ln1_beta_, file);
    save_tensor(ln2_gamma_, file);
    save_tensor(ln2_beta_, file);
}

// Composter model implementation
ComposterModel::ComposterModel(const ComposterConfig& config) 
    : config_(config), dropout_(config.dropout),
      token_embeddings_(Shape({config.vocab_size, config.embedding_dim}), DataType::FLOAT32),
      position_embeddings_(Shape({config.context_length, config.embedding_dim}), DataType::FLOAT32),
      final_ln_gamma_(Shape({config.embedding_dim}), DataType::FLOAT32),
      final_ln_beta_(Shape({config.embedding_dim}), DataType::FLOAT32),
      output_proj_weight_(Shape({config.embedding_dim, config.vocab_size}), DataType::FLOAT32),
      output_proj_bias_(Shape({config.vocab_size}), DataType::FLOAT32) {
    
    // Initialize embeddings
    token_embeddings_ = random_normal(Shape({config.vocab_size, config.embedding_dim}), 0.0f, 0.02f);
    position_embeddings_ = random_normal(Shape({config.context_length, config.embedding_dim}), 0.0f, 0.02f);
    
    // Initialize layer norm
    final_ln_gamma_ = ones(Shape({config.embedding_dim}), DataType::FLOAT32);
    final_ln_beta_ = zeros(Shape({config.embedding_dim}), DataType::FLOAT32);
    
    // Initialize output projection
    output_proj_weight_ = random_normal(Shape({config.embedding_dim, config.vocab_size}), 0.0f, 0.02f);
    output_proj_bias_ = zeros(Shape({config.vocab_size}), DataType::FLOAT32);
    
    // Create transformer layers
    for (int i = 0; i < config.num_layers; ++i) {
        layers_.emplace_back(config);
    }
}

ComposterModel::~ComposterModel() {
}

Tensor ComposterModel::forward(const Tensor& input_ids) {
    // input_ids shape: (batch_size, seq_len)
    size_t batch_size = input_ids.shape()[0];
    size_t seq_len = input_ids.shape()[1];
    
    // Get token embeddings
    // For simplicity, we'll use a simplified embedding lookup
    // In a full implementation, this would be optimized
    Tensor token_embeds = zeros(Shape({batch_size, seq_len, config_.embedding_dim}), DataType::FLOAT32);
    
    const int* input_data = input_ids.data<int>();
    const float* emb_data = token_embeddings_.data<float>();
    float* output_data = token_embeds.data<float>();
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            int token_id = input_data[b * seq_len + s];
            if (token_id < 0 || token_id >= config_.vocab_size) {
                // Out of bounds, use zero embedding
                token_id = 0;
            }
            for (int d = 0; d < config_.embedding_dim; ++d) {
                output_data[b * seq_len * config_.embedding_dim + s * config_.embedding_dim + d] = 
                    emb_data[token_id * config_.embedding_dim + d];
            }
        }
    }
    
    // Get positional embeddings
    Tensor pos_embeds = zeros(Shape({batch_size, seq_len, config_.embedding_dim}), DataType::FLOAT32);
    const float* pos_emb_data = position_embeddings_.data<float>();
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            int pos = s;
            if (pos >= config_.context_length) {
                pos = config_.context_length - 1;
            }
            for (int d = 0; d < config_.embedding_dim; ++d) {
                output_data[b * seq_len * config_.embedding_dim + s * config_.embedding_dim + d] += 
                    pos_emb_data[pos * config_.embedding_dim + d];
            }
        }
    }
    
    // Combine embeddings (already done in token_embeds)
    Tensor x = token_embeds;
    
    // Pass through transformer blocks
    for (auto& layer : layers_) {
        x = layer.forward(x);
    }
    
    // Final layer norm
    x = layer_norm(x, final_ln_gamma_, final_ln_beta_);
    
    // Output projection
    Tensor logits = matmul(x, output_proj_weight_);
    logits = add(logits, output_proj_bias_.reshape(Shape({1, 1, config_.vocab_size})));
    
    return logits;
}

std::vector<int> ComposterModel::generate(
    const std::vector<int>& input_ids,
    int max_new_tokens,
    float temperature,
    int top_k,
    int eos_token_id
) {
    std::vector<int> generated = input_ids;
    
    for (int step = 0; step < max_new_tokens; ++step) {
        // Get logits for current sequence
        Shape input_shape({1, generated.size()});
        Tensor input_tensor(input_shape, DataType::INT32);
        int* input_data = input_tensor.data<int>();
        for (size_t i = 0; i < generated.size(); ++i) {
            input_data[i] = generated[i];
        }
        
        Tensor logits = forward(input_tensor);
        
        // Get logits for the last position
        Tensor last_logits = logits.slice(1, logits.shape()[1] - 1, logits.shape()[1]);
        last_logits = last_logits.reshape(Shape({config_.vocab_size}));
        
        // Apply temperature
        if (temperature != 1.0f) {
            last_logits = scalar_multiply(last_logits, 1.0f / temperature);
        }
        
        // Apply top-k filtering
        if (top_k > 0) {
            // Find top-k values
            std::vector<std::pair<float, int>> logits_with_indices;
            const float* logits_data = last_logits.data<float>();
            for (int i = 0; i < config_.vocab_size; ++i) {
                logits_with_indices.emplace_back(logits_data[i], i);
            }
            
            // Sort by logit value (descending)
            std::sort(logits_with_indices.begin(), logits_with_indices.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
            
            // Set non-top-k logits to -inf
            float threshold = logits_with_indices[std::min(top_k, (int)logits_with_indices.size()) - 1].first;
            for (int i = 0; i < config_.vocab_size; ++i) {
                if (logits_data[i] < threshold) {
                    last_logits.at<float>(i) = -1e9f;
                }
            }
        }
        
        // Softmax
        Tensor probs = softmax(last_logits, 0);
        const float* probs_data = probs.data<float>();
        
        // Sample next token
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<int> dist(probs_data, probs_data + config_.vocab_size);
        int next_token = dist(gen);
        
        // Check for EOS
        if (eos_token_id >= 0 && next_token == eos_token_id) {
            break;
        }
        
        // Append to generated sequence
        generated.push_back(next_token);
        
        // Stop if we've reached context length
        if (generated.size() >= config_.context_length) {
            break;
        }
    }
    
    return generated;
}

void ComposterModel::load_from_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open model file: " + filepath);
    }
    
    // Read config
    file.read(reinterpret_cast<char*>(&config_), sizeof(config_));
    
    // Reinitialize model with loaded config
    *this = ComposterModel(config_);
    
    // Load token embeddings
    token_embeddings_ = load_tensor(file);
    position_embeddings_ = load_tensor(file);
    
    // Load transformer layers
    for (auto& layer : layers_) {
        layer.load_weights("", file);
    }
    
    // Load final layer norm
    final_ln_gamma_ = load_tensor(file);
    final_ln_beta_ = load_tensor(file);
    
    // Load output projection
    output_proj_weight_ = load_tensor(file);
    output_proj_bias_ = load_tensor(file);
}

void ComposterModel::save_to_file(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to create model file: " + filepath);
    }
    
    // Write config
    file.write(reinterpret_cast<const char*>(&config_), sizeof(config_));
    
    // Save token embeddings
    save_tensor(token_embeddings_, file);
    save_tensor(position_embeddings_, file);
    
    // Save transformer layers
    for (const auto& layer : layers_) {
        layer.save_weights("", file);
    }
    
    // Save final layer norm
    save_tensor(final_ln_gamma_, file);
    save_tensor(final_ln_beta_, file);
    
    // Save output projection
    save_tensor(output_proj_weight_, file);
    save_tensor(output_proj_bias_, file);
}

size_t ComposterModel::get_num_params() const {
    size_t count = 0;
    count += token_embeddings_.size();
    count += position_embeddings_.size();
    count += final_ln_gamma_.size();
    count += final_ln_beta_.size();
    count += output_proj_weight_.size();
    count += output_proj_bias_.size();
    
    for (const auto& layer : layers_) {
        // Count parameters in each layer
        // This is a simplified count
        count += layer.get_config().embedding_dim * layer.get_config().embedding_dim * 4; // Q, K, V, O weights
        count += layer.get_config().embedding_dim * 4; // Biases
        count += layer.get_config().embedding_dim * 4; // FFN weights
        count += layer.get_config().embedding_dim * 2; // Layer norm params
    }
    
    return count;
}

std::string ComposterModel::get_info() const {
    char buffer[1024];
    snprintf(buffer, sizeof(buffer),
             "ComposterModel:\n"
             "  vocab_size: %d\n"
             "  context_length: %d\n"
             "  embedding_dim: %d\n"
             "  num_layers: %d\n"
             "  num_heads: %d\n"
             "  num_params: %zu\n",
             config_.vocab_size,
             config_.context_length,
             config_.embedding_dim,
             config_.num_layers,
             config_.num_heads,
             get_num_params());
    return std::string(buffer);
}

Tensor ComposterModel::create_causal_mask(int seq_len) const {
    Tensor mask = zeros(Shape({seq_len, seq_len}), DataType::FLOAT32);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            if (j > i) {
                mask.at<float>(i * seq_len + j) = -1e9f;
            }
        }
    }
    return mask;
}

}} // namespace inference
} // namespace openahi
