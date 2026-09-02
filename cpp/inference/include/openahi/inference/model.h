#ifndef OPENAHI_INFERENCE_MODEL_H
#define OPENAHI_INFERENCE_MODEL_H

#include "openahi/inference/tensor.h"

#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <cstdint>

namespace openahi {
namespace inference {

/**
 * @brief Configuration for Composter model
 */
struct ComposterConfig {
    int vocab_size = 8192;
    int context_length = 512;
    int embedding_dim = 512;
    int num_layers = 6;
    int num_heads = 8;
    float dropout = 0.1f;
    
    // Derived values
    int head_dim = 0;
    
    ComposterConfig() {
        head_dim = embedding_dim / num_heads;
    }
    
    bool validate() const {
        if (embedding_dim % num_heads != 0) {
            return false;
        }
        if (head_dim != embedding_dim / num_heads) {
            return false;
        }
        return true;
    }
    
    void print() const {
        printf("ComposterConfig:\n");
        printf("  vocab_size: %d\n", vocab_size);
        printf("  context_length: %d\n", context_length);
        printf("  embedding_dim: %d\n", embedding_dim);
        printf("  num_layers: %d\n", num_layers);
        printf("  num_heads: %d\n", num_heads);
        printf("  head_dim: %d\n", head_dim);
        printf("  dropout: %f\n", dropout);
    }
};

/**
 * @brief Multi-head attention layer
 */
class MultiHeadAttention {
public:
    MultiHeadAttention(const ComposterConfig& config);
    
    /**
     * Forward pass
     * @param x Input tensor of shape (batch_size, seq_len, embedding_dim)
     * @return Output tensor of shape (batch_size, seq_len, embedding_dim)
     */
    Tensor forward(const Tensor& x);
    
    /**
     * Load weights from file
     */
    void load_weights(const std::string& prefix, std::ifstream& file);
    
    /**
     * Save weights to file
     */
    void save_weights(const std::string& prefix, std::ofstream& file) const;

private:
    ComposterConfig config_;
    
    // Weight matrices
    Tensor q_weight_;
    Tensor k_weight_;
    Tensor v_weight_;
    Tensor out_weight_;
    
    // Bias vectors
    Tensor q_bias_;
    Tensor k_bias_;
    Tensor v_bias_;
    Tensor out_bias_;
    
    float scale_;
};

/**
 * @brief Feed-forward network
 */
class FeedForward {
public:
    FeedForward(const ComposterConfig& config);
    
    Tensor forward(const Tensor& x);
    
    void load_weights(const std::string& prefix, std::ifstream& file);
    void save_weights(const std::string& prefix, std::ofstream& file) const;

private:
    ComposterConfig config_;
    int hidden_dim_;
    
    Tensor fc1_weight_;
    Tensor fc1_bias_;
    Tensor fc2_weight_;
    Tensor fc2_bias_;
};

/**
 * @brief Transformer block
 */
class TransformerBlock {
public:
    TransformerBlock(const ComposterConfig& config);
    
    Tensor forward(const Tensor& x);
    
    void load_weights(const std::string& prefix, std::ifstream& file);
    void save_weights(const std::string& prefix, std::ofstream& file) const;

private:
    ComposterConfig config_;
    MultiHeadAttention attention_;
    FeedForward ffn_;
    
    // Layer normalization
    Tensor ln1_gamma_;
    Tensor ln1_beta_;
    Tensor ln2_gamma_;
    Tensor ln2_beta_;
};

/**
 * @brief Composter model
 */
class ComposterModel {
public:
    ComposterModel(const ComposterConfig& config);
    ~ComposterModel();
    
    /**
     * Forward pass
     * @param input_ids Input token IDs of shape (batch_size, seq_len)
     * @return Logits of shape (batch_size, seq_len, vocab_size)
     */
    Tensor forward(const Tensor& input_ids);
    
    /**
     * Generate text autoregressively
     * @param input_ids Starting token IDs
     * @param max_new_tokens Maximum number of new tokens to generate
     * @param temperature Temperature for sampling
     * @param top_k Number of top tokens to sample from (0 = all)
     * @param eos_token_id End-of-sequence token ID (-1 = none)
     * @return Generated token IDs
     */
    std::vector<int> generate(
        const std::vector<int>& input_ids,
        int max_new_tokens = 100,
        float temperature = 1.0f,
        int top_k = 0,
        int eos_token_id = -1
    );
    
    /**
     * Load model from checkpoint file
     */
    void load_from_file(const std::string& filepath);
    
    /**
     * Save model to checkpoint file
     */
    void save_to_file(const std::string& filepath) const;
    
    /**
     * Get model configuration
     */
    const ComposterConfig& get_config() const { return config_; }
    
    /**
     * Get number of parameters
     */
    size_t get_num_params() const;
    
    /**
     * Get model info as string
     */
    std::string get_info() const;

private:
    ComposterConfig config_;
    
    // Embeddings
    Tensor token_embeddings_;
    Tensor position_embeddings_;
    
    // Transformer layers
    std::vector<TransformerBlock> layers_;
    
    // Final layer norm
    Tensor final_ln_gamma_;
    Tensor final_ln_beta_;
    
    // Output projection
    Tensor output_proj_weight_;
    Tensor output_proj_bias_;
    
    // Dropout mask (not implemented in C++ version for simplicity)
    float dropout_;
    
    // Helper for causal mask
    Tensor create_causal_mask(int seq_len) const;
};

/**
 * @brief Model registry
 */
class ModelRegistry {
public:
    static ComposterModel* load_model(const std::string& model_name, const std::string& version = "1.00.0");
    static void register_model(const std::string& name, const std::string& version, ComposterModel* model);
    static bool has_model(const std::string& name, const std::string& version = "1.00.0");
};

}} // namespace inference
} // namespace openahi

#endif // OPENAHI_INFERENCE_MODEL_H
