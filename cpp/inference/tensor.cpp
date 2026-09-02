#include "openahi/inference/tensor.h"
#include <cstring>
#include <cmath>
#include <random>
#include <iostream>

namespace openahi {
namespace inference {

// Helper function to get data type size
size_t get_dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return sizeof(float);
        case DataType::FLOAT64: return sizeof(double);
        case DataType::INT32: return sizeof(int32_t);
        case DataType::INT64: return sizeof(int64_t);
        case DataType::UINT8: return sizeof(uint8_t);
        default: return 0;
    }
}

// Tensor implementation
Tensor::Tensor(const Shape& shape, DataType dtype) : shape_(shape), dtype_(dtype) {
    size_ = shape.size();
    allocate(size_, dtype);
}

Tensor::Tensor(void* data, const Shape& shape, DataType dtype) 
    : data_(data), shape_(shape), dtype_(dtype), owns_data_(false) {
    size_ = shape.size();
}

Tensor::Tensor(const Tensor& other) 
    : shape_(other.shape_), dtype_(other.dtype_), size_(other.size_) {
    allocate(size_, dtype_);
    copy_data(other.data_, size_, dtype_);
}

Tensor::Tensor(Tensor&& other) noexcept 
    : data_(other.data_), shape_(other.shape_), dtype_(other.dtype_), 
      size_(other.size_), owns_data_(other.owns_data_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.owns_data_ = false;
}

Tensor::~Tensor() {
    deallocate();
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        deallocate();
        shape_ = other.shape_;
        dtype_ = other.dtype_;
        size_ = other.size_;
        allocate(size_, dtype_);
        copy_data(other.data_, size_, dtype_);
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        deallocate();
        data_ = other.data_;
        shape_ = other.shape_;
        dtype_ = other.dtype_;
        size_ = other.size_;
        owns_data_ = other.owns_data_;
        other.data_ = nullptr;
        other.size_ = 0;
        other.owns_data_ = false;
    }
    return *this;
}

void Tensor::allocate(size_t size, DataType dtype) {
    if (size == 0) return;
    size_t type_size = get_dtype_size(dtype);
    data_ = malloc(size * type_size);
    if (!data_) {
        throw std::bad_alloc();
    }
    owns_data_ = true;
}

void Tensor::deallocate() {
    if (data_ && owns_data_) {
        free(data_);
        data_ = nullptr;
    }
    size_ = 0;
}

void Tensor::copy_data(const void* src, size_t size, DataType dtype) {
    if (!src || size == 0) return;
    size_t type_size = get_dtype_size(dtype);
    memcpy(data_, src, size * type_size);
}

size_t Tensor::dtype_size() const {
    return get_dtype_size(dtype_);
}

void Tensor::zero() {
    if (!data_ || size_ == 0) return;
    memset(data_, 0, size_ * dtype_size());
}

template <typename T>
void Tensor::fill(T value) {
    T* ptr = data<T>();
    for (size_t i = 0; i < size_; ++i) {
        ptr[i] = value;
    }
}

void Tensor::reshape(const Shape& new_shape) {
    if (new_shape.size() != size_) {
        throw std::runtime_error("Reshape failed: total size mismatch");
    }
    shape_ = new_shape;
}

Tensor Tensor::flatten() const {
    Shape new_shape({size_});
    Tensor result(*this);
    result.shape_ = new_shape;
    return result;
}

Tensor Tensor::transpose() const {
    if (shape_.num_dims() != 2) {
        throw std::runtime_error("Transpose only supported for 2D tensors");
    }
    
    Shape new_shape({shape_[1], shape_[0]});
    Tensor result(new_shape, dtype_);
    
    // Simple transpose implementation
    for (size_t i = 0; i < shape_[0]; ++i) {
        for (size_t j = 0; j < shape_[1]; ++j) {
            size_t src_idx = i * shape_[1] + j;
            size_t dst_idx = j * shape_[0] + i;
            
            switch (dtype_) {
                case DataType::FLOAT32: {
                    float* src = data<float>();
                    float* dst = result.data<float>();
                    dst[dst_idx] = src[src_idx];
                    break;
                }
                case DataType::FLOAT64: {
                    double* src = data<double>();
                    double* dst = result.data<double>();
                    dst[dst_idx] = src[src_idx];
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported data type for transpose");
            }
        }
    }
    
    return result;
}

Tensor Tensor::slice(size_t dim, size_t start, size_t end) const {
    if (dim >= shape_.num_dims()) {
        throw std::runtime_error("Slice dimension out of range");
    }
    if (start >= shape_[dim] || end > shape_[dim] || start >= end) {
        throw std::runtime_error("Invalid slice range");
    }
    
    Shape new_shape = shape_;
    new_shape[dim] = end - start;
    Tensor result(new_shape, dtype_);
    
    // Calculate strides
    size_t stride = 1;
    std::vector<size_t> strides(shape_.num_dims(), 1);
    for (int i = shape_.num_dims() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape_[i];
    }
    
    // Copy data with slicing
    size_t src_stride = strides[dim];
    size_t dst_stride = strides[dim];
    
    for (size_t i = 0; i < result.size(); ++i) {
        // Calculate source index
        size_t src_index = 0;
        size_t remaining = i;
        for (size_t d = 0; d < shape_.num_dims(); ++d) {
            size_t size = (d == dim) ? (end - start) : shape_[d];
            size_t coord = remaining % size;
            if (d == dim) {
                coord += start;
            }
            remaining /= size;
            src_index += coord * strides[d];
        }
        
        // Copy element
        switch (dtype_) {
            case DataType::FLOAT32: {
                float* src = data<float>();
                float* dst = result.data<float>();
                dst[i] = src[src_index];
                break;
            }
            case DataType::FLOAT64: {
                double* src = data<double>();
                double* dst = result.data<double>();
                dst[i] = src[src_index];
                break;
            }
            default:
                throw std::runtime_error("Unsupported data type for slice");
        }
    }
    
    return result;
}

void Tensor::print_info(std::ostream& os) const {
    os << "Tensor(shape=" << shape_.to_string() 
       << ", dtype=" << static_cast<int>(dtype_) 
       << ", size=" << size_ 
       << ", memory=" << memory_usage() / 1024.0 / 1024.0 << "MB)";
}

// Factory functions
Tensor zeros(const Shape& shape, DataType dtype) {
    Tensor result(shape, dtype);
    result.zero();
    return result;
}

Tensor ones(const Shape& shape, DataType dtype) {
    Tensor result(shape, dtype);
    switch (dtype) {
        case DataType::FLOAT32: result.fill<float>(1.0f); break;
        case DataType::FLOAT64: result.fill<double>(1.0); break;
        case DataType::INT32: result.fill<int32_t>(1); break;
        case DataType::INT64: result.fill<int64_t>(1); break;
        case DataType::UINT8: result.fill<uint8_t>(1); break;
        default: break;
    }
    return result;
}

Tensor random_uniform(const Shape& shape, float min, float max, DataType dtype) {
    Tensor result(shape, dtype);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    
    switch (dtype) {
        case DataType::FLOAT32: {
            float* data = result.data<float>();
            for (size_t i = 0; i < result.size(); ++i) {
                data[i] = dist(gen);
            }
            break;
        }
        case DataType::FLOAT64: {
            double* data = result.data<double>();
            std::uniform_real_distribution<double> dist_d(min, max);
            for (size_t i = 0; i < result.size(); ++i) {
                data[i] = dist_d(gen);
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type for random_uniform");
    }
    
    return result;
}

Tensor random_normal(const Shape& shape, float mean, float stddev, DataType dtype) {
    Tensor result(shape, dtype);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, stddev);
    
    switch (dtype) {
        case DataType::FLOAT32: {
            float* data = result.data<float>();
            for (size_t i = 0; i < result.size(); ++i) {
                data[i] = dist(gen);
            }
            break;
        }
        case DataType::FLOAT64: {
            double* data = result.data<double>();
            std::normal_distribution<double> dist_d(mean, stddev);
            for (size_t i = 0; i < result.size(); ++i) {
                data[i] = dist_d(gen);
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type for random_normal");
    }
    
    return result;
}

// Matrix multiplication
template <typename T>
void matmul_impl(const T* a, const T* b, T* c, 
                 size_t a_rows, size_t a_cols, size_t b_cols) {
    for (size_t i = 0; i < a_rows; ++i) {
        for (size_t j = 0; j < b_cols; ++j) {
            T sum = 0;
            for (size_t k = 0; k < a_cols; ++k) {
                sum += a[i * a_cols + k] * b[k * b_cols + j];
            }
            c[i * b_cols + j] = sum;
        }
    }
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape_.num_dims() != 2 || b.shape_.num_dims() != 2) {
        throw std::runtime_error("Matmul only supported for 2D tensors");
    }
    if (a.shape_[1] != b.shape_[0]) {
        throw std::runtime_error("Matmul dimension mismatch");
    }
    
    Shape result_shape({a.shape_[0], b.shape_[1]});
    Tensor result(result_shape, a.dtype_);
    
    if (a.dtype_ != b.dtype_) {
        throw std::runtime_error("Matmul data type mismatch");
    }
    
    switch (a.dtype_) {
        case DataType::FLOAT32: {
            matmul_impl(a.data<float>(), b.data<float>(), result.data<float>(),
                       a.shape_[0], a.shape_[1], b.shape_[1]);
            break;
        }
        case DataType::FLOAT64: {
            matmul_impl(a.data<double>(), b.data<double>(), result.data<double>(),
                       a.shape_[0], a.shape_[1], b.shape_[1]);
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type for matmul");
    }
    
    return result;
}

// Element-wise operations
Tensor add(const Tensor& a, const Tensor& b) {
    if (a.shape_ != b.shape_) {
        throw std::runtime_error("Add shape mismatch");
    }
    if (a.dtype_ != b.dtype_) {
        throw std::runtime_error("Add data type mismatch");
    }
    
    Tensor result(a.shape_, a.dtype_);
    
    switch (a.dtype_) {
        case DataType::FLOAT32: {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            float* r_data = result.data<float>();
            for (size_t i = 0; i < a.size(); ++i) {
                r_data[i] = a_data[i] + b_data[i];
            }
            break;
        }
        case DataType::FLOAT64: {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            double* r_data = result.data<double>();
            for (size_t i = 0; i < a.size(); ++i) {
                r_data[i] = a_data[i] + b_data[i];
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type for add");
    }
    
    return result;
}

Tensor multiply(const Tensor& a, const Tensor& b) {
    if (a.shape_ != b.shape_) {
        throw std::runtime_error("Multiply shape mismatch");
    }
    if (a.dtype_ != b.dtype_) {
        throw std::runtime_error("Multiply data type mismatch");
    }
    
    Tensor result(a.shape_, a.dtype_);
    
    switch (a.dtype_) {
        case DataType::FLOAT32: {
            const float* a_data = a.data<float>();
            const float* b_data = b.data<float>();
            float* r_data = result.data<float>();
            for (size_t i = 0; i < a.size(); ++i) {
                r_data[i] = a_data[i] * b_data[i];
            }
            break;
        }
        case DataType::FLOAT64: {
            const double* a_data = a.data<double>();
            const double* b_data = b.data<double>();
            double* r_data = result.data<double>();
            for (size_t i = 0; i < a.size(); ++i) {
                r_data[i] = a_data[i] * b_data[i];
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type for multiply");
    }
    
    return result;
}

Tensor scalar_multiply(const Tensor& a, float scalar) {
    Tensor result(a.shape_, a.dtype_);
    
    switch (a.dtype_) {
        case DataType::FLOAT32: {
            const float* a_data = a.data<float>();
            float* r_data = result.data<float>();
            for (size_t i = 0; i < a.size(); ++i) {
                r_data[i] = a_data[i] * scalar;
            }
            break;
        }
        case DataType::FLOAT64: {
            const double* a_data = a.data<double>();
            double* r_data = result.data<double>();
            for (size_t i = 0; i < a.size(); ++i) {
                r_data[i] = a_data[i] * scalar;
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type for scalar_multiply");
    }
    
    return result;
}

// Softmax implementation
Tensor softmax(const Tensor& x, int dim) {
    if (x.shape_.num_dims() != 2) {
        throw std::runtime_error("Softmax only supported for 2D tensors");
    }
    
    // Adjust negative dim
    if (dim < 0) {
        dim = x.shape_.num_dims() + dim;
    }
    
    Tensor result(x.shape_, x.dtype_);
    
    switch (x.dtype_) {
        case DataType::FLOAT32: {
            const float* x_data = x.data<float>();
            float* r_data = result.data<float>();
            
            if (dim == 0) {
                // Softmax over rows
                for (size_t i = 0; i < x.shape_[0]; ++i) {
                    // Find max for numerical stability
                    float max_val = x_data[i * x.shape_[1]];
                    for (size_t j = 1; j < x.shape_[1]; ++j) {
                        float val = x_data[i * x.shape_[1] + j];
                        if (val > max_val) max_val = val;
                    }
                    
                    // Compute exp and sum
                    float sum = 0.0f;
                    for (size_t j = 0; j < x.shape_[1]; ++j) {
                        float exp_val = expf(x_data[i * x.shape_[1] + j] - max_val);
                        r_data[i * x.shape_[1] + j] = exp_val;
                        sum += exp_val;
                    }
                    
                    // Normalize
                    float inv_sum = 1.0f / sum;
                    for (size_t j = 0; j < x.shape_[1]; ++j) {
                        r_data[i * x.shape_[1] + j] *= inv_sum;
                    }
                }
            } else {
                // Softmax over columns
                for (size_t j = 0; j < x.shape_[1]; ++j) {
                    // Find max for numerical stability
                    float max_val = x_data[j];
                    for (size_t i = 1; i < x.shape_[0]; ++i) {
                        float val = x_data[i * x.shape_[1] + j];
                        if (val > max_val) max_val = val;
                    }
                    
                    // Compute exp and sum
                    float sum = 0.0f;
                    for (size_t i = 0; i < x.shape_[0]; ++i) {
                        float exp_val = expf(x_data[i * x.shape_[1] + j] - max_val);
                        r_data[i * x.shape_[1] + j] = exp_val;
                        sum += exp_val;
                    }
                    
                    // Normalize
                    float inv_sum = 1.0f / sum;
                    for (size_t i = 0; i < x.shape_[0]; ++i) {
                        r_data[i * x.shape_[1] + j] *= inv_sum;
                    }
                }
            }
            break;
        }
        case DataType::FLOAT64: {
            const double* x_data = x.data<double>();
            double* r_data = result.data<double>();
            
            if (dim == 0) {
                for (size_t i = 0; i < x.shape_[0]; ++i) {
                    double max_val = x_data[i * x.shape_[1]];
                    for (size_t j = 1; j < x.shape_[1]; ++j) {
                        double val = x_data[i * x.shape_[1] + j];
                        if (val > max_val) max_val = val;
                    }
                    
                    double sum = 0.0;
                    for (size_t j = 0; j < x.shape_[1]; ++j) {
                        double exp_val = exp(x_data[i * x.shape_[1] + j] - max_val);
                        r_data[i * x.shape_[1] + j] = exp_val;
                        sum += exp_val;
                    }
                    
                    double inv_sum = 1.0 / sum;
                    for (size_t j = 0; j < x.shape_[1]; ++j) {
                        r_data[i * x.shape_[1] + j] *= inv_sum;
                    }
                }
            } else {
                for (size_t j = 0; j < x.shape_[1]; ++j) {
                    double max_val = x_data[j];
                    for (size_t i = 1; i < x.shape_[0]; ++i) {
                        double val = x_data[i * x.shape_[1] + j];
                        if (val > max_val) max_val = val;
                    }
                    
                    double sum = 0.0;
                    for (size_t i = 0; i < x.shape_[0]; ++i) {
                        double exp_val = exp(x_data[i * x.shape_[1] + j] - max_val);
                        r_data[i * x.shape_[1] + j] = exp_val;
                        sum += exp_val;
                    }
                    
                    double inv_sum = 1.0 / sum;
                    for (size_t i = 0; i < x.shape_[0]; ++i) {
                        r_data[i * x.shape_[1] + j] *= inv_sum;
                    }
                }
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type for softmax");
    }
    
    return result;
}

Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps) {
    if (x.shape_ != gamma.shape_ || x.shape_ != beta.shape_) {
        throw std::runtime_error("Layer norm shape mismatch");
    }
    if (x.shape_.num_dims() != 2) {
        throw std::runtime_error("Layer norm only supported for 2D tensors");
    }
    
    Tensor result(x.shape_, x.dtype_);
    
    switch (x.dtype_) {
        case DataType::FLOAT32: {
            const float* x_data = x.data<float>();
            const float* gamma_data = gamma.data<float>();
            const float* beta_data = beta.data<float>();
            float* r_data = result.data<float>();
            
            for (size_t i = 0; i < x.shape_[0]; ++i) {
                // Compute mean and variance
                float mean = 0.0f;
                for (size_t j = 0; j < x.shape_[1]; ++j) {
                    mean += x_data[i * x.shape_[1] + j];
                }
                mean /= x.shape_[1];
                
                float variance = 0.0f;
                for (size_t j = 0; j < x.shape_[1]; ++j) {
                    float diff = x_data[i * x.shape_[1] + j] - mean;
                    variance += diff * diff;
                }
                variance /= x.shape_[1];
                
                // Normalize
                float stddev = sqrtf(variance + eps);
                for (size_t j = 0; j < x.shape_[1]; ++j) {
                    float normalized = (x_data[i * x.shape_[1] + j] - mean) / stddev;
                    r_data[i * x.shape_[1] + j] = normalized * gamma_data[j] + beta_data[j];
                }
            }
            break;
        }
        case DataType::FLOAT64: {
            const double* x_data = x.data<double>();
            const double* gamma_data = gamma.data<double>();
            const double* beta_data = beta.data<double>();
            double* r_data = result.data<double>();
            
            for (size_t i = 0; i < x.shape_[0]; ++i) {
                double mean = 0.0;
                for (size_t j = 0; j < x.shape_[1]; ++j) {
                    mean += x_data[i * x.shape_[1] + j];
                }
                mean /= x.shape_[1];
                
                double variance = 0.0;
                for (size_t j = 0; j < x.shape_[1]; ++j) {
                    double diff = x_data[i * x.shape_[1] + j] - mean;
                    variance += diff * diff;
                }
                variance /= x.shape_[1];
                
                double stddev = sqrt(variance + eps);
                for (size_t j = 0; j < x.shape_[1]; ++j) {
                    double normalized = (x_data[i * x.shape_[1] + j] - mean) / stddev;
                    r_data[i * x.shape_[1] + j] = normalized * gamma_data[j] + beta_data[j];
                }
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type for layer_norm");
    }
    
    return result;
}

Tensor gelu(const Tensor& x) {
    Tensor result(x.shape_, x.dtype_);
    
    switch (x.dtype_) {
        case DataType::FLOAT32: {
            const float* x_data = x.data<float>();
            float* r_data = result.data<float>();
            for (size_t i = 0; i < x.size(); ++i) {
                // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                float x_val = x_data[i];
                float x_cubed = x_val * x_val * x_val;
                float inner = 0.7978845608f * (x_val + 0.044715f * x_cubed);
                r_data[i] = 0.5f * x_val * (1.0f + tanhf(inner));
            }
            break;
        }
        case DataType::FLOAT64: {
            const double* x_data = x.data<double>();
            double* r_data = result.data<double>();
            for (size_t i = 0; i < x.size(); ++i) {
                double x_val = x_data[i];
                double x_cubed = x_val * x_val * x_val;
                double inner = 0.7978845608 * (x_val + 0.044715 * x_cubed);
                r_data[i] = 0.5 * x_val * (1.0 + tanh(inner));
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type for gelu");
    }
    
    return result;
}

Tensor relu(const Tensor& x) {
    Tensor result(x.shape_, x.dtype_);
    
    switch (x.dtype_) {
        case DataType::FLOAT32: {
            const float* x_data = x.data<float>();
            float* r_data = result.data<float>();
            for (size_t i = 0; i < x.size(); ++i) {
                r_data[i] = std::max(0.0f, x_data[i]);
            }
            break;
        }
        case DataType::FLOAT64: {
            const double* x_data = x.data<double>();
            double* r_data = result.data<double>();
            for (size_t i = 0; i < x.size(); ++i) {
                r_data[i] = std::max(0.0, x_data[i]);
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type for relu");
    }
    
    return result;
}

}} // namespace inference
} // namespace openahi
