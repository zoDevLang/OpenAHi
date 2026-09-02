#ifndef OPENAHI_INFERENCE_TENSOR_H
#define OPENAHI_INFERENCE_TENSOR_H

#include <vector>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <algorithm>

namespace openahi {
namespace inference {

/**
 * @brief Shape of a tensor (dimensions)
 */
class Shape {
public:
    std::vector<size_t> dims;
    
    Shape() = default;
    explicit Shape(const std::vector<size_t>& dims) : dims(dims) {}
    Shape(std::initializer_list<size_t> dims) : dims(dims) {}
    
    size_t size() const {
        size_t result = 1;
        for (auto d : dims) {
            result *= d;
        }
        return result;
    }
    
    size_t num_dims() const {
        return dims.size();
    }
    
    size_t& operator[](size_t index) { return dims[index]; }
    const size_t& operator[](size_t index) const { return dims[index]; }
    
    bool operator==(const Shape& other) const {
        return dims == other.dims;
    }
    
    bool operator!=(const Shape& other) const {
        return !(*this == other);
    }
    
    std::string to_string() const {
        std::string result = "(";
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) result += ", ";
            result += std::to_string(dims[i]);
        }
        result += ")";
        return result;
    }
};

/**
 * @brief Data type for tensor elements
 */
enum class DataType {
    FLOAT32,
    FLOAT64,
    INT32,
    INT64,
    UINT8,
    UNKNOWN
};

/**
 * @brief Tensor class for OpenAHI inference
 * 
 * Provides storage and basic operations for multi-dimensional arrays.
 */
class Tensor {
public:
    Tensor() : data_(nullptr), size_(0), dtype_(DataType::UNKNOWN) {}
    
    /**
     * Create a tensor with the given shape and data type
     */
    Tensor(const Shape& shape, DataType dtype = DataType::FLOAT32);
    
    /**
     * Create a tensor from existing data (takes ownership)
     */
    Tensor(void* data, const Shape& shape, DataType dtype);
    
    /**
     * Copy constructor
     */
    Tensor(const Tensor& other);
    
    /**
     * Move constructor
     */
    Tensor(Tensor&& other) noexcept;
    
    ~Tensor();
    
    /**
     * Copy assignment
     */
    Tensor& operator=(const Tensor& other);
    
    /**
     * Move assignment
     */
    Tensor& operator=(Tensor&& other) noexcept;
    
    /**
     * Get shape
     */
    const Shape& shape() const { return shape_; }
    
    /**
     * Get data type
     */
    DataType dtype() const { return dtype_; }
    
    /**
     * Get total number of elements
     */
    size_t size() const { return size_; }
    
    /**
     * Get raw data pointer
     */
    template <typename T>
    T* data() {
        return static_cast<T*>(data_);
    }
    
    template <typename T>
    const T* data() const {
        return static_cast<const T*>(data_);
    }
    
    /**
     * Access element at index (no bounds checking)
     */
    template <typename T>
    T& at(size_t index) {
        return data<T>()[index];
    }
    
    template <typename T>
    const T& at(size_t index) const {
        return data<T>()[index];
    }
    
    /**
     * Access element with bounds checking
     */
    template <typename T>
    T& operator()(size_t index) {
        if (index >= size_) {
            throw std::out_of_range("Tensor index out of range");
        }
        return data<T>()[index];
    }
    
    template <typename T>
    const T& operator()(size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("Tensor index out of range");
        }
        return data<T>()[index];
    }
    
    /**
     * Access multi-dimensional element
     */
    template <typename T>
    T& operator()(const std::vector<size_t>& indices) {
        size_t index = 0;
        size_t stride = 1;
        for (int i = shape_.num_dims() - 1; i >= 0; --i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Tensor index out of range");
            }
            index += indices[i] * stride;
            stride *= shape_[i];
        }
        return data<T>()[index];
    }
    
    /**
     * Fill tensor with a constant value
     */
    template <typename T>
    void fill(T value);
    
    /**
     * Zero out the tensor
     */
    void zero();
    
    /**
     * Get the size of the data type in bytes
     */
    size_t dtype_size() const;
    
    /**
     * Get total memory usage in bytes
     */
    size_t memory_usage() const {
        return size_ * dtype_size();
    }
    
    /**
     * Reshape the tensor (only if total size matches)
     */
    void reshape(const Shape& new_shape);
    
    /**
     * Flatten the tensor to 1D
     */
    Tensor flatten() const;
    
    /**
     * Create a transpose of the tensor
     */
    Tensor transpose() const;
    
    /**
     * Check if tensor is contiguous
     */
    bool is_contiguous() const { return true; } // Always contiguous in this simple implementation
    
    /**
     * Get a slice of the tensor
     */
    Tensor slice(size_t dim, size_t start, size_t end) const;
    
    /**
     * Print tensor info
     */
    void print_info(std::ostream& os = std::cout) const;

private:
    void* data_ = nullptr;
    size_t size_ = 0;
    Shape shape_;
    DataType dtype_ = DataType::UNKNOWN;
    bool owns_data_ = true;
    
    void allocate(size_t size, DataType dtype);
    void deallocate();
    void copy_data(const void* src, size_t size, DataType dtype);
};

/**
 * @brief Create a tensor filled with zeros
 */
Tensor zeros(const Shape& shape, DataType dtype = DataType::FLOAT32);

/**
 * @brief Create a tensor filled with ones
 */
Tensor ones(const Shape& shape, DataType dtype = DataType::FLOAT32);

/**
 * @brief Create a tensor with random values (uniform distribution)
 */
Tensor random_uniform(const Shape& shape, float min = 0.0f, float max = 1.0f, DataType dtype = DataType::FLOAT32);

/**
 * @brief Create a tensor with random values (normal distribution)
 */
Tensor random_normal(const Shape& shape, float mean = 0.0f, float stddev = 1.0f, DataType dtype = DataType::FLOAT32);

/**
 * @brief Matrix multiplication
 */
Tensor matmul(const Tensor& a, const Tensor& b);

/**
 * @brief Element-wise addition
 */
Tensor add(const Tensor& a, const Tensor& b);

/**
 * @brief Element-wise multiplication
 */
Tensor multiply(const Tensor& a, const Tensor& b);

/**
 * @brief Scalar multiplication
 */
Tensor scalar_multiply(const Tensor& a, float scalar);

/**
 * @brief Softmax operation
 */
Tensor softmax(const Tensor& x, int dim = -1);

/**
 * @brief Layer normalization
 */
Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps = 1e-5);

/**
 * @brief GELU activation function
 */
Tensor gelu(const Tensor& x);

/**
 * @brief ReLU activation function
 */
Tensor relu(const Tensor& x);

}} // namespace inference
} // namespace openahi

#endif // OPENAHI_INFERENCE_TENSOR_H
