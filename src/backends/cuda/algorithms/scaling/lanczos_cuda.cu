#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>
#include <vector>

#include "backends/cuda/internal/lanczos_cuda.cuh"
#include "backends/cuda/internal/conversion_cuda.cuh"
#include "backends/cuda/internal/coo_cuda.cuh"
#include "backends/cuda/internal/csr_builder_cuda.cuh"
#include "matgen/core/matrix/coo.h"
#include "matgen/core/matrix/csr.h"
#include "matgen/core/types.h"
#include "matgen/utils/log.h"

// =============================================================================
// CUDA Error Checking
// =============================================================================

#define CUDA_CHECK(call)                                              \
  do {                                                                \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
      MATGEN_LOG_ERROR("CUDA error at %s:%d: %s", __FILE__, __LINE__, \
                       cudaGetErrorString(err));                      \
      return MATGEN_ERROR_CUDA;                                       \
    }                                                                 \
  } while (0)

#define CUDA_BLOCK_SIZE 256
#define LANCZOS_A 3  // Kernel width parameter

// =============================================================================
// Device Helper Functions
// =============================================================================

__device__ __forceinline__ float lanczos_kernel_device(float x, int a) {
    if (x == 0.0f) return 1.0f;
    float ax = fabsf(x);
    if (ax >= (float)a) return 0.0f;
    
    const float PI = 3.14159265358979323846f;
    float pix = PI * x;
    float pix_over_a = pix / (float)a;
    float num = (float)a * __sinf(pix) * __sinf(pix_over_a);
    float den = (PI * PI) * (x * x);
    return num / den;
}

__device__ __forceinline__ float fetch_csr_value_device(
    const matgen_size_t* row_ptr,
    const matgen_index_t* col_indices,
    const matgen_value_t* values,
    matgen_index_t row, matgen_index_t col, matgen_index_t n)
{
    if (row >= n || col >= n) return 0.0f;
    
    matgen_size_t start = row_ptr[row];
    matgen_size_t end = row_ptr[row + 1];
    
    while (start < end) {
        matgen_size_t mid = start + ((end - start) >> 1);
        matgen_index_t c = col_indices[mid];
        if (c == col) return (float)values[mid];
        if (c < col) start = mid + 1;
        else end = mid;
    }
    return 0.0f;
}

// =============================================================================
// CUDA Kernels
// =============================================================================

/**
 * @brief Generate candidate positions for scaled matrix
 */
__global__ void lanczos_generate_candidates_kernel(
    const matgen_size_t* src_row_ptr,
    const matgen_index_t* src_col_indices,
    matgen_index_t src_size,
    matgen_index_t dst_size,
    float scale,
    int expansion_radius,
    matgen_index_t* out_rows,
    matgen_index_t* out_cols,
    matgen_size_t* out_count,
    matgen_size_t max_output)
{
    matgen_size_t nnz_idx = blockIdx.x * blockDim.x + threadIdx.x;
    matgen_size_t src_nnz = src_row_ptr[src_size];
    
    if (nnz_idx >= src_nnz) return;
    
    // Find row for this NNZ
    matgen_index_t src_row = 0;
    for (matgen_index_t r = 0; r < src_size; ++r) {
        if (src_row_ptr[r + 1] > nnz_idx) {
            src_row = r;
            break;
        }
    }
    matgen_index_t src_col = src_col_indices[nnz_idx];
    
    // Scale to new coordinates
    matgen_index_t base_row = (matgen_index_t)roundf(src_row * scale);
    matgen_index_t base_col = (matgen_index_t)roundf(src_col * scale);
    
    base_row = min(max((matgen_index_t)0, base_row), dst_size - 1);
    base_col = min(max((matgen_index_t)0, base_col), dst_size - 1);
    
    // Add center position
    matgen_size_t pos = atomicAdd((unsigned long long*)out_count, 1ULL);
    if (pos < max_output) {
        out_rows[pos] = base_row;
        out_cols[pos] = base_col;
    }
    
    // Add neighbors for upscaling
    if (expansion_radius > 0) {
        for (int dr = -expansion_radius; dr <= expansion_radius; ++dr) {
            for (int dc = -expansion_radius; dc <= expansion_radius; ++dc) {
                if (dr == 0 && dc == 0) continue;
                
                int nr = (int)base_row + dr;
                int nc = (int)base_col + dc;
                
                if (nr < 0 || nr >= (int)dst_size) continue;
                if (nc < 0 || nc >= (int)dst_size) continue;
                
                pos = atomicAdd((unsigned long long*)out_count, 1ULL);
                if (pos < max_output) {
                    out_rows[pos] = (matgen_index_t)nr;
                    out_cols[pos] = (matgen_index_t)nc;
                }
            }
        }
    }
}

/**
 * @brief Compute Lanczos interpolated values
 */
__global__ void lanczos_interpolate_kernel(
    const matgen_index_t* cand_rows,
    const matgen_index_t* cand_cols,
    matgen_size_t num_candidates,
    const matgen_size_t* src_row_ptr,
    const matgen_index_t* src_col_indices,
    const matgen_value_t* src_values,
    matgen_index_t src_size,
    float inv_scale,
    int a,
    matgen_value_t* out_values)
{
    matgen_size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;
    
    float new_row_f = (float)cand_rows[idx];
    float new_col_f = (float)cand_cols[idx];
    
    // Map back to source space
    float orig_row_f = new_row_f * inv_scale;
    float orig_col_f = new_col_f * inv_scale;
    
    int base_row = (int)floorf(orig_row_f);
    int base_col = (int)floorf(orig_col_f);
    
    float dx = orig_row_f - (float)base_row;
    float dy = orig_col_f - (float)base_col;
    
    float value_sum = 0.0f;
    float weight_sum = 0.0f;
    
    // Apply 2D Lanczos kernel
    for (int di = -a + 1; di <= a; ++di) {
        int sample_row = base_row + di;
        if (sample_row < 0 || sample_row >= (int)src_size) continue;
        
        float wx = lanczos_kernel_device(dx - (float)di, a);
        if (wx == 0.0f) continue;
        
        for (int dj = -a + 1; dj <= a; ++dj) {
            int sample_col = base_col + dj;
            if (sample_col < 0 || sample_col >= (int)src_size) continue;
            
            float wy = lanczos_kernel_device(dy - (float)dj, a);
            if (wy == 0.0f) continue;
            
            float weight = wx * wy;
            float orig_val = fetch_csr_value_device(
                src_row_ptr, src_col_indices, src_values,
                (matgen_index_t)sample_row, (matgen_index_t)sample_col, src_size);
            
            value_sum += orig_val * weight;
            weight_sum += weight;
        }
    }
    
    out_values[idx] = (weight_sum > 0.0f) ? (matgen_value_t)(value_sum / weight_sum) : 0.0;
}

// =============================================================================
// Host Helper: Remove duplicates and filter zeros
// =============================================================================

static void remove_duplicates_and_zeros(
    std::vector<matgen_index_t>& rows,
    std::vector<matgen_index_t>& cols,
    std::vector<matgen_value_t>& vals,
    matgen_value_t zero_eps)
{
    if (rows.empty()) return;
    
    size_t n = rows.size();
    
    // Create index array and sort by (row, col)
    std::vector<size_t> indices(n);
    for (size_t i = 0; i < n; ++i) indices[i] = i;
    
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        if (rows[a] != rows[b]) return rows[a] < rows[b];
        return cols[a] < cols[b];
    });
    
    // Compact: merge duplicates, skip zeros
    std::vector<matgen_index_t> new_rows, new_cols;
    std::vector<matgen_value_t> new_vals;
    new_rows.reserve(n);
    new_cols.reserve(n);
    new_vals.reserve(n);
    
    for (size_t i = 0; i < n; ++i) {
        size_t idx = indices[i];
        matgen_value_t v = vals[idx];
        
        if (fabs(v) < zero_eps) continue;
        
        if (!new_rows.empty() && 
            new_rows.back() == rows[idx] && 
            new_cols.back() == cols[idx]) {
            // Duplicate: sum values
            new_vals.back() += v;
        } else {
            new_rows.push_back(rows[idx]);
            new_cols.push_back(cols[idx]);
            new_vals.push_back(v);
        }
    }
    
    rows = std::move(new_rows);
    cols = std::move(new_cols);
    vals = std::move(new_vals);
}

// =============================================================================
// Main CUDA Implementation
// =============================================================================

matgen_error_t matgen_scale_lanczos_cuda(const matgen_csr_matrix_t* source,
                                         matgen_index_t new_size,
                                         matgen_csr_matrix_t** result)
{
    if (!source || !result) {
        return MATGEN_ERROR_INVALID_ARGUMENT;
    }
    
    // Lanczos requires square matrices
    if (source->rows != source->cols) {
        MATGEN_LOG_ERROR("Lanczos scaling requires square matrices");
        return MATGEN_ERROR_INVALID_ARGUMENT;
    }
    
    if (new_size == 0) {
        return MATGEN_ERROR_INVALID_ARGUMENT;
    }
    
    *result = NULL;
    
    matgen_index_t src_size = source->rows;
    float scale = (float)new_size / (float)src_size;
    float inv_scale = 1.0f / scale;
    
    MATGEN_LOG_DEBUG("Lanczos scaling (CUDA): %llu -> %llu (scale: %.3f)",
                     (unsigned long long)src_size,
                     (unsigned long long)new_size, scale);
    
    // Calculate expansion radius
    int expansion_radius = 0;
    if (scale > 1.0f) {
        expansion_radius = (int)ceilf(sqrtf(scale));
        if (expansion_radius < 2) expansion_radius = 2;
        if (expansion_radius > 6) expansion_radius = 6;
    }
    
    // Estimate output size
    size_t neighborhood_size = (2 * expansion_radius + 1) * (2 * expansion_radius + 1);
    size_t max_candidates = source->nnz * neighborhood_size;
    size_t size_limit = (size_t)new_size * (size_t)new_size;
    if (max_candidates > size_limit) max_candidates = size_limit;
    
    MATGEN_LOG_DEBUG("Max candidates: %zu (expansion radius: %d)", 
                     max_candidates, expansion_radius);
    
    // Allocate device memory for source
    matgen_size_t* d_src_row_ptr = nullptr;
    matgen_index_t* d_src_col_indices = nullptr;
    matgen_value_t* d_src_values = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_src_row_ptr, (src_size + 1) * sizeof(matgen_size_t)));
    CUDA_CHECK(cudaMalloc(&d_src_col_indices, source->nnz * sizeof(matgen_index_t)));
    CUDA_CHECK(cudaMalloc(&d_src_values, source->nnz * sizeof(matgen_value_t)));
    
    CUDA_CHECK(cudaMemcpy(d_src_row_ptr, source->row_ptr, 
                          (src_size + 1) * sizeof(matgen_size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_col_indices, source->col_indices,
                          source->nnz * sizeof(matgen_index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_values, source->values,
                          source->nnz * sizeof(matgen_value_t), cudaMemcpyHostToDevice));
    
    // Allocate candidate buffers
    matgen_index_t* d_cand_rows = nullptr;
    matgen_index_t* d_cand_cols = nullptr;
    matgen_size_t* d_cand_count = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_cand_rows, max_candidates * sizeof(matgen_index_t)));
    CUDA_CHECK(cudaMalloc(&d_cand_cols, max_candidates * sizeof(matgen_index_t)));
    CUDA_CHECK(cudaMalloc(&d_cand_count, sizeof(matgen_size_t)));
    CUDA_CHECK(cudaMemset(d_cand_count, 0, sizeof(matgen_size_t)));
    
    // Generate candidates
    int blocks = (source->nnz + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    
    lanczos_generate_candidates_kernel<<<blocks, CUDA_BLOCK_SIZE>>>(
        d_src_row_ptr, d_src_col_indices, src_size, new_size, scale,
        expansion_radius, d_cand_rows, d_cand_cols, d_cand_count, max_candidates);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Get candidate count
    matgen_size_t num_candidates;
    CUDA_CHECK(cudaMemcpy(&num_candidates, d_cand_count, sizeof(matgen_size_t),
                          cudaMemcpyDeviceToHost));
    
    if (num_candidates > max_candidates) num_candidates = max_candidates;
    
    MATGEN_LOG_DEBUG("Generated %zu candidates", (size_t)num_candidates);
    
    if (num_candidates == 0) {
        cudaFree(d_src_row_ptr);
        cudaFree(d_src_col_indices);
        cudaFree(d_src_values);
        cudaFree(d_cand_rows);
        cudaFree(d_cand_cols);
        cudaFree(d_cand_count);
        
        *result = matgen_csr_create(new_size, new_size, 0);
        return *result ? MATGEN_SUCCESS : MATGEN_ERROR_OUT_OF_MEMORY;
    }
    
    // Allocate output values
    matgen_value_t* d_out_values = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out_values, num_candidates * sizeof(matgen_value_t)));
    
    // Interpolate values
    blocks = (num_candidates + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    
    lanczos_interpolate_kernel<<<blocks, CUDA_BLOCK_SIZE>>>(
        d_cand_rows, d_cand_cols, num_candidates,
        d_src_row_ptr, d_src_col_indices, d_src_values,
        src_size, inv_scale, LANCZOS_A, d_out_values);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results to host
    std::vector<matgen_index_t> h_rows(num_candidates);
    std::vector<matgen_index_t> h_cols(num_candidates);
    std::vector<matgen_value_t> h_vals(num_candidates);
    
    CUDA_CHECK(cudaMemcpy(h_rows.data(), d_cand_rows, 
                          num_candidates * sizeof(matgen_index_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cols.data(), d_cand_cols,
                          num_candidates * sizeof(matgen_index_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vals.data(), d_out_values,
                          num_candidates * sizeof(matgen_value_t), cudaMemcpyDeviceToHost));
    
    // Free device memory
    cudaFree(d_src_row_ptr);
    cudaFree(d_src_col_indices);
    cudaFree(d_src_values);
    cudaFree(d_cand_rows);
    cudaFree(d_cand_cols);
    cudaFree(d_cand_count);
    cudaFree(d_out_values);
    
    // Remove duplicates and filter zeros on host
    remove_duplicates_and_zeros(h_rows, h_cols, h_vals, 1e-10);
    
    MATGEN_LOG_DEBUG("After deduplication: %zu entries", h_vals.size());
    
    if (h_vals.empty()) {
        *result = matgen_csr_create(new_size, new_size, 0);
        return *result ? MATGEN_SUCCESS : MATGEN_ERROR_OUT_OF_MEMORY;
    }
    
    // Build CSR matrix
    *result = matgen_csr_create(new_size, new_size, h_vals.size());
    if (!*result) {
        return MATGEN_ERROR_OUT_OF_MEMORY;
    }
    
    // Fill CSR arrays
    matgen_csr_matrix_t* out = *result;
    
    // Count entries per row
    std::vector<matgen_size_t> row_counts(new_size + 1, 0);
    for (size_t i = 0; i < h_rows.size(); ++i) {
        row_counts[h_rows[i] + 1]++;
    }
    
    // Compute row_ptr (prefix sum)
    out->row_ptr[0] = 0;
    for (matgen_index_t r = 0; r < new_size; ++r) {
        out->row_ptr[r + 1] = out->row_ptr[r] + row_counts[r + 1];
    }
    
    // Fill col_indices and values
    std::vector<matgen_size_t> row_offsets(new_size, 0);
    for (size_t i = 0; i < h_rows.size(); ++i) {
        matgen_index_t r = h_rows[i];
        matgen_size_t pos = out->row_ptr[r] + row_offsets[r];
        out->col_indices[pos] = h_cols[i];
        out->values[pos] = h_vals[i];
        row_offsets[r]++;
    }
    
    MATGEN_LOG_DEBUG("Lanczos scaling (CUDA) completed: output NNZ = %zu", out->nnz);
    
    return MATGEN_SUCCESS;
}
