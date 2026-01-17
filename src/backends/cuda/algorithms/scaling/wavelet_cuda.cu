/**
 * @file wavelet_cuda.cu
 * @brief CUDA implementation of wavelet-based matrix scaling
 *
 * Uses 2D Haar wavelet transform with block-based processing.
 * Optimized for GPU with Thrust sorting and parallel block processing.
 */

#ifdef MATGEN_HAS_CUDA

#include "backends/cuda/internal/wavelet_cuda.cuh"
#include "backends/cuda/internal/coo_cuda.cuh"
#include "backends/cuda/internal/conversion_cuda.cuh"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/tuple.h>

#include "matgen/core/matrix/coo.h"
#include "matgen/core/matrix/conversion.h"
#include "matgen/utils/log.h"

// =========================================================
// CONFIGURATION
// =========================================================
#define WAVELET_BLOCK_SIZE 4
#define WAVELET_THRESHOLD 1e-5f
#define WAVELET_MAX_SCALE 10
#define WAVELET_MAX_NEW_SIZE (WAVELET_BLOCK_SIZE * WAVELET_MAX_SCALE)

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            MATGEN_LOG_ERROR("CUDA Error: %s (Line %d)", cudaGetErrorString(err), __LINE__); \
            return MATGEN_ERROR_CUDA; \
        } \
    } while (0)

// =========================================================
// DEVICE FUNCTIONS - HAAR TRANSFORMS
// =========================================================

__device__ void haar_forward_1d_dev(float* data, int n) {
    float temp[WAVELET_MAX_NEW_SIZE];
    int half = n / 2;
    for (int i = 0; i < half; i++) {
        float a = data[2 * i];
        float b = data[2 * i + 1];
        temp[i] = (a + b) * 0.5f;
        temp[half + i] = (a - b) * 0.5f;
    }
    for (int i = 0; i < n; i++) data[i] = temp[i];
}

__device__ void haar_inverse_1d_dev(float* data, int n) {
    float temp[WAVELET_MAX_NEW_SIZE];
    int half = n / 2;
    for (int i = 0; i < half; i++) {
        float avg = data[i];
        float diff = data[half + i];
        temp[2 * i] = avg + diff;
        temp[2 * i + 1] = avg - diff;
    }
    for (int i = 0; i < n; i++) data[i] = temp[i];
}

__device__ void dwt_2d_level2_dev(float* block, int size) {
    float col_buf[WAVELET_MAX_NEW_SIZE];
    
    for (int r = 0; r < size; r++) haar_forward_1d_dev(block + r * size, size);
    for (int c = 0; c < size; c++) {
        for (int r = 0; r < size; r++) col_buf[r] = block[r * size + c];
        haar_forward_1d_dev(col_buf, size);
        for (int r = 0; r < size; r++) block[r * size + c] = col_buf[r];
    }
    
    int half = size / 2;
    for (int r = 0; r < half; r++) haar_forward_1d_dev(block + r * size, half);
    for (int c = 0; c < half; c++) {
        for (int r = 0; r < half; r++) col_buf[r] = block[r * size + c];
        haar_forward_1d_dev(col_buf, half);
        for (int r = 0; r < half; r++) block[r * size + c] = col_buf[r];
    }
}

__device__ void idwt_2d_level2_dev(float* block, int size) {
    float col_buf[WAVELET_MAX_NEW_SIZE];
    int half = size / 2;
    
    for (int c = 0; c < half; c++) {
        for (int r = 0; r < half; r++) col_buf[r] = block[r * size + c];
        haar_inverse_1d_dev(col_buf, half);
        for (int r = 0; r < half; r++) block[r * size + c] = col_buf[r];
    }
    for (int r = 0; r < half; r++) haar_inverse_1d_dev(block + r * size, half);
    
    for (int c = 0; c < size; c++) {
        for (int r = 0; r < size; r++) col_buf[r] = block[r * size + c];
        haar_inverse_1d_dev(col_buf, size);
        for (int r = 0; r < size; r++) block[r * size + c] = col_buf[r];
    }
    for (int r = 0; r < size; r++) haar_inverse_1d_dev(block + r * size, size);
}

// =========================================================
// CUDA KERNELS
// =========================================================

__global__ void compute_block_ids_kernel(
    const matgen_index_t* rows, const matgen_index_t* cols, 
    matgen_size_t nnz, matgen_index_t grid_w, matgen_index_t* block_ids
) {
    matgen_size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    matgen_index_t br = rows[idx] / WAVELET_BLOCK_SIZE;
    matgen_index_t bc = cols[idx] / WAVELET_BLOCK_SIZE;
    block_ids[idx] = br * grid_w + bc;
}

__global__ void fill_dense_blocks_kernel(
    const matgen_index_t* rows, const matgen_index_t* cols, const float* vals,
    const matgen_size_t* block_starts, const matgen_size_t* block_counts,
    int num_blocks, float* dense_batch
) {
    int blk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk_idx >= num_blocks) return;
    
    matgen_size_t start = block_starts[blk_idx];
    matgen_size_t count = block_counts[blk_idx];
    int base = blk_idx * WAVELET_BLOCK_SIZE * WAVELET_BLOCK_SIZE;
    
    // Zero out
    for (int i = 0; i < WAVELET_BLOCK_SIZE * WAVELET_BLOCK_SIZE; i++) 
        dense_batch[base + i] = 0.0f;
    
    // Fill
    for (matgen_size_t i = 0; i < count; i++) {
        matgen_size_t idx = start + i;
        int r = rows[idx] % WAVELET_BLOCK_SIZE;
        int c = cols[idx] % WAVELET_BLOCK_SIZE;
        dense_batch[base + r * WAVELET_BLOCK_SIZE + c] = vals[idx];
    }
}

__global__ void process_wavelet_blocks_kernel(
    float* dense_batch, int num_blocks,
    float scale_factor, int new_block_size,
    float* out_batch
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;
    
    float tile[WAVELET_BLOCK_SIZE * WAVELET_BLOCK_SIZE];
    int base = idx * WAVELET_BLOCK_SIZE * WAVELET_BLOCK_SIZE;
    for (int i = 0; i < WAVELET_BLOCK_SIZE * WAVELET_BLOCK_SIZE; i++) 
        tile[i] = dense_batch[base + i];
    
    dwt_2d_level2_dev(tile, WAVELET_BLOCK_SIZE);
    
    float resized[WAVELET_MAX_NEW_SIZE * WAVELET_MAX_NEW_SIZE];
    for (int i = 0; i < new_block_size * new_block_size; i++) resized[i] = 0.0f;
    
    // Simple nearest-neighbor resize in wavelet domain
    for (int r = 0; r < new_block_size; r++) {
        for (int c = 0; c < new_block_size; c++) {
            int src_r = (int)floorf((float)r / scale_factor);
            int src_c = (int)floorf((float)c / scale_factor);
            if (src_r >= WAVELET_BLOCK_SIZE) src_r = WAVELET_BLOCK_SIZE - 1;
            if (src_c >= WAVELET_BLOCK_SIZE) src_c = WAVELET_BLOCK_SIZE - 1;
            resized[r * new_block_size + c] = tile[src_r * WAVELET_BLOCK_SIZE + src_c];
        }
    }
    
    idwt_2d_level2_dev(resized, new_block_size);
    
    int out_base = idx * new_block_size * new_block_size;
    for (int i = 0; i < new_block_size * new_block_size; i++) 
        out_batch[out_base + i] = resized[i];
}

__global__ void count_nnz_kernel(
    float* out_batch, int num_blocks, int block_area, matgen_size_t* counts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;
    int start = idx * block_area;
    matgen_size_t count = 0;
    for (int i = 0; i < block_area; i++) {
        if (fabsf(out_batch[start + i]) > WAVELET_THRESHOLD) count++;
    }
    counts[idx] = count;
}

__global__ void scatter_sparse_kernel(
    float* out_batch, matgen_index_t* block_ids, matgen_size_t* offsets,
    int num_blocks, int new_block_size, matgen_index_t orig_width,
    float scale_factor, matgen_index_t target_rows, matgen_index_t target_cols,
    matgen_index_t* out_rows, matgen_index_t* out_cols, float* out_vals
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;
    
    matgen_index_t block_id = block_ids[idx];
    matgen_index_t grid_w = (orig_width + WAVELET_BLOCK_SIZE - 1) / WAVELET_BLOCK_SIZE;
    matgen_index_t old_br = block_id / grid_w;
    matgen_index_t old_bc = block_id % grid_w;
    
    matgen_index_t new_start_r = (matgen_index_t)(old_br * WAVELET_BLOCK_SIZE * scale_factor);
    matgen_index_t new_start_c = (matgen_index_t)(old_bc * WAVELET_BLOCK_SIZE * scale_factor);
    
    int read_off = idx * new_block_size * new_block_size;
    matgen_size_t write_off = offsets[idx];
    matgen_size_t k = 0;
    
    for (int r = 0; r < new_block_size; r++) {
        for (int c = 0; c < new_block_size; c++) {
            float val = out_batch[read_off + r * new_block_size + c];
            matgen_index_t final_r = new_start_r + r;
            matgen_index_t final_c = new_start_c + c;
            
            if (fabsf(val) > WAVELET_THRESHOLD && final_r < target_rows && final_c < target_cols) {
                out_rows[write_off + k] = final_r;
                out_cols[write_off + k] = final_c;
                out_vals[write_off + k] = val;
                k++;
            }
        }
    }
}

__global__ void compute_sort_keys_kernel(
    const matgen_index_t* rows, const matgen_index_t* cols,
    matgen_size_t total_nnz, matgen_index_t num_cols,
    unsigned long long* sort_keys
) {
    matgen_size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_nnz) return;
    sort_keys[idx] = (unsigned long long)rows[idx] * (unsigned long long)(num_cols + 1) + (unsigned long long)cols[idx];
}

// =========================================================
// MAIN IMPLEMENTATION
// =========================================================

extern "C" matgen_error_t matgen_scale_wavelet_cuda(
    const matgen_csr_matrix_t* source,
    matgen_index_t new_rows,
    matgen_index_t new_cols,
    matgen_csr_matrix_t** result
) {
    if (source == NULL || result == NULL) {
        return MATGEN_ERROR_INVALID_ARGUMENT;
    }

    float scale_factor = (float)new_rows / (float)source->rows;
    
    if (scale_factor > WAVELET_MAX_SCALE) {
        MATGEN_LOG_ERROR("Scale factor %.2f exceeds maximum %d", scale_factor, WAVELET_MAX_SCALE);
        return MATGEN_ERROR_INVALID_ARGUMENT;
    }

    MATGEN_LOG_INFO("CUDA Wavelet scaling: %llux%llu -> %llux%llu (scale=%.2f)",
                    (unsigned long long)source->rows, (unsigned long long)source->cols,
                    (unsigned long long)new_rows, (unsigned long long)new_cols,
                    scale_factor);

    matgen_size_t nnz = source->nnz;
    matgen_index_t rows = source->rows;
    matgen_index_t cols = source->cols;

    // CUDA warmup
    cudaFree(0);

    // Allocate and copy CSR data to device
    matgen_index_t *d_csr_rows, *d_cols_arr;
    float *d_vals;
    matgen_index_t *d_block_ids;

    // Convert CSR row_ptr to explicit row indices
    matgen_index_t* h_row_indices = (matgen_index_t*)malloc(nnz * sizeof(matgen_index_t));
    for (matgen_index_t row = 0; row < rows; row++) {
        for (matgen_size_t j = source->row_ptr[row]; j < source->row_ptr[row + 1]; j++) {
            h_row_indices[j] = row;
        }
    }

    CHECK_CUDA(cudaMalloc(&d_csr_rows, nnz * sizeof(matgen_index_t)));
    CHECK_CUDA(cudaMalloc(&d_cols_arr, nnz * sizeof(matgen_index_t)));
    CHECK_CUDA(cudaMalloc(&d_vals, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_block_ids, nnz * sizeof(matgen_index_t)));

    CHECK_CUDA(cudaMemcpy(d_csr_rows, h_row_indices, nnz * sizeof(matgen_index_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cols_arr, source->col_indices, nnz * sizeof(matgen_index_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals, source->values, nnz * sizeof(float), cudaMemcpyHostToDevice));

    free(h_row_indices);

    // Compute block IDs
    matgen_index_t grid_w = (cols + WAVELET_BLOCK_SIZE - 1) / WAVELET_BLOCK_SIZE;
    int threads = 256;
    int blocks = (nnz + threads - 1) / threads;
    
    compute_block_ids_kernel<<<blocks, threads>>>(d_csr_rows, d_cols_arr, nnz, grid_w, d_block_ids);

    // Sort by block ID using Thrust
    thrust::device_ptr<matgen_index_t> t_block_ids(d_block_ids);
    thrust::device_ptr<matgen_index_t> t_rows(d_csr_rows);
    thrust::device_ptr<matgen_index_t> t_cols(d_cols_arr);
    thrust::device_ptr<float> t_vals(d_vals);

    auto values_zip = thrust::make_zip_iterator(thrust::make_tuple(t_rows, t_cols, t_vals));
    thrust::sort_by_key(t_block_ids, t_block_ids + nnz, values_zip);

    // Reduce to find unique blocks
    matgen_index_t *d_unique_blocks;
    matgen_size_t *d_block_counts, *d_block_starts;
    
    CHECK_CUDA(cudaMalloc(&d_unique_blocks, nnz * sizeof(matgen_index_t)));
    CHECK_CUDA(cudaMalloc(&d_block_counts, nnz * sizeof(matgen_size_t)));
    CHECK_CUDA(cudaMalloc(&d_block_starts, nnz * sizeof(matgen_size_t)));

    thrust::device_ptr<matgen_index_t> t_unique_blocks(d_unique_blocks);
    thrust::device_ptr<matgen_size_t> t_block_counts(d_block_counts);
    thrust::constant_iterator<matgen_size_t> const_one(1);

    auto end_pair = thrust::reduce_by_key(
        t_block_ids, t_block_ids + nnz,
        const_one,
        t_unique_blocks,
        t_block_counts
    );

    int num_blocks_proc = end_pair.first - t_unique_blocks;

    thrust::device_ptr<matgen_size_t> t_block_starts(d_block_starts);
    thrust::exclusive_scan(t_block_counts, t_block_counts + num_blocks_proc, t_block_starts);

    // Calculate new block size
    int new_size = (int)(WAVELET_BLOCK_SIZE * scale_factor);
    if (new_size % 2 != 0) new_size++;

    // Allocate processing buffers
    float *d_dense, *d_out;
    CHECK_CUDA(cudaMalloc(&d_dense, num_blocks_proc * WAVELET_BLOCK_SIZE * WAVELET_BLOCK_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, num_blocks_proc * new_size * new_size * sizeof(float)));

    int proc_blocks = (num_blocks_proc + threads - 1) / threads;

    fill_dense_blocks_kernel<<<proc_blocks, threads>>>(
        d_csr_rows, d_cols_arr, d_vals, d_block_starts, d_block_counts, num_blocks_proc, d_dense
    );

    process_wavelet_blocks_kernel<<<proc_blocks, threads>>>(
        d_dense, num_blocks_proc, scale_factor, new_size, d_out
    );

    // Count NNZ
    matgen_size_t *d_nnz_counts, *d_offsets;
    CHECK_CUDA(cudaMalloc(&d_nnz_counts, num_blocks_proc * sizeof(matgen_size_t)));
    CHECK_CUDA(cudaMalloc(&d_offsets, num_blocks_proc * sizeof(matgen_size_t)));

    count_nnz_kernel<<<proc_blocks, threads>>>(d_out, num_blocks_proc, new_size * new_size, d_nnz_counts);

    thrust::device_ptr<matgen_size_t> t_nnz_counts(d_nnz_counts);
    thrust::device_ptr<matgen_size_t> t_offsets(d_offsets);
    thrust::exclusive_scan(t_nnz_counts, t_nnz_counts + num_blocks_proc, t_offsets);

    matgen_size_t total_nnz = thrust::reduce(t_nnz_counts, t_nnz_counts + num_blocks_proc);

    MATGEN_LOG_INFO("CUDA Wavelet: output NNZ = %llu", (unsigned long long)total_nnz);

    // Allocate output
    matgen_index_t *d_out_rows, *d_out_cols;
    float *d_out_vals;
    CHECK_CUDA(cudaMalloc(&d_out_rows, total_nnz * sizeof(matgen_index_t)));
    CHECK_CUDA(cudaMalloc(&d_out_cols, total_nnz * sizeof(matgen_index_t)));
    CHECK_CUDA(cudaMalloc(&d_out_vals, total_nnz * sizeof(float)));

    scatter_sparse_kernel<<<proc_blocks, threads>>>(
        d_out, d_unique_blocks, d_offsets, num_blocks_proc, new_size, cols,
        scale_factor, new_rows, new_cols,
        d_out_rows, d_out_cols, d_out_vals
    );

    // Sort output by (row, col) on GPU for proper COO order
    // Allocate sort keys on device
    unsigned long long* d_sort_keys;
    CHECK_CUDA(cudaMalloc(&d_sort_keys, total_nnz * sizeof(unsigned long long)));
    
    // Compute sort keys using kernel
    int sort_threads = 256;
    int sort_blocks = (total_nnz + sort_threads - 1) / sort_threads;
    compute_sort_keys_kernel<<<sort_blocks, sort_threads>>>(d_out_rows, d_out_cols, total_nnz, new_cols, d_sort_keys);
    
    // Sort by keys using Thrust
    thrust::device_ptr<unsigned long long> t_sort_keys(d_sort_keys);
    thrust::device_ptr<matgen_index_t> t_out_rows(d_out_rows);
    thrust::device_ptr<matgen_index_t> t_out_cols(d_out_cols);
    thrust::device_ptr<float> t_out_vals(d_out_vals);
    
    auto out_vals_zip = thrust::make_zip_iterator(thrust::make_tuple(t_out_rows, t_out_cols, t_out_vals));
    thrust::sort_by_key(t_sort_keys, t_sort_keys + total_nnz, out_vals_zip);
    
    cudaFree(d_sort_keys);

    // Copy sorted data back to host
    matgen_index_t* h_out_rows = (matgen_index_t*)malloc(total_nnz * sizeof(matgen_index_t));
    matgen_index_t* h_out_cols = (matgen_index_t*)malloc(total_nnz * sizeof(matgen_index_t));
    float* h_out_vals = (float*)malloc(total_nnz * sizeof(float));

    CHECK_CUDA(cudaMemcpy(h_out_rows, d_out_rows, total_nnz * sizeof(matgen_index_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_out_cols, d_out_cols, total_nnz * sizeof(matgen_index_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_out_vals, d_out_vals, total_nnz * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup GPU
    cudaFree(d_csr_rows); cudaFree(d_cols_arr); cudaFree(d_vals); cudaFree(d_block_ids);
    cudaFree(d_unique_blocks); cudaFree(d_block_counts); cudaFree(d_block_starts);
    cudaFree(d_dense); cudaFree(d_out);
    cudaFree(d_nnz_counts); cudaFree(d_offsets);
    cudaFree(d_out_rows); cudaFree(d_out_cols); cudaFree(d_out_vals);

    // Create COO and convert to CSR
    matgen_coo_matrix_t* coo = matgen_coo_create(new_rows, new_cols, total_nnz);
    if (!coo) {
        free(h_out_rows); free(h_out_cols); free(h_out_vals);
        return MATGEN_ERROR_OUT_OF_MEMORY;
    }

    memcpy(coo->row_indices, h_out_rows, total_nnz * sizeof(matgen_index_t));
    memcpy(coo->col_indices, h_out_cols, total_nnz * sizeof(matgen_index_t));
    memcpy(coo->values, h_out_vals, total_nnz * sizeof(float));
    coo->nnz = total_nnz;

    free(h_out_rows); free(h_out_cols); free(h_out_vals);

    *result = matgen_coo_to_csr(coo);
    matgen_coo_destroy(coo);

    if (*result == NULL) {
        return MATGEN_ERROR_OUT_OF_MEMORY;
    }

    return MATGEN_SUCCESS;
}

#endif  // MATGEN_HAS_CUDA
