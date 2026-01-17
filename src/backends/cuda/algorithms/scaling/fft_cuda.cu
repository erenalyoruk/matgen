/**
 * @file fft_cuda.cu
 * @brief CUDA FFT scaling implementation using cuFFT (optimized from your code)
 */

#include "backends/cuda/internal/fft_cuda.cuh"

#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>

#include "backends/cuda/internal/coo_cuda.cuh"
#include "backends/cuda/internal/conversion_cuda.cuh"
#include "backends/cuda/internal/csr_builder_cuda.cuh"
#include "matgen/core/matrix/coo.h"
#include "matgen/core/matrix/csr.h"
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

#define CUFFT_CHECK(call)                                             \
  do {                                                                \
    cufftResult err = call;                                           \
    if (err != CUFFT_SUCCESS) {                                       \
      MATGEN_LOG_ERROR("cuFFT error at %s:%d: code %d", __FILE__, __LINE__, err); \
      return MATGEN_ERROR_CUDA;                                       \
    }                                                                 \
  } while (0)

#define CUDA_BLOCK_SIZE 1024
#define FFT_BATCH_SIZE 1024  // Maximum GPU parallelism

// =============================================================================
// CUDA Kernels
// =============================================================================

/**
 * @brief Kernel to scale FFT frequencies (frequency domain zero-padding/truncation)
 */
__global__ void scale_fft_kernel(const cufftDoubleComplex* freq_in,
                                 cufftDoubleComplex* freq_out,
                                 matgen_index_t old_len,
                                 matgen_index_t new_len,
                                 matgen_value_t scale) {
  matgen_index_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < new_len) {
    freq_out[i].x = 0.0;
    freq_out[i].y = 0.0;

    matgen_index_t low = (old_len + 1) / 2;
    matgen_index_t high = old_len / 2;

    // Copy low frequencies
    if (i < low && i < old_len) {
      freq_out[i].x = freq_in[i].x * scale;
      freq_out[i].y = freq_in[i].y * scale;
    }

    // Copy high frequencies (wrap around)
    if (i >= new_len - high) {
      matgen_index_t old_idx = old_len - (new_len - i);
      if (old_idx < old_len) {
        freq_out[i].x = freq_in[old_idx].x * scale;
        freq_out[i].y = freq_in[old_idx].y * scale;
      }
    }
  }
}

/**
 * @brief GPU-side batched thresholding kernel
 * 
 * Processes multiple rows in parallel, storing entries above threshold
 */
__global__ void gpu_threshold_batch_kernel(
    const cufftDoubleComplex* d_fft_out,
    matgen_value_t* d_out_values,
    matgen_index_t* d_out_rows,
    matgen_index_t* d_out_cols,
    matgen_size_t* d_out_count,
    matgen_index_t new_c,
    matgen_index_t batch_size,
    matgen_value_t threshold,
    matgen_value_t norm_factor,
    matgen_index_t base_row,
    matgen_index_t new_r,
    matgen_index_t orig_rows,
    matgen_size_t max_output_size) {
  
  matgen_index_t col = blockIdx.x * blockDim.x + threadIdx.x;
  matgen_index_t row_in_batch = blockIdx.y;

  if (col < new_c && row_in_batch < batch_size) {
    matgen_size_t fft_idx = (matgen_size_t)row_in_batch * new_c + col;
    matgen_value_t val = (matgen_value_t)(d_fft_out[fft_idx].x * norm_factor);

    if (fabsf(val) >= threshold && fabsf(val) > 1e-15f) {
      matgen_size_t idx = atomicAdd((unsigned long long*)d_out_count, 1ULL);
      
      if (idx < max_output_size) {
        matgen_index_t orig_row = base_row + row_in_batch;
        matgen_index_t out_row = (matgen_index_t)(
            (matgen_value_t)orig_row * new_r / (matgen_value_t)orig_rows);
        
        d_out_values[idx] = val;
        d_out_rows[idx] = out_row;
        d_out_cols[idx] = col;
      }
    }
  }
}

// =============================================================================
// Main CUDA FFT Implementation
// =============================================================================

matgen_error_t matgen_scale_fft_cuda(const matgen_csr_matrix_t* source,
                                    matgen_index_t new_rows,
                                    matgen_index_t new_cols,
                                    matgen_value_t threshold,
                                    matgen_csr_matrix_t** result) {
  if (!source || !result) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  *result = NULL;

  MATGEN_LOG_DEBUG("FFT scaling (CUDA): %llu×%llu -> %llu×%llu (threshold=%.3f)",
                   (unsigned long long)source->rows,
                   (unsigned long long)source->cols,
                   (unsigned long long)new_rows, (unsigned long long)new_cols,
                   threshold);

  // Estimate output size
  matgen_value_t orig_density =
      (matgen_value_t)source->nnz /
      ((matgen_value_t)source->rows * (matgen_value_t)source->cols);
  matgen_size_t target_nnz =
      (matgen_size_t)(orig_density * (matgen_value_t)new_rows *
                      (matgen_value_t)new_cols * 2.0);
  if (target_nnz > 100000000) target_nnz = 100000000;

  MATGEN_LOG_DEBUG("Original density: %.6f%%, estimated NNZ: %zu",
                   orig_density * 100.0, target_nnz);

  // Allocate device memory for CSR source
  matgen_size_t* d_src_row_ptr = nullptr;
  matgen_index_t* d_src_col_indices = nullptr;
  matgen_value_t* d_src_values = nullptr;

  CUDA_CHECK(cudaMalloc(&d_src_row_ptr, (source->rows + 1) * sizeof(matgen_size_t)));
  CUDA_CHECK(cudaMalloc(&d_src_col_indices, source->nnz * sizeof(matgen_index_t)));
  CUDA_CHECK(cudaMalloc(&d_src_values, source->nnz * sizeof(matgen_value_t)));

  CUDA_CHECK(cudaMemcpy(d_src_row_ptr, source->row_ptr,
                        (source->rows + 1) * sizeof(matgen_size_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_src_col_indices, source->col_indices,
                        source->nnz * sizeof(matgen_index_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_src_values, source->values,
                        source->nnz * sizeof(matgen_value_t),
                        cudaMemcpyHostToDevice));

  // Adaptive batch size allocation with memory fallback
  matgen_index_t actual_batch_size = FFT_BATCH_SIZE;
  
  // Allocate FFT buffers with retry logic for large new_cols
  cufftDoubleComplex* d_fft_in = nullptr;
  cufftDoubleComplex* d_fft_freq = nullptr;
  cufftDoubleComplex* d_fft_scaled = nullptr;
  cufftDoubleComplex* d_fft_out = nullptr;

  // Try to allocate with full batch size, reduce if needed
  bool allocation_success = false;
  while (actual_batch_size >= 1 && !allocation_success) {
    cudaError_t err1 = cudaMalloc(&d_fft_in, sizeof(cufftDoubleComplex) * actual_batch_size * source->cols);
    cudaError_t err2 = cudaMalloc(&d_fft_freq, sizeof(cufftDoubleComplex) * actual_batch_size * source->cols);
    cudaError_t err3 = cudaMalloc(&d_fft_scaled, sizeof(cufftDoubleComplex) * actual_batch_size * new_cols);
    cudaError_t err4 = cudaMalloc(&d_fft_out, sizeof(cufftDoubleComplex) * actual_batch_size * new_cols);
    
    if (err1 == cudaSuccess && err2 == cudaSuccess && err3 == cudaSuccess && err4 == cudaSuccess) {
      allocation_success = true;
      MATGEN_LOG_DEBUG("FFT batch size: %llu", (unsigned long long)actual_batch_size);
    } else {
      // Free what was allocated and retry with smaller batch
      if (d_fft_in) cudaFree(d_fft_in);
      if (d_fft_freq) cudaFree(d_fft_freq);
      if (d_fft_scaled) cudaFree(d_fft_scaled);
      if (d_fft_out) cudaFree(d_fft_out);
      d_fft_in = d_fft_freq = d_fft_scaled = d_fft_out = nullptr;
      
      actual_batch_size = MATGEN_MAX((matgen_index_t)(actual_batch_size / 2), 1);
      
      if (actual_batch_size == 1) {
        MATGEN_LOG_ERROR("Cannot allocate FFT buffers even with batch size 1");
        return MATGEN_ERROR_OUT_OF_MEMORY;
      }
    }
  }

  // Allocate output buffers
  matgen_value_t* d_out_values = nullptr;
  matgen_index_t* d_out_rows = nullptr;
  matgen_index_t* d_out_cols = nullptr;
  matgen_size_t* d_out_count = nullptr;

  CUDA_CHECK(cudaMalloc(&d_out_values, target_nnz * sizeof(matgen_value_t)));
  CUDA_CHECK(cudaMalloc(&d_out_rows, target_nnz * sizeof(matgen_index_t)));
  CUDA_CHECK(cudaMalloc(&d_out_cols, target_nnz * sizeof(matgen_index_t)));
  CUDA_CHECK(cudaMalloc(&d_out_count, sizeof(matgen_size_t)));
  CUDA_CHECK(cudaMemset(d_out_count, 0, sizeof(matgen_size_t)));

  // Pinned host memory for faster transfers
  // (adjust after batch size is finalized)
  cufftDoubleComplex* h_complex_batch = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_complex_batch,
                            sizeof(cufftDoubleComplex) * actual_batch_size * source->cols));

  // Create cuFFT plans with actual batch size
  // Retry plan creation with smaller batch size if needed
  cufftHandle plan_batch_fwd, plan_batch_inv;
  bool plan_success = false;
  
  while (actual_batch_size >= 1 && !plan_success) {
    cufftResult plan_err_fwd = cufftPlan1d(&plan_batch_fwd, (int)source->cols, CUFFT_Z2Z, (int)actual_batch_size);
    cufftResult plan_err_inv = cufftPlan1d(&plan_batch_inv, (int)new_cols, CUFFT_Z2Z, (int)actual_batch_size);
    
    if (plan_err_fwd == CUFFT_SUCCESS && plan_err_inv == CUFFT_SUCCESS) {
      plan_success = true;
      MATGEN_LOG_DEBUG("cuFFT plans created with batch size: %llu", (unsigned long long)actual_batch_size);
    } else {
      // Plan creation failed, try with smaller batch size
      if (plan_err_fwd == CUFFT_SUCCESS) cufftDestroy(plan_batch_fwd);
      if (plan_err_inv == CUFFT_SUCCESS) cufftDestroy(plan_batch_inv);
      
      actual_batch_size = MATGEN_MAX((matgen_index_t)(actual_batch_size / 2), 1);
      
      if (actual_batch_size == 1) {
        MATGEN_LOG_ERROR("Cannot create cuFFT plans even with batch size 1");
        return MATGEN_ERROR_OUT_OF_MEMORY;
      }
      
      MATGEN_LOG_DEBUG("Retrying cuFFT plan creation with batch size: %llu", (unsigned long long)actual_batch_size);
    }
  }

  // CUDA streams for overlap
  cudaStream_t stream_compute, stream_transfer;
  CUDA_CHECK(cudaStreamCreate(&stream_compute));
  CUDA_CHECK(cudaStreamCreate(&stream_transfer));

  CUFFT_CHECK(cufftSetStream(plan_batch_fwd, stream_compute));
  CUFFT_CHECK(cufftSetStream(plan_batch_inv, stream_compute));

  MATGEN_LOG_DEBUG("Processing rows in batches of %llu...", (unsigned long long)actual_batch_size);

  // Batch processing loop
  for (matgen_index_t batch_start = 0; batch_start < source->rows;
       batch_start += actual_batch_size) {
    matgen_index_t batch_end = MATGEN_MIN(batch_start + actual_batch_size, source->rows);
    matgen_index_t current_batch_size = batch_end - batch_start;

    // Fill batch (sparse to dense)
    for (matgen_index_t b = 0; b < current_batch_size; b++) {
      matgen_index_t row = batch_start + b;

      // Zero-fill row
      for (matgen_index_t c = 0; c < source->cols; c++) {
        h_complex_batch[b * source->cols + c].x = 0.0;
        h_complex_batch[b * source->cols + c].y = 0.0;
      }

      // Fill non-zeros from CSR (this should use device-side kernel, but simplified here)
      for (matgen_size_t k = source->row_ptr[row]; k < source->row_ptr[row + 1]; k++) {
        matgen_index_t col = source->col_indices[k];
        matgen_value_t val = source->values[k];
        h_complex_batch[b * source->cols + col].x = (double)val;
      }
    }

    // Transfer to device
    CUDA_CHECK(cudaMemcpyAsync(d_fft_in, h_complex_batch,
                               sizeof(cufftDoubleComplex) * current_batch_size * source->cols,
                               cudaMemcpyHostToDevice, stream_transfer));
    CUDA_CHECK(cudaStreamSynchronize(stream_transfer));

    // Forward FFT (batched)
    CUFFT_CHECK(cufftExecZ2Z(plan_batch_fwd, d_fft_in, d_fft_freq, CUFFT_FORWARD));

    // Scale frequencies (custom kernel)
    {
      matgen_value_t scale = (matgen_value_t)new_cols / (matgen_value_t)source->cols;
      for (matgen_index_t b = 0; b < current_batch_size; b++) {
        int blocks = (new_cols + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        scale_fft_kernel<<<blocks, CUDA_BLOCK_SIZE, 0, stream_compute>>>(
            d_fft_freq + b * source->cols,
            d_fft_scaled + b * new_cols,
            source->cols, new_cols, scale);
      }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_compute));

    // Inverse FFT (batched)
    CUFFT_CHECK(cufftExecZ2Z(plan_batch_inv, d_fft_scaled, d_fft_out, CUFFT_INVERSE));
    CUDA_CHECK(cudaStreamSynchronize(stream_compute));

    // Threshold and store (GPU-side)
    {
      int blocks_x = (new_cols + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
      int blocks_y = current_batch_size;
      dim3 blocks(blocks_x, blocks_y);

      matgen_value_t norm_factor = 1.0f / new_cols;

      gpu_threshold_batch_kernel<<<blocks, CUDA_BLOCK_SIZE, 0, stream_compute>>>(
          d_fft_out, d_out_values, d_out_rows, d_out_cols, d_out_count,
          new_cols, current_batch_size, threshold, norm_factor,
          batch_start, new_rows, source->rows, target_nnz);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_compute));

    // Progress
    if ((batch_end % 1000) == 0 || batch_end == source->rows) {
      MATGEN_LOG_DEBUG("  Progress: %llu/%llu rows",
                       (unsigned long long)batch_end,
                       (unsigned long long)source->rows);
    }
  }

  // Get final NNZ count
  matgen_size_t actual_nnz = 0;
  CUDA_CHECK(cudaMemcpy(&actual_nnz, d_out_count, sizeof(matgen_size_t),
                        cudaMemcpyDeviceToHost));

  if (actual_nnz > target_nnz) {
    MATGEN_LOG_ERROR("Buffer overflow: %zu entries, buffer: %zu", actual_nnz, target_nnz);
    actual_nnz = target_nnz;
  }

  MATGEN_LOG_DEBUG("FFT transform complete: %zu entries", actual_nnz);

  // Create COO matrix
  matgen_coo_matrix_t* coo = matgen_coo_create_cuda(new_rows, new_cols, actual_nnz);
  if (!coo) {
    // Cleanup and return error
    cudaFree(d_src_row_ptr);
    cudaFree(d_src_col_indices);
    cudaFree(d_src_values);
    cudaFree(d_fft_in);
    cudaFree(d_fft_freq);
    cudaFree(d_fft_scaled);
    cudaFree(d_fft_out);
    cudaFree(d_out_values);
    cudaFree(d_out_rows);
    cudaFree(d_out_cols);
    cudaFree(d_out_count);
    cudaFreeHost(h_complex_batch);
    cufftDestroy(plan_batch_fwd);
    cufftDestroy(plan_batch_inv);
    cudaStreamDestroy(stream_compute);
    cudaStreamDestroy(stream_transfer);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  // Copy results to host
  CUDA_CHECK(cudaMemcpy(coo->row_indices, d_out_rows,
                        actual_nnz * sizeof(matgen_index_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(coo->col_indices, d_out_cols,
                        actual_nnz * sizeof(matgen_index_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(coo->values, d_out_values,
                        actual_nnz * sizeof(matgen_value_t),
                        cudaMemcpyDeviceToHost));

  coo->nnz = actual_nnz;
  coo->is_sorted = false;

  // Cleanup device memory
  cudaFree(d_src_row_ptr);
  cudaFree(d_src_col_indices);
  cudaFree(d_src_values);
  cudaFree(d_fft_in);
  cudaFree(d_fft_freq);
  cudaFree(d_fft_scaled);
  cudaFree(d_fft_out);
  cudaFree(d_out_values);
  cudaFree(d_out_rows);
  cudaFree(d_out_cols);
  cudaFree(d_out_count);
  cudaFreeHost(h_complex_batch);

  cufftDestroy(plan_batch_fwd);
  cufftDestroy(plan_batch_inv);
  cudaStreamDestroy(stream_compute);
  cudaStreamDestroy(stream_transfer);

  // Sort and sum duplicates (CUDA)
  matgen_error_t err = matgen_coo_sort_cuda(coo);
  if (err != MATGEN_SUCCESS) {
    matgen_coo_destroy(coo);
    return err;
  }

  err = matgen_coo_sum_duplicates_cuda(coo);
  if (err != MATGEN_SUCCESS) {
    matgen_coo_destroy(coo);
    return err;
  }

  MATGEN_LOG_DEBUG("After deduplication: %zu entries", coo->nnz);

  // Convert to CSR
  *result = matgen_coo_to_csr_cuda(coo);
  matgen_coo_destroy(coo);

  if (!*result) {
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  MATGEN_LOG_DEBUG("FFT scaling (CUDA) completed: output NNZ = %zu", (*result)->nnz);

  return MATGEN_SUCCESS;
}