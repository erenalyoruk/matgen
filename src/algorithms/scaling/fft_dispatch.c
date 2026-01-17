/**
 * @file fft_dispatch.c
 * @brief Dispatch logic for FFT-based sparse matrix scaling
 */

#include "matgen/algorithms/scaling.h"
#include "matgen/core/execution/policy.h"
#include "matgen/core/matrix/csr.h"
#include "matgen/core/types.h"
#include "matgen/utils/log.h"

#include <math.h>

/* Backend headers */
#include "backends/seq/internal/fft_seq.h"

#ifdef MATGEN_HAS_CUDA
#include "backends/cuda/internal/fft_cuda.cuh"
#endif

// Default thresholds (binary vs general matrices)
#define FFT_THRESHOLD_BINARY 0.7
#define FFT_THRESHOLD_GENERAL 0.1

/**
 * @brief Detect if matrix is binary (all values ≈ 1.0)
 */
static bool is_binary_matrix(const matgen_csr_matrix_t* matrix) {
  const matgen_size_t sample_size = MATGEN_MIN(matrix->nnz, 10000);
  
  for (matgen_size_t i = 0; i < sample_size; i++) {
    if (fabs(matrix->values[i] - 1.0) > 1e-9) {
      return false;
    }
  }
  
  return true;
}

matgen_error_t matgen_scale_fft_with_policy(
    matgen_exec_policy_t policy,
    const matgen_csr_matrix_t* source,
    matgen_index_t new_rows,
    matgen_index_t new_cols,
    matgen_csr_matrix_t** result)
{
  if (!source || !result) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (new_rows == 0 || new_cols == 0) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  *result = NULL;

  // Auto-detect threshold based on matrix type
  bool is_binary = is_binary_matrix(source);
  matgen_value_t threshold = is_binary ? FFT_THRESHOLD_BINARY : FFT_THRESHOLD_GENERAL;

  MATGEN_LOG_DEBUG(
      "FFT scaling: %llu×%llu -> %llu×%llu (type: %s, threshold: %.3f)",
      (unsigned long long)source->rows, (unsigned long long)source->cols,
      (unsigned long long)new_rows, (unsigned long long)new_cols,
      is_binary ? "binary" : "general", threshold);

  // Resolve policy
  matgen_exec_policy_t resolved = matgen_exec_resolve(policy);

  MATGEN_LOG_DEBUG("FFT dispatch: policy=%d, resolved=%d", policy, resolved);

  switch (resolved) {
    case MATGEN_EXEC_SEQ:
      MATGEN_LOG_DEBUG("Using sequential FFT backend (FFTW3)");
      return matgen_scale_fft_seq(source, new_rows, new_cols, threshold, result);

#ifdef MATGEN_HAS_CUDA
    case MATGEN_EXEC_PAR_UNSEQ:
      MATGEN_LOG_DEBUG("Using CUDA FFT backend (cuFFT)");
      return matgen_scale_fft_cuda(source, new_rows, new_cols, threshold, result);
#endif

    case MATGEN_EXEC_PAR:
      // OpenMP not implemented for FFT, fall back to sequential
      MATGEN_LOG_DEBUG("OpenMP not implemented for FFT, using sequential");
      return matgen_scale_fft_seq(source, new_rows, new_cols, threshold, result);

    default:
      MATGEN_LOG_ERROR("Unsupported execution policy: %d", resolved);
      return MATGEN_ERROR_UNSUPPORTED;
  }
}

matgen_error_t matgen_scale_fft_with_policy_detailed(
    matgen_exec_policy_t policy,
    const matgen_csr_matrix_t* source,
    matgen_index_t new_rows,
    matgen_index_t new_cols,
    matgen_value_t threshold,
    matgen_csr_matrix_t** result)
{
  if (!source || !result) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (new_rows == 0 || new_cols == 0) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (threshold < 0.0 || threshold > 1.0) {
    MATGEN_LOG_ERROR("Invalid threshold: %.3f (must be in [0, 1])", threshold);
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  *result = NULL;

  MATGEN_LOG_DEBUG(
      "FFT scaling (custom threshold): %llu×%llu -> %llu×%llu (threshold: %.3f)",
      (unsigned long long)source->rows, (unsigned long long)source->cols,
      (unsigned long long)new_rows, (unsigned long long)new_cols, threshold);

  // Resolve policy
  matgen_exec_policy_t resolved = matgen_exec_resolve(policy);

  switch (resolved) {
    case MATGEN_EXEC_SEQ:
      return matgen_scale_fft_seq(source, new_rows, new_cols, threshold, result);

#ifdef MATGEN_HAS_CUDA
    case MATGEN_EXEC_PAR_UNSEQ:
      return matgen_scale_fft_cuda(source, new_rows, new_cols, threshold, result);
#endif

    case MATGEN_EXEC_PAR:
      MATGEN_LOG_DEBUG("OpenMP not available, using sequential");
      return matgen_scale_fft_seq(source, new_rows, new_cols, threshold, result);

    default:
      return MATGEN_ERROR_UNSUPPORTED;
  }
}