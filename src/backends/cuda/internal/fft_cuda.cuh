#ifndef MATGEN_BACKENDS_CUDA_INTERNAL_FFT_CUDA_CUH
#define MATGEN_BACKENDS_CUDA_INTERNAL_FFT_CUDA_CUH

/**
 * @file fft_cuda.cuh
 * @brief CUDA FFT-based scaling backend (cuFFT)
 */

#include "matgen/core/matrix/csr.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CUDA FFT scaling implementation using cuFFT
 *
 * @param source Source CSR matrix
 * @param new_rows Target rows
 * @param new_cols Target columns
 * @param threshold Minimum absolute value to keep
 * @param result Output CSR matrix
 * @return MATGEN_SUCCESS or error code
 */
matgen_error_t matgen_scale_fft_cuda(
    const matgen_csr_matrix_t* source,
    matgen_index_t new_rows,
    matgen_index_t new_cols,
    matgen_value_t threshold,
    matgen_csr_matrix_t** result);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_BACKENDS_CUDA_INTERNAL_FFT_CUDA_CUH