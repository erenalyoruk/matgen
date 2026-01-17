#ifndef MATGEN_BACKENDS_CUDA_INTERNAL_WAVELET_CUDA_H
#define MATGEN_BACKENDS_CUDA_INTERNAL_WAVELET_CUDA_H

/**
 * @file wavelet_cuda.cuh
 * @brief Internal header for CUDA wavelet-based scaling
 *
 * This is an internal header used only by the library implementation.
 * Users should use the public API in <matgen/algorithms/scaling.h> instead.
 */

#ifdef MATGEN_HAS_CUDA

#include "matgen/core/matrix/csr.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Scale a CSR matrix using wavelet-based interpolation (CUDA)
 *
 * GPU-accelerated implementation using block-based 2D Haar transforms.
 * Each CUDA thread processes one block independently.
 *
 * Algorithm:
 *   1. Sort elements by block ID on GPU
 *   2. Extract unique blocks via reduce_by_key
 *   3. Parallel kernel: load block, DWT, resize, IDWT per thread
 *   4. Count NNZ per block, prefix sum for output positions
 *   5. Scatter sparse output and convert to CSR
 *
 * @param source Source CSR matrix
 * @param new_rows Target number of rows
 * @param new_cols Target number of columns
 * @param result Output CSR matrix
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_scale_wavelet_cuda(const matgen_csr_matrix_t* source,
                                         matgen_index_t new_rows,
                                         matgen_index_t new_cols,
                                         matgen_csr_matrix_t** result);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_HAS_CUDA

#endif  // MATGEN_BACKENDS_CUDA_INTERNAL_WAVELET_CUDA_H
