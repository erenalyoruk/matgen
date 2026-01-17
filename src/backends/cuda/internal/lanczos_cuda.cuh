#ifndef MATGEN_BACKENDS_CUDA_INTERNAL_LANCZOS_CUDA_H
#define MATGEN_BACKENDS_CUDA_INTERNAL_LANCZOS_CUDA_H

/**
 * @file lanczos_cuda.cuh
 * @brief Internal header for CUDA Lanczos interpolation scaling
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
 * @brief Scale a CSR matrix using Lanczos interpolation (CUDA)
 *
 * GPU-accelerated implementation of Lanczos sparse matrix scaling.
 * Preserves structural properties like bandwidth and symmetry.
 *
 * @param source Source CSR matrix (must be square)
 * @param new_size Target size (rows and cols, must be equal)
 * @param result Output CSR matrix
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_scale_lanczos_cuda(const matgen_csr_matrix_t* source,
                                         matgen_index_t new_size,
                                         matgen_csr_matrix_t** result);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_HAS_CUDA

#endif  // MATGEN_BACKENDS_CUDA_INTERNAL_LANCZOS_CUDA_H
