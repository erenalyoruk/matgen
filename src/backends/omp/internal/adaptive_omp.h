#ifndef MATGEN_BACKENDS_OMP_INTERNAL_BILINEAR_OMP_H
#define MATGEN_BACKENDS_OMP_INTERNAL_BILINEAR_OMP_H

/**
 * @file adaptive_omp.h
 * @brief Internal header for OpenMP parallel bilinear interpolation
 *
 * This is an internal header used only by the library implementation.
 * Users should use the public API in <matgen/algorithms/scaling.h> instead.
 */

#ifdef MATGEN_HAS_OPENMP

#include "matgen/core/matrix/csr.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Scale sparse matrix using adaptive interpolation (OpenMP parallel)
 *
 * @param source Source matrix (CSR format)
 * @param new_rows Target number of rows
 * @param new_cols Target number of columns
 * @param result Output: scaled matrix (CSR format)
 * @return MATGEN_SUCCESS on success, error code otherwise
 */
matgen_error_t matgen_scale_adaptive_omp(const matgen_csr_matrix_t* source,
                                         matgen_index_t new_rows,
                                         matgen_index_t new_cols,
                                         matgen_csr_matrix_t** result);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_HAS_OPENMP

#endif  // MATGEN_BACKENDS_OMP_INTERNAL_BILINEAR_OMP_H
