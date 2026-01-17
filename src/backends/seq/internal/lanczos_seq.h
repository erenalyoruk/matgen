#ifndef MATGEN_BACKENDS_SEQ_INTERNAL_LANCZOS_SEQ_H
#define MATGEN_BACKENDS_SEQ_INTERNAL_LANCZOS_SEQ_H

/**
 * @file lanczos_seq.h
 * @brief Internal header for sequential Lanczos interpolation scaling
 *
 * This is an internal header used only by the library implementation.
 * Users should use the public API in <matgen/algorithms/scaling.h> instead.
 */

#include "matgen/core/matrix/csr.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Scale a CSR matrix using Lanczos interpolation (Sequential)
 *
 * Sequential CPU implementation of Lanczos sparse matrix scaling.
 * Preserves structural properties like bandwidth and symmetry.
 *
 * @param source Source CSR matrix (must be square)
 * @param new_size Target size (rows and cols, must be equal)
 * @param result Output CSR matrix
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_scale_lanczos_seq(const matgen_csr_matrix_t* source,
                                        matgen_index_t new_size,
                                        matgen_csr_matrix_t** result);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_BACKENDS_SEQ_INTERNAL_LANCZOS_SEQ_H
