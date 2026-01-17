#ifndef MATGEN_ALGORITHMS_SCALING_H
#define MATGEN_ALGORITHMS_SCALING_H

/**
 * @file scaling.h
 * @brief Sparse matrix scaling algorithms with execution policy support
 *
 * This is the main public interface for matrix scaling operations.
 * It provides execution policy-aware functions that automatically dispatch
 * to the appropriate backend (Sequential, OpenMP, CUDA, MPI).
 */

#include "matgen/core/execution/policy.h"
#include "matgen/core/matrix/csr.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Bilinear Interpolation Scaling
// =============================================================================

/**
 * @brief Scale sparse matrix using bilinear interpolation with execution policy
 *
 * Automatically dispatches to the appropriate backend based on the execution
 * policy. This is the recommended entry point for bilinear scaling.
 *
 * Bilinear interpolation distributes each source entry's value to neighboring
 * target cells using bilinear weights. Values are accumulated (summed) at each
 * target cell.
 *
 * @param policy Execution policy (MATGEN_EXEC_SEQ, MATGEN_EXEC_PAR, etc.)
 * @param source Source matrix (CSR format)
 * @param new_rows Target number of rows
 * @param new_cols Target number of columns
 * @param result Output: scaled matrix (CSR format)
 * @return MATGEN_SUCCESS on success, error code otherwise
 *
 * @note The policy is resolved at runtime. If the requested backend is not
 *       available, it will fall back to the next best option.
 *
 * @example
 * ```c
 * // Use automatic selection based on problem size
 * matgen_exec_policy_t policy = matgen_exec_select_auto(
 *     source->nnz, source->rows, source->cols);
 * matgen_scale_bilinear_with_policy(policy, source, 2000, 2000, &result);
 *
 * // Or explicitly request CUDA
 * matgen_scale_bilinear_with_policy(MATGEN_EXEC_PAR_UNSEQ, source,
 *                                   2000, 2000, &result);
 * ```
 */
matgen_error_t matgen_scale_bilinear_with_policy(
    matgen_exec_policy_t policy, const matgen_csr_matrix_t* source,
    matgen_index_t new_rows, matgen_index_t new_cols,
    matgen_csr_matrix_t** result);
    
   
/**
 * @brief Scale sparse matrix with detailed execution policy parameters
 *
 * Like matgen_scale_bilinear_with_policy(), but accepts a policy union
 * that can contain detailed parameters (e.g., thread count, CUDA device).
 *
 * @param policy_union Policy union with detailed parameters
 * @param source Source matrix (CSR format)
 * @param new_rows Target number of rows
 * @param new_cols Target number of columns
 * @param result Output: scaled matrix (CSR format)
 * @return MATGEN_SUCCESS on success, error code otherwise
 *
 * @example
 * ```c
 * // Use OpenMP with 8 threads
 * matgen_exec_par_t par_policy = matgen_exec_par_with_threads(8);
 * matgen_exec_policy_union_t policy_union;
 * policy_union.par = par_policy;
 * matgen_scale_bilinear_with_policy_detailed(&policy_union, source,
 *                                            2000, 2000, &result);
 * ```
 */
matgen_error_t matgen_scale_bilinear_with_policy_detailed(
    const matgen_exec_policy_union_t* policy_union,
    const matgen_csr_matrix_t* source, matgen_index_t new_rows,
    matgen_index_t new_cols, matgen_csr_matrix_t** result);

// =============================================================================
// Nearest Neighbor Scaling
// =============================================================================

/**
 * @brief Scale sparse matrix using nearest neighbor interpolation with
 * execution policy
 *
 * Automatically dispatches to the appropriate backend based on the execution
 * policy. Uses SUM collision policy by default.
 *
 * Nearest neighbor interpolation maps each source entry to a block of target
 * cells, distributing the value uniformly across the block.
 *
 * @param policy Execution policy (MATGEN_EXEC_SEQ, MATGEN_EXEC_PAR, etc.)
 * @param source Source matrix (CSR format)
 * @param new_rows Target number of rows
 * @param new_cols Target number of columns
 * @param result Output: scaled matrix (CSR format)
 * @return MATGEN_SUCCESS on success, error code otherwise
 */
matgen_error_t matgen_scale_nearest_neighbor_with_policy(
    matgen_exec_policy_t policy, const matgen_csr_matrix_t* source,
    matgen_index_t new_rows, matgen_index_t new_cols,
    matgen_csr_matrix_t** result);

/**
 * @brief Scale sparse matrix using nearest neighbor with collision policy
 *
 * Like matgen_scale_nearest_neighbor_with_policy(), but allows specifying
 * how to handle collisions when multiple source entries map to the same
 * target cell.
 *
 * @param policy Execution policy (MATGEN_EXEC_SEQ, MATGEN_EXEC_PAR, etc.)
 * @param source Source matrix (CSR format)
 * @param new_rows Target number of rows
 * @param new_cols Target number of columns
 * @param collision_policy How to handle collisions (SUM, AVG, MAX, MIN, LAST)
 * @param result Output: scaled matrix (CSR format)
 * @return MATGEN_SUCCESS on success, error code otherwise
 */
matgen_error_t matgen_scale_nearest_neighbor_with_policy_detailed(
    matgen_exec_policy_t policy, const matgen_csr_matrix_t* source,
    matgen_index_t new_rows, matgen_index_t new_cols,
    matgen_collision_policy_t collision_policy, matgen_csr_matrix_t** result);
    
// =============================================================================
// Lanczos Interpolation Scaling
// =============================================================================

/**
 * @brief Scale sparse matrix using Lanczos interpolation with execution policy
 *
 * @param policy Execution policy
 * @param source Source matrix (CSR format, must be square)
 * @param new_rows Target number of rows (must equal new_cols)
 * @param new_cols Target number of columns (must equal new_rows)
 * @param result Output: scaled matrix (CSR format)
 * @return MATGEN_SUCCESS on success, error code otherwise
 */
matgen_error_t matgen_scale_lanczos_with_policy(
    matgen_exec_policy_t policy,
    const matgen_csr_matrix_t* source,
    matgen_index_t new_rows,
    matgen_index_t new_cols,
    matgen_csr_matrix_t** result);
/**
 * @brief Scale sparse matrix using FFT-based interpolation with execution policy
 *
 * FFT scaling uses frequency domain interpolation to resize sparse matrices while
 * preserving spectral characteristics. Suitable for both upscaling and downscaling.
 *
 * Algorithm characteristics:
 * - Uses 1D FFT on rows, then 1D FFT on columns (separable transform)
 * - Maintains matrix density through adaptive thresholding
 * - Excellent for preserving frequency-domain features
 * - Memory-efficient batched processing for large matrices
 *
 * Requirements:
 * - Sequential backend: Requires FFTW3 library
 * - CUDA backend: Requires cuFFT library
 *
 * @param policy Execution policy (MATGEN_EXEC_SEQ, MATGEN_EXEC_PAR_UNSEQ for CUDA)
 * @param source Source matrix (CSR format)
 * @param new_rows Target number of rows
 * @param new_cols Target number of columns
 * @param result Output: scaled matrix (CSR format)
 * @return MATGEN_SUCCESS on success, error code otherwise
 *
 * @note For binary matrices (values near 1.0), uses threshold=0.7
 *       For general matrices, uses threshold=0.1
 *
 * @example
 * ```c
 * // Use CUDA for large matrices
 * matgen_scale_fft_with_policy(MATGEN_EXEC_PAR_UNSEQ, source,
 *                              4000, 4000, &result);
 *
 * // Use sequential CPU with FFTW3
 * matgen_scale_fft_with_policy(MATGEN_EXEC_SEQ, source,
 *                              2000, 2000, &result);
 * ```
 */
matgen_error_t matgen_scale_fft_with_policy(
    matgen_exec_policy_t policy,
    const matgen_csr_matrix_t* source,
    matgen_index_t new_rows,
    matgen_index_t new_cols,
    matgen_csr_matrix_t** result);

/**
 * @brief Scale sparse matrix using FFT with custom threshold
 *
 * Like matgen_scale_fft_with_policy(), but allows manual threshold control.
 *
 * @param policy Execution policy
 * @param source Source matrix (CSR format)
 * @param new_rows Target number of rows
 * @param new_cols Target number of columns
 * @param threshold Minimum absolute value to keep (0.0 to 1.0)
 * @param result Output: scaled matrix (CSR format)
 * @return MATGEN_SUCCESS on success, error code otherwise
 *
 * @note Lower threshold = denser output, higher threshold = sparser output
 */
matgen_error_t matgen_scale_fft_with_policy_detailed(
    matgen_exec_policy_t policy,
    const matgen_csr_matrix_t* source,
    matgen_index_t new_rows,
    matgen_index_t new_cols,
    matgen_value_t threshold,
    matgen_csr_matrix_t** result);
    
// =============================================================================
// Wavelet-Based Scaling
// =============================================================================

/**
 * @brief Scale sparse matrix using wavelet-based interpolation with execution policy
 *
 * Uses 2D Haar wavelet transform with block-based processing to scale sparse matrices.
 * The algorithm preserves structural characteristics by working in the wavelet domain:
 *   1. Groups non-zeros into BLOCK_SIZE×BLOCK_SIZE blocks
 *   2. Applies 2-level 2D Haar DWT to each block
 *   3. Resizes wavelet coefficients to target block size
 *   4. Applies inverse DWT to reconstruct scaled blocks
 *   5. Sparsifies output using threshold
 *
 * @param policy Execution policy (MATGEN_EXEC_SEQ, MATGEN_EXEC_PAR_UNSEQ for CUDA)
 * @param source Source matrix (CSR format)
 * @param new_rows Target number of rows
 * @param new_cols Target number of columns
 * @param result Output: scaled matrix (CSR format)
 * @return MATGEN_SUCCESS on success, error code otherwise
 *
 * @note Block size is fixed at 4×4. Scale factors up to 4x are supported.
 *       For 10x scaling, the algorithm uses larger internal buffers.
 */
matgen_error_t matgen_scale_wavelet_with_policy(
    matgen_exec_policy_t policy,
    const matgen_csr_matrix_t* source,
    matgen_index_t new_rows,
    matgen_index_t new_cols,
    matgen_csr_matrix_t** result);

// =============================================================================
// Adaptive Scaling
// =============================================================================

/**
 * @brief Scale sparse matrix using adaptive logic with execution policy
 *
 * Automatically dispatches to the appropriate backend based on the execution
 * policy. Uses SUM collision policy by default.
 *
 * Adaptive scaling uses a combination of nearest neighbor and bilinear
 * interpolation based on the size ratio between source and target matrices.
 *
 * @param policy Execution policy (MATGEN_EXEC_SEQ, MATGEN_EXEC_PAR, etc.)
 * @param source Source matrix (CSR format)
 * @param new_rows Target number of rows
 * @param new_cols Target number of columns
 * @param result Output: scaled matrix (COO format)
 * @return MATGEN_SUCCESS on success, error code otherwise
 */
matgen_error_t matgen_scale_adaptive_with_policy(
    matgen_exec_policy_t policy, const matgen_csr_matrix_t* source,
    matgen_index_t new_rows, matgen_index_t new_cols,
    matgen_csr_matrix_t** result);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_ALGORITHMS_SCALING_H
