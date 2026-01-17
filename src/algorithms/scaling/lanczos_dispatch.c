/**
 * @file lanczos_dispatch.c
 * @brief Dispatch logic for Lanczos sparse matrix scaling
 */

#include "matgen/algorithms/scaling.h"
#include "matgen/core/execution/policy.h"
#include "matgen/utils/log.h"

/* Backend headers */
#include "backends/seq/internal/lanczos_seq.h"

#ifdef MATGEN_HAS_CUDA
#include "backends/cuda/internal/lanczos_cuda.cuh"
#endif

matgen_error_t matgen_scale_lanczos_with_policy(
    matgen_exec_policy_t policy,
    const matgen_csr_matrix_t* source,
    matgen_index_t new_rows,
    matgen_index_t new_cols,
    matgen_csr_matrix_t** result)
{
    if (!source || !result) {
        return MATGEN_ERROR_INVALID_ARGUMENT;
    }
    
    /* Lanczos requires square matrices */
    if (new_rows != new_cols) {
        MATGEN_LOG_ERROR("Lanczos scaling requires square output (rows == cols)");
        return MATGEN_ERROR_INVALID_ARGUMENT;
    }
    
    if (source->rows != source->cols) {
        MATGEN_LOG_ERROR("Lanczos scaling requires square input matrix");
        return MATGEN_ERROR_INVALID_ARGUMENT;
    }
    
    matgen_index_t new_size = new_rows;
    
    /* Resolve policy */
    matgen_exec_policy_t resolved = matgen_exec_resolve(policy);
    
    MATGEN_LOG_DEBUG("Lanczos dispatch: policy=%d, resolved=%d", policy, resolved);
    
    switch (resolved) {
        case MATGEN_EXEC_SEQ:
            MATGEN_LOG_DEBUG("Using sequential backend");
            return matgen_scale_lanczos_seq(source, new_size, result);
        
#ifdef MATGEN_HAS_CUDA
        case MATGEN_EXEC_PAR_UNSEQ:
            MATGEN_LOG_DEBUG("Using CUDA backend");
            return matgen_scale_lanczos_cuda(source, new_size, result);
#endif
        
        case MATGEN_EXEC_PAR:
            /* OpenMP not implemented, fall back to sequential */
            MATGEN_LOG_DEBUG("OpenMP not implemented for Lanczos, using sequential");
            return matgen_scale_lanczos_seq(source, new_size, result);
        
        default:
            MATGEN_LOG_DEBUG("Unsupported policy, using sequential");
            return matgen_scale_lanczos_seq(source, new_size, result);
    }
}
