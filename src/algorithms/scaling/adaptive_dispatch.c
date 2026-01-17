#include "matgen/algorithms/scaling.h"
#include "matgen/core/execution/dispatch.h"
#include "matgen/core/execution/policy.h"
#include "matgen/utils/log.h"

// Include backend-specific headers
#include "backends/seq/internal/adaptive_seq.h"

#ifdef MATGEN_HAS_OPENMP
#include "backends/omp/internal/adaptive_omp.h"
#endif


matgen_error_t matgen_scale_adaptive_with_policy(
    matgen_exec_policy_t policy, const matgen_csr_matrix_t* source,
    matgen_index_t new_rows, matgen_index_t new_cols,
    matgen_csr_matrix_t** result) {
  // Validate inputs
  if (source == NULL || result == NULL) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  // Handle auto policy: select based on problem size
  if (policy == MATGEN_EXEC_AUTO) {
    policy = matgen_exec_select_auto(source->nnz, source->rows, source->cols);
    MATGEN_LOG_DEBUG("Auto policy selected: %s",
                     matgen_exec_policy_name(policy));
  }

  // Resolve policy and create dispatch context
  matgen_exec_policy_t resolved = matgen_exec_resolve(policy);
  matgen_dispatch_context_t ctx = matgen_dispatch_create(resolved);

  // Dispatch to appropriate backend
  MATGEN_DISPATCH_BEGIN(ctx, "adaptive_scale") {
  MATGEN_DISPATCH_CASE_SEQ:
    return matgen_scale_adaptive_seq(source, new_rows, new_cols, result);

#ifdef MATGEN_HAS_OPENMP
  MATGEN_DISPATCH_CASE_PAR:
    return matgen_scale_adaptive_omp(source, new_rows, new_cols, result);
#endif

  MATGEN_DISPATCH_DEFAULT:
    // Fallback to sequential
    return matgen_scale_adaptive_seq(source, new_rows, new_cols, result);
  }
  MATGEN_DISPATCH_END();

  return MATGEN_ERROR_UNKNOWN;
}

matgen_error_t matgen_scale_adaptive_with_policy_detailed(
    const matgen_exec_policy_union_t* policy_union,
    const matgen_csr_matrix_t* source, matgen_index_t new_rows,
    matgen_index_t new_cols, matgen_csr_matrix_t** result) {
  // Validate inputs
  if (policy_union == NULL || source == NULL || result == NULL) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  // Create dispatch context from union
  matgen_dispatch_context_t ctx =
      matgen_dispatch_create_from_union(policy_union);

  // Dispatch to appropriate backend
  MATGEN_DISPATCH_BEGIN(ctx, "adaptive_scale") {
  MATGEN_DISPATCH_CASE_SEQ:
    return matgen_scale_adaptive_seq(source, new_rows, new_cols, result);

#ifdef MATGEN_HAS_OPENMP
  MATGEN_DISPATCH_CASE_PAR:
    // TODO: Accept num_threads parameter in matgen_scale_adaptive_omp()
    return matgen_scale_adaptive_omp(source, new_rows, new_cols, result);
#endif

  MATGEN_DISPATCH_DEFAULT:
    // Fallback to sequential
    return matgen_scale_adaptive_seq(source, new_rows, new_cols, result);
  }
  MATGEN_DISPATCH_END();

  return MATGEN_ERROR_UNKNOWN;
}
