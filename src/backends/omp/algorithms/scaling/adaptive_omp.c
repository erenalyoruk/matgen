#include "backends/seq/internal/adaptive_seq.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "matgen/core/execution/policy.h"
#include "matgen/core/matrix/conversion.h"
#include "matgen/core/matrix/coo.h"
#include "matgen/core/types.h"
#include "matgen/utils/log.h"
#include "matgen/utils/triplet_buffer.h"

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
matgen_error_t matgen_scale_adaptive_omp(const matgen_csr_matrix_t* source,
                                         matgen_index_t new_rows,
                                         matgen_index_t new_cols,
                                         matgen_csr_matrix_t** result) {
  if (!source || !result) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (new_rows == 0 || new_cols == 0) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  *result = NULL;

  matgen_value_t row_scale =
      (matgen_value_t)new_rows / (matgen_value_t)source->rows;
  matgen_value_t col_scale =
      (matgen_value_t)new_cols / (matgen_value_t)source->cols;

  MATGEN_LOG_DEBUG(
      "Adaptive scaling (SEQ): %zu×%zu -> %zu×%zu (scale: %.3fx%.3f)",
      source->rows, source->cols, new_rows, new_cols, row_scale, col_scale);  
  
  //Creeate memory for the worst possible scenario which is all bilinear resizes
  matgen_value_t max_row_contrib = ceilf((2.0F * row_scale) + 2.0F);
  matgen_value_t max_col_contrib = ceilf((2.0F * col_scale) + 2.0F);
  matgen_value_t max_contributions_per_source =
      max_row_contrib * max_col_contrib;

  // Use 1.5x safety factor for edge cases and rounding
  size_t estimated_nnz = (size_t)((matgen_value_t)source->nnz *
                                  max_contributions_per_source * 1.5F);

  // Ensure minimum buffer size
  if (estimated_nnz < source->nnz * 4) {
    estimated_nnz = source->nnz * 4;
  }
  
  MATGEN_LOG_DEBUG("Estimated output NNZ: %zu", estimated_nnz);

  // Create triplet buffer
  matgen_triplet_buffer_t* global_buffer = matgen_triplet_buffer_create(estimated_nnz);
  matgen_error_t global_err = MATGEN_SUCCESS;

  // 2. Start Parallel Region
  #pragma omp parallel
  {
    // Each thread gets its own small temporary buffer to avoid locking the global one
    // We estimate a portion of the total size for each thread
    int num_threads = omp_get_num_threads();
    matgen_triplet_buffer_t* local_buffer = matgen_triplet_buffer_create(estimated_nnz / num_threads);

    #pragma omp for schedule(dynamic) // Dynamic schedule is best for adaptive density
    for (matgen_index_t src_row = 0; src_row < source->rows; src_row++) {
      size_t row_start = source->row_ptr[src_row];
      size_t row_end = source->row_ptr[src_row + 1];
      size_t local_nnz = row_end - row_start;

      for (size_t idx = row_start; idx < row_end; idx++) {
        matgen_index_t src_col = source->col_indices[idx];
        matgen_value_t src_val = source->values[idx];

        if (src_val == 0.0) continue;
        // --- BILINEAR BLOCK ---
        // Determine contribution method based on local density heuristic
        // Here we use a simple heuristic: if the source entry is in a dense
        // region (many non-zeros nearby), use bilinear; otherwise, use nearest
        // neighbor.
        size_t local_nnz = row_end - row_start;
        if (local_nnz > 5) { 
          // Dense region: Bilinear interpolation
          // Each source entry at (src_row, src_col) contributes to destination
          // cells in range:
          // - Row range: [(src_row - 1) * row_scale, (src_row + 1) * row_scale]
          // - Col range: [(src_col - 1) * col_scale, (src_col + 1) * col_scale]
          matgen_index_t y0 = (matgen_index_t)floorf((matgen_value_t)(src_row) *
                                                    row_scale);
          matgen_index_t y1 = (matgen_index_t)ceilf((matgen_value_t)(src_row + 1) *
                                                    row_scale);
          matgen_index_t x0 = (matgen_index_t)floorf((matgen_value_t)(src_col) *
                                                    col_scale);
          matgen_index_t x1 = (matgen_index_t)ceilf((matgen_value_t)(src_col + 1) *
                                                    col_scale);

          for (matgen_index_t dst_row = y0; dst_row <= y1; dst_row++) {
            for (matgen_index_t dst_col = x0; dst_col <= x1; dst_col++) {
              // Compute relative position in source space
              matgen_value_t src_y =
                  (matgen_value_t)(dst_row) / row_scale;
              matgen_value_t src_x =
                  (matgen_value_t)(dst_col) / col_scale;

              matgen_value_t dy = src_y - (matgen_value_t)(src_row);
              matgen_value_t dx = src_x - (matgen_value_t)(src_col);

              // Clamp to [0, 1]
              dy = MATGEN_CLAMP(dy, 0.0, 1.0);
              dx = MATGEN_CLAMP(dx, 0.0, 1.0);

              // Determine which of the 4 neighbors we are and calculate bilinear
              // weight
              matgen_value_t weight = (matgen_value_t)0.0;

              if (dst_row == y0 && dst_col == x0) {
                // Bottom-left neighbor
                weight = (matgen_value_t)(1.0 - dy) * (matgen_value_t)(1.0 - dx);
              } else if (dst_row == y0 && dst_col == x1) {
                // Bottom-right neighbor
                weight = (matgen_value_t)(1.0 - dy) * dx;
              } else if (dst_row == y1 && dst_col == x0) {
                // Top-left neighbor
                weight = dy * (matgen_value_t)(1.0 - dx);
              } else if (dst_row == y1 && dst_col == x1) {  
                // Top-right neighbor
                weight = dy * dx;
              }
              if (weight > 0.0) {
                matgen_triplet_buffer_add(local_buffer, dst_row, dst_col, src_val * weight);
              }
            }
          }
        } else {
          // --- NEAREST NEIGHBOR BLOCK ---
          matgen_index_t dst_row = (matgen_index_t)roundf((matgen_value_t)src_row * row_scale);
          matgen_index_t dst_col = (matgen_index_t)roundf((matgen_value_t)src_col * col_scale);
          dst_row = MATGEN_CLAMP(dst_row, 0, new_rows - 1);
          dst_col = MATGEN_CLAMP(dst_col, 0, new_cols - 1);

          matgen_triplet_buffer_add(local_buffer, dst_row, dst_col, src_val);
        }
      }
    }

 // 3. Merge Local Buffers into Global Buffer
    #pragma omp critical
    {
        // We manually loop through the local buffer and add each entry to the global one
        for (size_t i = 0; i < local_buffer->size; i++) {
            matgen_triplet_buffer_add(
                global_buffer, 
                local_buffer->rows[i], 
                local_buffer->cols[i], 
                local_buffer->vals[i]
            );
        }
        
        // Clean up the local memory now that we are done with it
        matgen_triplet_buffer_destroy(local_buffer);
    }
  }

  size_t total_triplets = matgen_triplet_buffer_size(global_buffer);
  MATGEN_LOG_DEBUG("Generated %zu triplets", total_triplets);

  // Create COO matrix using sequential backend explicitly
  matgen_coo_matrix_t* coo =
      matgen_coo_create(new_rows, new_cols, total_triplets);
  if (!coo) {
    MATGEN_LOG_ERROR("Failed to create COO matrix");
    matgen_triplet_buffer_destroy(global_buffer);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  // Copy triplets to COO matrix
  memcpy(coo->row_indices, global_buffer->rows,
         total_triplets * sizeof(matgen_index_t));
  memcpy(coo->col_indices, global_buffer->cols,
         total_triplets * sizeof(matgen_index_t));
  memcpy(coo->col_indices, global_buffer->cols,
         total_triplets * sizeof(matgen_index_t));
  memcpy(coo->values, global_buffer->vals,
         total_triplets * sizeof(matgen_value_t));
  coo->nnz = total_triplets;
  coo->is_sorted = false;

  matgen_triplet_buffer_destroy(global_buffer);

  // Sort and sum duplicates using sequential policy
  MATGEN_LOG_DEBUG("Sorting and summing duplicates...");
  global_err = matgen_coo_sort_with_policy(coo, MATGEN_EXEC_SEQ);
  if (global_err != MATGEN_SUCCESS) {
    MATGEN_LOG_ERROR("Failed to sort COO matrix");
    matgen_coo_destroy(coo);
    return global_err;
  }

  global_err = matgen_coo_sum_duplicates_with_policy(coo, MATGEN_EXEC_SEQ);
  if (global_err != MATGEN_SUCCESS) {
    MATGEN_LOG_ERROR("Failed to sum duplicates");
    matgen_coo_destroy(coo);
    return global_err;
  }

  MATGEN_LOG_DEBUG("After deduplication: %zu entries", coo->nnz);

  // Convert to CSR using sequential policy
  *result = matgen_coo_to_csr_with_policy(coo, MATGEN_EXEC_SEQ);
  matgen_coo_destroy(coo);

  if (!(*result)) {
    MATGEN_LOG_ERROR("Failed to convert COO to CSR matrix");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  MATGEN_LOG_DEBUG("Bilinear scaling (SEQ) completed: output NNZ = %zu",
                   (*result)->nnz);

  return MATGEN_SUCCESS;
}
