/**
 * @file fft_seq.c
 * @brief Sequential FFT scaling implementation using FFTW3
 */

#include "backends/seq/internal/fft_seq.h"

#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "matgen/core/matrix/coo.h"
#include "matgen/core/matrix/conversion.h"
#include "matgen/core/matrix/csr.h"
#include "matgen/utils/log.h"

#define FFT_BATCH_SIZE 256  // Process 256 rows per batch

// =============================================================================
// FFT Context (reusable plans)
// =============================================================================

typedef struct {
  matgen_index_t old_len;
  matgen_index_t new_len;
  fftw_complex* in;
  fftw_complex* freq;
  fftw_complex* freq_scaled;
  fftw_complex* out;
  fftw_plan plan_forward;
  fftw_plan plan_inverse;
} fft_context_t;

static fft_context_t* fft_context_create(matgen_index_t old_len,
                                         matgen_index_t new_len) {
  fft_context_t* ctx = (fft_context_t*)malloc(sizeof(fft_context_t));
  if (!ctx) return NULL;

  ctx->old_len = old_len;
  ctx->new_len = new_len;

  ctx->in = fftw_malloc(sizeof(fftw_complex) * old_len);
  ctx->freq = fftw_malloc(sizeof(fftw_complex) * old_len);
  ctx->freq_scaled = fftw_malloc(sizeof(fftw_complex) * new_len);
  ctx->out = fftw_malloc(sizeof(fftw_complex) * new_len);

  if (!ctx->in || !ctx->freq || !ctx->freq_scaled || !ctx->out) {
    if (ctx->in) fftw_free(ctx->in);
    if (ctx->freq) fftw_free(ctx->freq);
    if (ctx->freq_scaled) fftw_free(ctx->freq_scaled);
    if (ctx->out) fftw_free(ctx->out);
    free(ctx);
    return NULL;
  }

  ctx->plan_forward = fftw_plan_dft_1d((int)old_len, ctx->in, ctx->freq,
                                       FFTW_FORWARD, FFTW_ESTIMATE);
  ctx->plan_inverse = fftw_plan_dft_1d((int)new_len, ctx->freq_scaled, ctx->out,
                                       FFTW_BACKWARD, FFTW_ESTIMATE);

  return ctx;
}

static void fft_context_destroy(fft_context_t* ctx) {
  if (!ctx) return;
  fftw_destroy_plan(ctx->plan_forward);
  fftw_destroy_plan(ctx->plan_inverse);
  fftw_free(ctx->in);
  fftw_free(ctx->freq);
  fftw_free(ctx->freq_scaled);
  fftw_free(ctx->out);
  free(ctx);
}

// =============================================================================
// FFT Interpolation (1D)
// =============================================================================

static void fft_interpolate_1d(fft_context_t* ctx, const matgen_value_t* src,
                               matgen_value_t* dst) {
  // Fill input with source data
  for (matgen_index_t i = 0; i < ctx->old_len; i++) {
    ctx->in[i][0] = (double)src[i];
    ctx->in[i][1] = 0.0;
  }

  // Forward FFT
  fftw_execute(ctx->plan_forward);

  // Zero-pad or truncate frequency domain
  memset(ctx->freq_scaled, 0, sizeof(fftw_complex) * ctx->new_len);

  matgen_index_t low = (ctx->old_len + 1) / 2;
  matgen_index_t high = ctx->old_len / 2;

  // Copy low frequencies
  for (matgen_index_t i = 0; i < low && i < ctx->new_len; i++) {
    ctx->freq_scaled[i][0] = ctx->freq[i][0];
    ctx->freq_scaled[i][1] = ctx->freq[i][1];
  }

  // Copy high frequencies (wrap around)
  for (matgen_index_t i = 0; i < high && i < ctx->new_len; i++) {
    matgen_index_t old_idx = ctx->old_len - high + i;
    matgen_index_t new_idx = ctx->new_len - high + i;
    ctx->freq_scaled[new_idx][0] = ctx->freq[old_idx][0];
    ctx->freq_scaled[new_idx][1] = ctx->freq[old_idx][1];
  }

  // Scale frequencies
  matgen_value_t scale = (matgen_value_t)ctx->new_len / (matgen_value_t)ctx->old_len;
  for (matgen_index_t i = 0; i < ctx->new_len; i++) {
    ctx->freq_scaled[i][0] *= scale;
    ctx->freq_scaled[i][1] *= scale;
  }

  // Inverse FFT
  fftw_execute(ctx->plan_inverse);

  // Normalize and copy output
  matgen_value_t norm = 1.0 / (matgen_value_t)ctx->new_len;
  for (matgen_index_t i = 0; i < ctx->new_len; i++) {
    dst[i] = (matgen_value_t)(ctx->out[i][0] * norm);
  }
}

// =============================================================================
// Main FFT Scaling Implementation
// =============================================================================

matgen_error_t matgen_scale_fft_seq(const matgen_csr_matrix_t* source,
                                   matgen_index_t new_rows,
                                   matgen_index_t new_cols,
                                   matgen_value_t threshold,
                                   matgen_csr_matrix_t** result) {
  if (!source || !result) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  MATGEN_LOG_DEBUG("FFT scaling (SEQ): %llu×%llu -> %llu×%llu (threshold=%.3f)",
                   (unsigned long long)source->rows,
                   (unsigned long long)source->cols,
                   (unsigned long long)new_rows, (unsigned long long)new_cols,
                   threshold);

  *result = NULL;

  // Estimate output size
  matgen_value_t orig_density =
      (matgen_value_t)source->nnz /
      ((matgen_value_t)source->rows * (matgen_value_t)source->cols);
  matgen_size_t target_nnz =
      (matgen_size_t)(orig_density * (matgen_value_t)new_rows *
                      (matgen_value_t)new_cols * 1.5);
  if (target_nnz > 100000000) target_nnz = 100000000;

  MATGEN_LOG_DEBUG("Original density: %.6f%%, estimated NNZ: %zu",
                   orig_density * 100.0, target_nnz);

  // Allocate result buffer
  matgen_index_t* out_rows = (matgen_index_t*)malloc(sizeof(matgen_index_t) * target_nnz);
  matgen_index_t* out_cols = (matgen_index_t*)malloc(sizeof(matgen_index_t) * target_nnz);
  matgen_value_t* out_vals = (matgen_value_t*)malloc(sizeof(matgen_value_t) * target_nnz);

  if (!out_rows || !out_cols || !out_vals) {
    free(out_rows);
    free(out_cols);
    free(out_vals);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  matgen_size_t total_nnz = 0;

  // Create FFT context for rows
  fft_context_t* row_ctx = fft_context_create(source->cols, new_cols);
  if (!row_ctx) {
    free(out_rows);
    free(out_cols);
    free(out_vals);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  // Batch processing
  matgen_value_t** row_scaled =
      (matgen_value_t**)malloc(sizeof(matgen_value_t*) * FFT_BATCH_SIZE);
  matgen_value_t* col_buf = (matgen_value_t*)malloc(sizeof(matgen_value_t) * FFT_BATCH_SIZE);

  MATGEN_LOG_DEBUG("Processing rows in batches of %d...", FFT_BATCH_SIZE);

  for (matgen_index_t batch_start = 0; batch_start < source->rows;
       batch_start += FFT_BATCH_SIZE) {
    matgen_index_t batch_end =
        MATGEN_MIN(batch_start + FFT_BATCH_SIZE, source->rows);
    matgen_index_t batch_len = batch_end - batch_start;

    // Row-wise FFT
    for (matgen_index_t b = 0; b < batch_len; b++) {
      matgen_index_t src_row = batch_start + b;

      // Extract sparse row to dense
      matgen_value_t* sparse_row =
          (matgen_value_t*)calloc(source->cols, sizeof(matgen_value_t));
      for (matgen_size_t idx = source->row_ptr[src_row];
           idx < source->row_ptr[src_row + 1]; idx++) {
        sparse_row[source->col_indices[idx]] = source->values[idx];
      }

      // Apply row FFT
      row_scaled[b] = (matgen_value_t*)malloc(sizeof(matgen_value_t) * new_cols);
      fft_interpolate_1d(row_ctx, sparse_row, row_scaled[b]);
      free(sparse_row);
    }

    // Calculate output row range
    matgen_index_t out_row_start =
        (matgen_index_t)((matgen_value_t)batch_start * (matgen_value_t)new_rows /
                        (matgen_value_t)source->rows);
    matgen_index_t out_row_end =
        (matgen_index_t)((matgen_value_t)batch_end * (matgen_value_t)new_rows /
                        (matgen_value_t)source->rows);
    matgen_index_t out_batch_rows = MATGEN_MAX(out_row_end - out_row_start, 1);

    // Create column FFT context
    fft_context_t* col_ctx = fft_context_create(batch_len, out_batch_rows);
    matgen_value_t* col_out = (matgen_value_t*)malloc(sizeof(matgen_value_t) * out_batch_rows);

    // Column-wise FFT with thresholding
    for (matgen_index_t c = 0; c < new_cols; c++) {
      // Extract column
      for (matgen_index_t b = 0; b < batch_len; b++) {
        col_buf[b] = row_scaled[b][c];
      }

      // Apply column FFT
      fft_interpolate_1d(col_ctx, col_buf, col_out);

      // Store values above threshold
      for (matgen_index_t r = 0; r < out_batch_rows; r++) {
        matgen_value_t val = col_out[r];
        if (fabs(val) >= threshold && fabs(val) > 1e-15) {
          if (total_nnz >= target_nnz) {
            // Reallocate if needed
            target_nnz = (matgen_size_t)(target_nnz * 1.5);
            out_rows = (matgen_index_t*)realloc(out_rows, sizeof(matgen_index_t) * target_nnz);
            out_cols = (matgen_index_t*)realloc(out_cols, sizeof(matgen_index_t) * target_nnz);
            out_vals = (matgen_value_t*)realloc(out_vals, sizeof(matgen_value_t) * target_nnz);
          }
          out_rows[total_nnz] = out_row_start + r;
          out_cols[total_nnz] = c;
          out_vals[total_nnz] = val;
          total_nnz++;
        }
      }
    }

    fft_context_destroy(col_ctx);
    free(col_out);

    // Cleanup batch
    for (matgen_index_t b = 0; b < batch_len; b++) {
      free(row_scaled[b]);
    }

    // Progress report
    if ((batch_end % 1000) == 0 || batch_end == source->rows) {
      MATGEN_LOG_DEBUG("  Progress: %llu/%llu rows (%zu nnz)",
                       (unsigned long long)batch_end,
                       (unsigned long long)source->rows, total_nnz);
    }
  }

  free(row_scaled);
  free(col_buf);
  fft_context_destroy(row_ctx);

  MATGEN_LOG_DEBUG("FFT transform complete: %zu entries", total_nnz);

  // Create COO matrix
  matgen_coo_matrix_t* coo = matgen_coo_create(new_rows, new_cols, total_nnz);
  if (!coo) {
    free(out_rows);
    free(out_cols);
    free(out_vals);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  memcpy(coo->row_indices, out_rows, total_nnz * sizeof(matgen_index_t));
  memcpy(coo->col_indices, out_cols, total_nnz * sizeof(matgen_index_t));
  memcpy(coo->values, out_vals, total_nnz * sizeof(matgen_value_t));
  coo->nnz = total_nnz;
  coo->is_sorted = false;

  free(out_rows);
  free(out_cols);
  free(out_vals);

  // Sort and sum duplicates
  matgen_error_t err = matgen_coo_sort_with_policy(coo, MATGEN_EXEC_SEQ);
  if (err != MATGEN_SUCCESS) {
    matgen_coo_destroy(coo);
    return err;
  }

  err = matgen_coo_sum_duplicates_with_policy(coo, MATGEN_EXEC_SEQ);
  if (err != MATGEN_SUCCESS) {
    matgen_coo_destroy(coo);
    return err;
  }

  MATGEN_LOG_DEBUG("After deduplication: %zu entries", coo->nnz);

  // Convert to CSR
  *result = matgen_coo_to_csr_with_policy(coo, MATGEN_EXEC_SEQ);
  matgen_coo_destroy(coo);

  if (!*result) {
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  MATGEN_LOG_DEBUG("FFT scaling (SEQ) completed: output NNZ = %zu", (*result)->nnz);

  return MATGEN_SUCCESS;
}