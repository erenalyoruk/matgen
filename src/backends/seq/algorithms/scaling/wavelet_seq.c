/**
 * @file wavelet_seq.c
 * @brief Sequential implementation of wavelet-based matrix scaling
 *
 * Uses 2D Haar wavelet transform with block-based processing.
 */

#include "backends/seq/internal/wavelet_seq.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "matgen/core/matrix/coo.h"
#include "matgen/core/matrix/conversion.h"
#include "matgen/utils/log.h"

// =========================================================
// CONFIGURATION
// =========================================================
#define WAVELET_BLOCK_SIZE 4
#define WAVELET_THRESHOLD 1e-5f
#define WAVELET_MAX_SCALE 10
#define WAVELET_MAX_NEW_SIZE (WAVELET_BLOCK_SIZE * WAVELET_MAX_SCALE)

// =========================================================
// INTERNAL STRUCTURES
// =========================================================

typedef struct {
    matgen_index_t r, c;
    matgen_value_t v;
    matgen_index_t block_id;
} wavelet_element_t;

// =========================================================
// HAAR WAVELET TRANSFORMS
// =========================================================

static void haar_forward_1d(matgen_value_t* data, int n) {
    matgen_value_t temp[WAVELET_MAX_NEW_SIZE];
    int half = n / 2;
    for (int i = 0; i < half; i++) {
        matgen_value_t a = data[2 * i];
        matgen_value_t b = data[2 * i + 1];
        temp[i] = (a + b) * 0.5f;
        temp[half + i] = (a - b) * 0.5f;
    }
    for (int i = 0; i < n; i++) data[i] = temp[i];
}

static void haar_inverse_1d(matgen_value_t* data, int n) {
    matgen_value_t temp[WAVELET_MAX_NEW_SIZE];
    int half = n / 2;
    for (int i = 0; i < half; i++) {
        matgen_value_t avg = data[i];
        matgen_value_t diff = data[half + i];
        temp[2 * i] = avg + diff;
        temp[2 * i + 1] = avg - diff;
    }
    for (int i = 0; i < n; i++) data[i] = temp[i];
}

static void dwt_2d_level2(matgen_value_t* block, int size) {
    matgen_value_t col_buf[WAVELET_MAX_NEW_SIZE];
    
    // Level 1: Rows
    for (int r = 0; r < size; r++) haar_forward_1d(block + r * size, size);
    
    // Level 1: Cols
    for (int c = 0; c < size; c++) {
        for (int r = 0; r < size; r++) col_buf[r] = block[r * size + c];
        haar_forward_1d(col_buf, size);
        for (int r = 0; r < size; r++) block[r * size + c] = col_buf[r];
    }
    
    // Level 2: Top-Left quadrant (sub)
    int sub_size = size / 2;
    for (int r = 0; r < sub_size; r++) haar_forward_1d(block + r * size, sub_size);
    for (int c = 0; c < sub_size; c++) {
        for (int r = 0; r < sub_size; r++) col_buf[r] = block[r * size + c];
        haar_forward_1d(col_buf, sub_size);
        for (int r = 0; r < sub_size; r++) block[r * size + c] = col_buf[r];
    }
}

static void idwt_2d_level2(matgen_value_t* block, int size) {
    matgen_value_t col_buf[WAVELET_MAX_NEW_SIZE];
    int sub_size = size / 2;
    
    // Level 2 Inverse: Cols (sub)
    for (int c = 0; c < sub_size; c++) {
        for (int r = 0; r < sub_size; r++) col_buf[r] = block[r * size + c];
        haar_inverse_1d(col_buf, sub_size);
        for (int r = 0; r < sub_size; r++) block[r * size + c] = col_buf[r];
    }
    // Level 2 Inverse: Rows (sub)
    for (int r = 0; r < sub_size; r++) haar_inverse_1d(block + r * size, sub_size);

    // Level 1 Inverse: Cols
    for (int c = 0; c < size; c++) {
        for (int r = 0; r < size; r++) col_buf[r] = block[r * size + c];
        haar_inverse_1d(col_buf, size);
        for (int r = 0; r < size; r++) block[r * size + c] = col_buf[r];
    }
    // Level 1 Inverse: Rows
    for (int r = 0; r < size; r++) haar_inverse_1d(block + r * size, size);
}

// =========================================================
// RESIZE HELPER
// =========================================================

static matgen_value_t get_resized_value(matgen_value_t* src, int src_w, int src_h,
                                         int r, int c, float scale_r, float scale_c) {
    int src_r = (int)floorf((float)r / scale_r);
    int src_c = (int)floorf((float)c / scale_c);
    
    if (src_r >= src_h) src_r = src_h - 1;
    if (src_c >= src_w) src_c = src_w - 1;
    if (src_r < 0) src_r = 0;
    if (src_c < 0) src_c = 0;

    return src[src_r * src_w + src_c];
}

// =========================================================
// COMPARATOR FOR QSORT
// =========================================================

static int compare_elements(const void* a, const void* b) {
    const wavelet_element_t* ea = (const wavelet_element_t*)a;
    const wavelet_element_t* eb = (const wavelet_element_t*)b;
    if (ea->block_id < eb->block_id) return -1;
    if (ea->block_id > eb->block_id) return 1;
    return 0;
}

// Comparator for output elements (sort by row, then column)
typedef struct {
    matgen_index_t r, c;
    matgen_value_t v;
} output_element_t;

static int compare_output_elements(const void* a, const void* b) {
    const output_element_t* ea = (const output_element_t*)a;
    const output_element_t* eb = (const output_element_t*)b;
    if (ea->r < eb->r) return -1;
    if (ea->r > eb->r) return 1;
    if (ea->c < eb->c) return -1;
    if (ea->c > eb->c) return 1;
    return 0;
}

// =========================================================
// MAIN IMPLEMENTATION
// =========================================================

matgen_error_t matgen_scale_wavelet_seq(const matgen_csr_matrix_t* source,
                                        matgen_index_t new_rows,
                                        matgen_index_t new_cols,
                                        matgen_csr_matrix_t** result) {
    if (source == NULL || result == NULL) {
        return MATGEN_ERROR_INVALID_ARGUMENT;
    }

    // Calculate scale factor (assume uniform scaling for simplicity)
    float scale_factor = (float)new_rows / (float)source->rows;
    
    // Validate scale factor
    if (scale_factor > WAVELET_MAX_SCALE) {
        MATGEN_LOG_ERROR("Scale factor %.2f exceeds maximum %d", scale_factor, WAVELET_MAX_SCALE);
        return MATGEN_ERROR_INVALID_ARGUMENT;
    }

    MATGEN_LOG_INFO("Wavelet scaling: %llux%llu -> %llux%llu (scale=%.2f)",
                    (unsigned long long)source->rows, (unsigned long long)source->cols,
                    (unsigned long long)new_rows, (unsigned long long)new_cols,
                    scale_factor);

    // Convert CSR to element list
    matgen_size_t nnz = source->nnz;
    wavelet_element_t* elements = (wavelet_element_t*)malloc(nnz * sizeof(wavelet_element_t));
    if (!elements) {
        return MATGEN_ERROR_OUT_OF_MEMORY;
    }

    matgen_index_t grid_width = (source->cols + WAVELET_BLOCK_SIZE - 1) / WAVELET_BLOCK_SIZE;

    // Extract elements from CSR and compute block IDs
    matgen_size_t idx = 0;
    for (matgen_index_t row = 0; row < source->rows; row++) {
        matgen_size_t row_start = source->row_ptr[row];
        matgen_size_t row_end = source->row_ptr[row + 1];
        
        for (matgen_size_t j = row_start; j < row_end; j++) {
            elements[idx].r = row;
            elements[idx].c = source->col_indices[j];
            elements[idx].v = source->values[j];
            
            matgen_index_t br = row / WAVELET_BLOCK_SIZE;
            matgen_index_t bc = source->col_indices[j] / WAVELET_BLOCK_SIZE;
            elements[idx].block_id = br * grid_width + bc;
            idx++;
        }
    }

    // Sort by block ID
    qsort(elements, nnz, sizeof(wavelet_element_t), compare_elements);

    // Allocate output COO arrays (estimate 4x growth initially, will realloc if needed)
    matgen_size_t out_capacity = nnz * 4;
    matgen_size_t out_count = 0;
    
    matgen_index_t* out_rows = (matgen_index_t*)malloc(out_capacity * sizeof(matgen_index_t));
    matgen_index_t* out_cols = (matgen_index_t*)malloc(out_capacity * sizeof(matgen_index_t));
    matgen_value_t* out_vals = (matgen_value_t*)malloc(out_capacity * sizeof(matgen_value_t));
    
    if (!out_rows || !out_cols || !out_vals) {
        free(elements);
        free(out_rows);
        free(out_cols);
        free(out_vals);
        return MATGEN_ERROR_OUT_OF_MEMORY;
    }

    // Calculate new block size
    int new_dim = (int)(WAVELET_BLOCK_SIZE * scale_factor);
    if (new_dim % 2 != 0) new_dim += 1;  // Ensure even size for Haar
    
    // Process blocks
    matgen_size_t i = 0;
    while (i < nnz) {
        matgen_index_t current_block_id = elements[i].block_id;
        
        // A. Load Block (Dense)
        matgen_value_t tile[WAVELET_BLOCK_SIZE * WAVELET_BLOCK_SIZE];
        memset(tile, 0, sizeof(tile));
        
        while (i < nnz && elements[i].block_id == current_block_id) {
            int r = elements[i].r % WAVELET_BLOCK_SIZE;
            int c = elements[i].c % WAVELET_BLOCK_SIZE;
            tile[r * WAVELET_BLOCK_SIZE + c] = elements[i].v;
            i++;
        }

        // B. Wavelet Transform
        dwt_2d_level2(tile, WAVELET_BLOCK_SIZE);

        // C. Resize in wavelet domain
        matgen_value_t new_tile[WAVELET_MAX_NEW_SIZE * WAVELET_MAX_NEW_SIZE];
        for (int r = 0; r < new_dim; r++) {
            for (int c = 0; c < new_dim; c++) {
                new_tile[r * new_dim + c] = get_resized_value(tile, WAVELET_BLOCK_SIZE, 
                                                               WAVELET_BLOCK_SIZE, r, c, 
                                                               scale_factor, scale_factor);
            }
        }

        // D. Inverse Transform
        idwt_2d_level2(new_tile, new_dim);

        // E. Write Back (Scatter)
        matgen_index_t block_r = current_block_id / grid_width;
        matgen_index_t block_c = current_block_id % grid_width;
        
        matgen_index_t new_start_r = (matgen_index_t)(block_r * WAVELET_BLOCK_SIZE * scale_factor);
        matgen_index_t new_start_c = (matgen_index_t)(block_c * WAVELET_BLOCK_SIZE * scale_factor);

        for (int r = 0; r < new_dim; r++) {
            for (int c = 0; c < new_dim; c++) {
                matgen_value_t val = new_tile[r * new_dim + c];
                matgen_index_t final_r = new_start_r + r;
                matgen_index_t final_c = new_start_c + c;

                if (fabsf(val) > WAVELET_THRESHOLD && final_r < new_rows && final_c < new_cols) {
                    // Check capacity
                    if (out_count >= out_capacity) {
                        out_capacity *= 2;
                        out_rows = (matgen_index_t*)realloc(out_rows, out_capacity * sizeof(matgen_index_t));
                        out_cols = (matgen_index_t*)realloc(out_cols, out_capacity * sizeof(matgen_index_t));
                        out_vals = (matgen_value_t*)realloc(out_vals, out_capacity * sizeof(matgen_value_t));
                        if (!out_rows || !out_cols || !out_vals) {
                            free(elements);
                            return MATGEN_ERROR_OUT_OF_MEMORY;
                        }
                    }
                    
                    out_rows[out_count] = final_r;
                    out_cols[out_count] = final_c;
                    out_vals[out_count] = val;
                    out_count++;
                }
            }
        }
    }

    free(elements);

    MATGEN_LOG_INFO("Wavelet scaling complete: output NNZ = %llu", (unsigned long long)out_count);

    // Sort output by (row, col) for proper COO order
    output_element_t* out_elements = (output_element_t*)malloc(out_count * sizeof(output_element_t));
    if (!out_elements) {
        free(out_rows);
        free(out_cols);
        free(out_vals);
        return MATGEN_ERROR_OUT_OF_MEMORY;
    }
    
    for (matgen_size_t k = 0; k < out_count; k++) {
        out_elements[k].r = out_rows[k];
        out_elements[k].c = out_cols[k];
        out_elements[k].v = out_vals[k];
    }
    
    qsort(out_elements, out_count, sizeof(output_element_t), compare_output_elements);
    
    for (matgen_size_t k = 0; k < out_count; k++) {
        out_rows[k] = out_elements[k].r;
        out_cols[k] = out_elements[k].c;
        out_vals[k] = out_elements[k].v;
    }
    
    free(out_elements);

    // Create COO matrix from output
    matgen_coo_matrix_t* coo = matgen_coo_create(new_rows, new_cols, out_count);
    if (!coo) {
        free(out_rows);
        free(out_cols);
        free(out_vals);
        return MATGEN_ERROR_OUT_OF_MEMORY;
    }

    // Copy data to COO
    memcpy(coo->row_indices, out_rows, out_count * sizeof(matgen_index_t));
    memcpy(coo->col_indices, out_cols, out_count * sizeof(matgen_index_t));
    memcpy(coo->values, out_vals, out_count * sizeof(matgen_value_t));
    coo->nnz = out_count;

    free(out_rows);
    free(out_cols);
    free(out_vals);

    // Convert COO to CSR
    *result = matgen_coo_to_csr(coo);
    matgen_coo_destroy(coo);

    if (*result == NULL) {
        return MATGEN_ERROR_OUT_OF_MEMORY;
    }

    return MATGEN_SUCCESS;
}
