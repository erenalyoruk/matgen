# MatGen Integration & Extension Guide

This document is intended for developers who wish to extend MatGen with new scaling algorithms, execution backends, or simply understand the internal architecture better.

## 🏗️ Architecture Overview

MatGen follows a strict **Dispatch-Implementation** pattern to separate the public API from backend-specific logic.

```
matgen/
├── include/matgen/
│   ├── algorithms/       # Public API for algorithms (e.g., scaling.h)
│   └── core/
│       └── execution/    # Execution policies (policy.h)
└── src/
    ├── algorithms/       # Dispatch logic (decides which backend to call)
    └── backends/         # Concrete implementations
        ├── seq/          # Sequential code
        ├── omp/          # OpenMP code
        ├── cuda/         # CUDA kernels
        └── mpi/          # MPI logic
```

---

## ➕ Adding a New Scaling Algorithm

To add a new algorithm (e.g., `Bicubic Interpolation`), follow these steps:

### 1. Define Public API

Add the function prototype to `include/matgen/algorithms/scaling.h`. Use `matgen_exec_policy_t` to allow backend selection.

```c
// include/matgen/algorithms/scaling.h
matgen_error_t matgen_scale_bicubic_with_policy(
    matgen_exec_policy_t policy,
    const matgen_csr_matrix_t* source,
    matgen_index_t new_rows,
    matgen_index_t new_cols,
    matgen_csr_matrix_t** result);
```

### 2. Implement Dispatch Logic

Create a dispatch file in `src/algorithms/scaling/`. This function acts as a router.

```c
// src/algorithms/scaling/bicubic_dispatch.c
#include "matgen/algorithms/scaling.h"
#include "matgen/core/execution/dispatch.h"

matgen_error_t matgen_scale_bicubic_with_policy(...) {
    // 1. Resolve Policy (Auto -> Specific)
    matgen_exec_policy_t resolved = matgen_exec_resolve(policy);

    // 2. Create Context
    matgen_dispatch_context_t ctx = matgen_dispatch_create(resolved);

    // 3. Dispatch Macro
    MATGEN_DISPATCH_BEGIN(ctx, "bicubic_scale") {
        MATGEN_DISPATCH_CASE_SEQ:
            return matgen_scale_bicubic_seq(...); // You must implement this!

        #ifdef MATGEN_HAS_OPENMP
        MATGEN_DISPATCH_CASE_PAR:
            return matgen_scale_bicubic_omp(...);
        #endif

        // ... other backends

        MATGEN_DISPATCH_DEFAULT:
            return MATGEN_ERROR_UNSUPPORTED;
    }
    MATGEN_DISPATCH_END();
}
```

### 3. Implement Sequential Backend (Required)

Always implement the sequential version first in `src/backends/seq/algorithms/scaling/`. This serves as the reference implementation and fallback. Also, you must define a header for backend functions (e.g. `matgen_scale_bicubic_{backend}`) in `src/backends/{backend}/internal/bicubic_{backend}.h` for convenient include in dispatch.

### 4. Add New Files to the Corresponding CMakeLists.txt
Add the new files to the corresponding CMakeLists.txt file's target sources for the linker to successfully link your algorithms to the program:
- Add dispatch file to `src/algorithms/scaling/CMakeLists.txt`
- Add back-end sequential version of your algorithm to `src/backends/seq/algorithms/scaling/CMakeLists.txt`
- Add back-end parallel version of your algorithm to `src/backends/{backend}/algorithms/scaling/CMakeLists.txt` where backend is the parallelization framework (OPENMP/MPI/CUDA)

### 5. Add Options in Executable

`testbed/scale_cli.c` is a basic CLI program that has options to test backends and algorithms in runtime. You must add your public definition of algorithm (`include/algorithms/scaling.h` -> `matgen_scale_bicubic_with_policy`) in CLI to run and test.

---

## 🧩 Adding a New Matrix Format (OPTIONAL)

### THIS STEP IS TOTALLY OPTIONAL BECAUSE OF YOUR INTERNAL CODE STAYS INTERNAL AND YOU CAN JUST CREATE CONVERSION FUNCTION TO FINAL FORM TO WRITE MATRIX MARKET FILE.

Currently, MatGen relies on **CSR** for processing and **COO** for I/O. Adding a new format requires a few steps.

### 1. Define the Data Structure

Create a new header in `include/matgen/core/matrix/`.

```c
// include/matgen/core/matrix/ellpack.h
typedef struct {
    matgen_index_t rows;
    matgen_index_t cols;
    matgen_index_t max_row_nnz;
    matgen_index_t* col_indices; // Dimensions: rows * max_row_nnz
    matgen_value_t* values;      // Dimensions: rows * max_row_nnz
} matgen_ell_matrix_t;
```

### 2. Integration Strategy

You have two options to make this usable:

- **Option A: The Adapter Path (Recommended)**
  - Implement conversion functions to/from CSR.
  - **Benefit**: You can immediately use all existing scaling algorithms (Nearest Neighbor, Bilinear) by converting, scaling, and converting back.
  - **Drawback**: Conversion overhead.
- **Option B: Native Implementation**
  - Re-implement the scaling algorithms specifically for your format.
  - **Benefit**: Maximum performance.
  - **Drawback**: Significant development effort.

### 3. Implementing Converters

If you choose Option A, implement:

```c
matgen_error_t matgen_ell_from_csr(const matgen_csr_matrix_t* src, matgen_ell_matrix_t** dst);
matgen_error_t matgen_ell_to_csr(const matgen_ell_matrix_t* src, matgen_csr_matrix_t** dst);
```

### 4. Making it Parallel or CUDA-Accelerated

To add high-performance support for your new matrix type (e.g., fast creation or conversion):

#### Parallel (OpenMP)

Implement the conversion logic using OpenMP parallel loops.

- **Location**: `src/backends/omp/core/matrix/`
- **Strategy**: Parallelize the loop over rows.

```c
#pragma omp parallel for
for (matgen_index_t i = 0; i < rows; ++i) {
    // Process row i independenty
}
```

#### CUDA (GPU)

1.  **Define a Device Struct**: If the matrix lives on GPU, the pointers in your struct must point to device memory.
2.  **Location**: `src/backends/cuda/core/matrix/`
3.  **Implementation**:
    - Allocate memory using `cudaMalloc`.
    - Write a `__global__` kernel to perform the conversion/construction.
    - One thread per row (or one thread per element) strategy usually works best.

```cpp
// src/backends/cuda/core/matrix/ell_cuda.cu
__global__ void csr_to_ell_kernel(...) {
    matgen_index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        // ...
    }
}
```

---

## 💡 Tips for Types & Precision

MatGen is designed to handle massive matrices, often exceeding the limits of standard 32-bit integers.

### 1. Indexing (`matgen_index_t`)

- **What is it?** Typically `uint64_t` (unsigned long long).
- **When to use it?** For anything related to **rows**, **columns**, or **indices** in the matrix.
- **Why?** Matrix dimensions can easily exceed 4 billion (2^32). Using `int` will cause overflows and segfaults on large datasets.

### 2. Counts (`matgen_size_t`)

- **What is it?** Typically `size_t` or `uint64_t`.
- **When to use it?** For the number of non-zeros (`nnz`), array lengths, or loop counters iterating over data arrays.
- **Why?** `nnz` is often much larger than rows/cols.

### 3. Values (`matgen_value_t`)

- **What is it?** Typically `double` (f64) or `float` (f32).
- **When to use it?** For the actual data values inside the matrix.
- **Tip**: Avoid hardcoding `float` or `double`. Use `matgen_value_t` so precision can be switched project-wide via `types.h`.

### 4. Error Handling

- **Always** return `matgen_error_t`.
- **Never** use `exit()` inside the library.
- Propagate errors up the stack. If a backend fails (e.g., CUDA out of memory), the dispatch logic (if smart enough) could theoretically fallback, but currently, it returns the error.
