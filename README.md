# MatGen: High-Performance Parallel Sparse Matrix Generation

**MatGen** is a production-ready C library designed for generating and scaling massive sparse matrices. It employs advanced parallel algorithms including **Nearest Neighbor**, **Bilinear Interpolation** and **Lanczos Resampling** to preserve structural and value integrity during scaling operations.

Architected for modern hardware, MatGen seamlessly dispatches computations across **CPUs (OpenMP)**, **GPUs (CUDA)**, and **Distributed Clusters (MPI)**, making it an ideal tool for HPC benchmarking and large-scale system testing.

---

## 🚀 Key Features

- **Advanced Scaling Algorithms**:
  - **Nearest Neighbor**: Preserves original values, ideal for discrete state matrices.
  - **Bilinear Interpolation**: Smooths values, perfect for physical simulations.
  - **Lanczos Resampling**: Preserves structure and symmetry.
- **Multi-Backend Architecture**:
  - **Sequential**: Optimized baseline for verification and small data.
  - **OpenMP**: Shared-memory parallelism for multi-core workstations.
  - **CUDA**: GPU acceleration for massive throughput.
  - **MPI**: Distributed scaling for datasets exceeding single-node memory.
- **Flexible I/O**:
  - Full **Matrix Market (.mtx)** support.
  - Efficient **CSR** (Compressed Sparse Row) and **COO** (Coordinate) internal formats.
- **Production Quality**:
  - Type-safe execution policies.
  - Comprehensive error handling.
  - Modular, extensible design.

---

## 🛠️ Quick Start

### Prerequisites

- **CMake** 3.28+
- **C Compiler** (GCC 9+, Clang 10+, MSVC 2019+)
- **Ninja** (Recommended)

### Build & Install

```bash
# Clone the repository
git clone https://github.com/erenalyoruk/matgen.git
cd matgen

# Configure and Build
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Or with presets (preferred)
#
# Current presets are:
# - windows-msvc-debug
# - windows-msvc-release
# - windows-clang-debug
# - windows-clang-release
# - linux-gcc-debug
# - linux-gcc-release
cmake --preset {preset-name}
cmake --build --preset {build-preset-name}
```

### CLI Usage (`scale_cli`)

MatGen comes with a powerful CLI tool for immediate matrix scaling.

```bash
# Scale input.mtx to 10k x 10k using Bilinear interpolation on GPU
./build/testbed/scale_cli -i input.mtx -o output.mtx \
                          -m bilinear \
                          -r 10000 -c 10000 \
                          -p par-unseq
```

---

## 📘 Documentation

- [**Integration Guide**](INTEGRATION.md): Detailed guide for developers on adding new algorithms and backends.

### C Library Usage

```c
#include "matgen/algorithms/scaling.h"
#include "matgen/io/mtx_reader.h"

// 1. Read Input
matgen_csr_matrix_t* source;
matgen_mtx_read("data/graph.mtx", &source, MATGEN_EXEC_AUTO);

// 2. Scale with Automatic Backend Selection
matgen_csr_matrix_t* result;
matgen_scale_bilinear_with_policy(MATGEN_EXEC_AUTO, source,
                                  5000, 5000, &result);

// 3. Write Output
matgen_mtx_write("data/scaled_graph.mtx", result);

// 4. Cleanup
matgen_csr_destroy(source);
matgen_csr_destroy(result);
```

---

## 📦 Supported Backends

| Backend        | Flag                    | Description                                                |
| :------------- | :---------------------- | :--------------------------------------------------------- |
| **Sequential** | `MATGEN_EXEC_SEQ`       | Single-threaded CPU execution. Always available.           |
| **OpenMP**     | `MATGEN_EXEC_PAR`       | Multi-threaded CPU execution. Requires OpenMP runtime.     |
| **CUDA**       | `MATGEN_EXEC_PAR_UNSEQ` | NVIDIA GPU acceleration. Requires CUDA Toolkit.            |
| **MPI**        | `MATGEN_EXEC_MPI`       | Distributed memory execution. Requires MPI implementation. |
| **Auto**       | `MATGEN_EXEC_AUTO`      | Runtime heuristic selection based on matrix size.          |

---

## 🤝 Contributing

We welcome contributions! Please see our [Integration Guide](INTEGRATION.md) for details on how to extend the library.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/amazing-algo`).
3.  Commit your changes (`git commit -m 'Add Amazing Algo'`).
4.  Push to the branch (`git push origin feature/amazing-algo`).
5.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
