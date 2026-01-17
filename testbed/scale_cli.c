/**
 * @file scale_cli.c
 * @brief CLI tool for sparse matrix scaling with execution policy support
 *
 * Usage:
 *   scale_cli -i <input.mtx> -o <output.mtx> -m <method> -r <rows> -c <cols>
 * [options]
 *
 * Methods:
 *   nearest      - Nearest neighbor interpolation
 *   bilinear     - Bilinear interpolation
 *   lanczos      - Lanczos interpolation (structure-preserving)
 *
 * Execution Policies:
 *   seq          - Sequential (single-threaded)
 *   par          - Parallel (OpenMP)
 *   par-unseq    - Parallel unsequenced (CUDA)
 *   mpi          - Distributed (MPI)
 *   auto         - Automatic selection based on problem size
 *
 * Examples:
 *   # Scale to 100x100 using bilinear interpolation (auto policy)
 *   scale_cli -i input.mtx -o output.mtx -m bilinear -r 100 -c 100
 *
 *   # Scale to 100x100 using bilinear with explicit OpenMP
 *   scale_cli -i input.mtx -o output.mtx -m bilinear -r 100 -c 100 -p par
 *
 *   # Scale to 200x200 using nearest neighbor with averaging on collision
 *   scale_cli -i input.mtx -o output.mtx -m nearest -r 200 -c 200 --collision
 * avg
 *
 *   # Scale with CUDA backend and 8 OpenMP threads for fallback
 *   scale_cli -i input.mtx -o output.mtx -m bilinear -r 1000 -c 1000 -p
 * par-unseq -t 8
 *
 *   # Scale using MPI (run with mpirun -np 4 scale_cli ...)
 *   mpirun -np 4 scale_cli -i input.mtx -o output.mtx -m bilinear -r 1000 -c
 * 1000 -p mpi
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matgen/algorithms/scaling.h"
#include "matgen/core/execution/policy.h"
#include "matgen/core/matrix/conversion.h"
#include "matgen/core/matrix/coo.h"
#include "matgen/core/matrix/csr.h"
#include "matgen/io/mtx_reader.h"
#include "matgen/io/mtx_writer.h"
#include "matgen/utils/log.h"

#ifdef MATGEN_HAS_OPENMP
#include <omp.h>
#endif

#ifdef MATGEN_HAS_MPI
#include <mpi.h>
#endif

#ifndef _WIN32
#include <sys/time.h>
#else
#include <windows.h>
#endif

// =============================================================================
// Configuration Structure
// =============================================================================

typedef struct {
  const char* input_file;
  const char* output_file;
  const char* method;      // "nearest" or "bilinear"
  const char* policy_str;  // "seq", "par", "par-unseq", "mpi", "auto"
  matgen_index_t new_rows;
  matgen_index_t new_cols;
  matgen_collision_policy_t collision_policy;
  int num_threads;  // For OpenMP (0 = auto)
  int cuda_device;  // For CUDA (-1 = default)
  bool verbose;
  bool show_stats;
  bool quiet;
  bool show_backend_info;
} cli_config_t;

// =============================================================================
// MPI Helper Macros
// =============================================================================

#ifdef MATGEN_HAS_MPI
#define MPI_RANK_0_ONLY(code)                  \
  do {                                         \
    int _mpi_rank;                             \
    MPI_Comm_rank(MPI_COMM_WORLD, &_mpi_rank); \
    if (_mpi_rank == 0) {                      \
      code                                     \
    }                                          \
  } while (0)

#define IS_MPI_RANK_0()                        \
  ({                                           \
    int _mpi_rank;                             \
    MPI_Comm_rank(MPI_COMM_WORLD, &_mpi_rank); \
    (_mpi_rank == 0);                          \
  })
#else
#define MPI_RANK_0_ONLY(code) \
  do {                        \
    code                      \
  } while (0)
#define IS_MPI_RANK_0() (1)
#endif

// =============================================================================
// Helper Functions
// =============================================================================

static double get_wall_time(void) {
#ifdef _WIN32
  LARGE_INTEGER frequency;
  LARGE_INTEGER counter;
  QueryPerformanceFrequency(&frequency);
  QueryPerformanceCounter(&counter);
  return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
#endif
}

/**
 * @brief Get MPI rank (0 if MPI not enabled)
 */
static int get_mpi_rank(void) {
#ifdef MATGEN_HAS_MPI
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
#else
  return 0;
#endif
}

/**
 * @brief Get MPI size (1 if MPI not enabled)
 */
static int get_mpi_size(void) {
#ifdef MATGEN_HAS_MPI
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size;
#else
  return 1;
#endif
}

/**
 * @brief Print usage information (rank 0 only)
 */
static void print_usage(const char* prog_name) {
  if (get_mpi_rank() != 0) {
    return;
  }

  printf("Usage: %s [options]\n", prog_name);
  printf("\n");
  printf("Required arguments:\n");
  printf("  -i, --input <file>     Input matrix file (.mtx format)\n");
  printf("  -o, --output <file>    Output matrix file (.mtx format)\n");
  printf("  -m, --method <method>  Scaling method:\n");
  printf("                           'nearest'  - Nearest neighbor\n");
  printf("                           'bilinear' - Bilinear interpolation\n"
         "                           'adaptive' - Adaptive scaling\n");
  printf("                           'bilinear' - Bilinear interpolation\n");
  printf("                           'lanczos'  - Lanczos interpolation\n");
  printf("                           'fft'      - FFT-based interpolation\n");
  printf("                           'wavelet'  - Wavelet-based interpolation\n");
  printf("  -r, --rows <N>         Target number of rows\n");
  printf("  -c, --cols <N>         Target number of columns\n");
  printf("\n");
  printf("Optional arguments:\n");
  printf("  -p, --policy <policy>  Execution policy:\n");
  printf(
      "                           'seq'        - Sequential "
      "(single-threaded)\n");
#ifdef MATGEN_HAS_OPENMP
  printf("                           'par'        - Parallel (OpenMP)\n");
#endif
#ifdef MATGEN_HAS_CUDA
  printf(
      "                           'par-unseq'  - Parallel unsequenced "
      "(CUDA)\n");
#endif
#ifdef MATGEN_HAS_MPI
  printf("                           'mpi'        - Distributed (MPI)\n");
#endif
  printf(
      "                           'auto'       - Automatic selection "
      "(default)\n");
  printf("  --collision <policy>   Collision policy for nearest neighbor:\n");
  printf(
      "                           'sum' (default), 'avg', 'max', 'min', "
      "'last'\n");
#ifdef MATGEN_HAS_OPENMP
  printf("  -t, --threads <N>      Number of OpenMP threads (0 = auto)\n");
#endif
#ifdef MATGEN_HAS_CUDA
  printf("  -d, --device <N>       CUDA device ID (-1 = default)\n");
#endif
  printf("  -v, --verbose          Enable verbose output\n");
  printf("  -s, --stats            Show detailed statistics\n");
  printf("  -q, --quiet            Suppress all non-error output\n");
  printf("  -b, --backend-info     Show available backends\n");
  printf("  -h, --help             Show this help message\n");
  printf("\n");
  printf("Examples:\n");
  printf("  # Auto policy selection\n");
  printf("  %s -i input.mtx -o output.mtx -m bilinear -r 100 -c 100\n",
         prog_name);
  printf("\n");
  printf("  # Explicit sequential\n");
  printf("  %s -i input.mtx -o output.mtx -m bilinear -r 100 -c 100 -p seq\n",
         prog_name);
  printf("\n");
#ifdef MATGEN_HAS_OPENMP
  printf("  # Parallel with 8 threads\n");
  printf(
      "  %s -i input.mtx -o output.mtx -m bilinear -r 100 -c 100 -p par -t 8\n",
      prog_name);
  printf("\n");
#endif
#ifdef MATGEN_HAS_MPI
  printf("  # MPI with 4 processes\n");
  printf(
      "  mpirun -np 4 %s -i input.mtx -o output.mtx -m bilinear -r 100 -c 100 "
      "-p mpi\n",
      prog_name);
  printf("\n");
#endif
}

/**
 * @brief Print available backends (rank 0 only)
 */
static void print_backend_info(void) {
  if (get_mpi_rank() != 0) {
    return;
  }

  printf("\n");
  printf("========================================\n");
  printf("Available Execution Backends\n");
  printf("========================================\n");
  printf("Sequential:   %s\n",
         matgen_exec_is_available(MATGEN_EXEC_SEQ) ? "YES" : "NO");
  printf("OpenMP:       %s",
         matgen_exec_is_available(MATGEN_EXEC_PAR) ? "YES" : "NO");
#ifdef MATGEN_HAS_OPENMP
  printf(" (max threads: %d)", matgen_exec_get_num_threads());
#endif
  printf("\n");

  printf("CUDA:         %s",
         matgen_exec_is_available(MATGEN_EXEC_PAR_UNSEQ) ? "YES" : "NO");
#ifdef MATGEN_HAS_CUDA
  printf(" (devices: %d)", matgen_exec_get_num_cuda_devices());
#endif
  printf("\n");

  printf("MPI:          %s",
         matgen_exec_is_available(MATGEN_EXEC_MPI) ? "YES" : "NO");
#ifdef MATGEN_HAS_MPI
  printf(" (size: %d, rank: %d)", get_mpi_size(), get_mpi_rank());
#endif
  printf("\n");
  printf("========================================\n\n");
}

/**
 * @brief Parse execution policy string
 */
static matgen_exec_policy_t parse_policy(const char* policy_str) {
  if (!policy_str || strcmp(policy_str, "auto") == 0) {
    return MATGEN_EXEC_AUTO;
  }

  if (strcmp(policy_str, "seq") == 0) {
    return MATGEN_EXEC_SEQ;
  }

  if (strcmp(policy_str, "par") == 0) {
    return MATGEN_EXEC_PAR;
  }

  if (strcmp(policy_str, "par-unseq") == 0 || strcmp(policy_str, "cuda") == 0) {
    return MATGEN_EXEC_PAR_UNSEQ;
  }

  if (strcmp(policy_str, "mpi") == 0) {
    return MATGEN_EXEC_MPI;
  }

  return MATGEN_EXEC_AUTO;
}

/**
 * @brief Parse command line arguments
 */
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static bool parse_args(int argc, char** argv, cli_config_t* config) {
  // Initialize defaults
  config->input_file = NULL;
  config->output_file = NULL;
  config->method = NULL;
  config->policy_str = "auto";
  config->new_rows = 0;
  config->new_cols = 0;
  config->collision_policy = MATGEN_COLLISION_SUM;
  config->num_threads = 0;   // Auto
  config->cuda_device = -1;  // Default
  config->verbose = false;
  config->show_stats = false;
  config->quiet = false;
  config->show_backend_info = false;
  
  int rank = get_mpi_rank();

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      return false;
    }

    if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--backend-info") == 0) {
      config->show_backend_info = true;
    } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {
      if (++i >= argc) {
        if (rank == 0) {
          fprintf(stderr, "Error: %s requires an argument\n", argv[i - 1]);
        }
        return false;
      }
      config->input_file = argv[i];
    } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
      if (++i >= argc) {
        if (rank == 0) {
          fprintf(stderr, "Error: %s requires an argument\n", argv[i - 1]);
        }
        return false;
      }
      config->output_file = argv[i];
    } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--method") == 0) {
      if (++i >= argc) {
        if (rank == 0) {
          fprintf(stderr, "Error: %s requires an argument\n", argv[i - 1]);
        }
        return false;
      }
      config->method = argv[i];
    } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--policy") == 0) {
      if (++i >= argc) {
        if (rank == 0) {
          fprintf(stderr, "Error: %s requires an argument\n", argv[i - 1]);
        }
        return false;
      }
      config->policy_str = argv[i];
    } else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--rows") == 0) {
      if (++i >= argc) {
        if (rank == 0) {
          fprintf(stderr, "Error: %s requires an argument\n", argv[i - 1]);
        }
        return false;
      }
      config->new_rows = (matgen_index_t)atoll(argv[i]);
    } else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--cols") == 0) {
      if (++i >= argc) {
        if (rank == 0) {
          fprintf(stderr, "Error: %s requires an argument\n", argv[i - 1]);
        }
        return false;
      }
      config->new_cols = (matgen_index_t)atoll(argv[i]);
    } else if (strcmp(argv[i], "-t") == 0 ||
               strcmp(argv[i], "--threads") == 0) {
      if (++i >= argc) {
        if (rank == 0) {
          fprintf(stderr, "Error: %s requires an argument\n", argv[i - 1]);
        }
        return false;
      }
      config->num_threads = atoi(argv[i]);
      if (config->num_threads < 0) {
        if (rank == 0) {
          fprintf(stderr, "Error: Number of threads must be >= 0\n");
        }
        return false;
      }
    } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--device") == 0) {
      if (++i >= argc) {
        if (rank == 0) {
          fprintf(stderr, "Error: %s requires an argument\n", argv[i - 1]);
        }
        return false;
      }
      config->cuda_device = atoi(argv[i]);
    } else if (strcmp(argv[i], "--collision") == 0) {
      if (++i >= argc) {
        if (rank == 0) {
          fprintf(stderr, "Error: --collision requires an argument\n");
        }
        return false;
      }
      if (strcmp(argv[i], "sum") == 0) {
        config->collision_policy = MATGEN_COLLISION_SUM;
      } else if (strcmp(argv[i], "avg") == 0) {
        config->collision_policy = MATGEN_COLLISION_AVG;
      } else if (strcmp(argv[i], "max") == 0) {
        config->collision_policy = MATGEN_COLLISION_MAX;
      } else if (strcmp(argv[i], "min") == 0) {
        config->collision_policy = MATGEN_COLLISION_MIN;
      } else if (strcmp(argv[i], "last") == 0) {
        config->collision_policy = MATGEN_COLLISION_LAST;
      } else {
        if (rank == 0) {
          fprintf(stderr, "Error: Unknown collision policy '%s'\n", argv[i]);
        }
        return false;
      }
    } 
    else if (strcmp(argv[i], "-v") == 0 ||
               strcmp(argv[i], "--verbose") == 0) {
      config->verbose = true;
    } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--stats") == 0) {
      config->show_stats = true;
    } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quiet") == 0) {
      config->quiet = true;
    } else {
      if (rank == 0) {
        fprintf(stderr, "Error: Unknown argument '%s'\n", argv[i]);
      }
      return false;
    }
  }

  // Show backend info and exit if requested
  if (config->show_backend_info) {
    return true;  // Will be handled specially in main
  }

  // Validate required arguments (errors only printed on rank 0)
  if (!config->input_file) {
    if (rank == 0) {
      fprintf(stderr, "Error: Input file (-i) is required\n");
    }
    return false;
  }
  if (!config->output_file) {
    if (rank == 0) {
      fprintf(stderr, "Error: Output file (-o) is required\n");
    }
    return false;
  }
  if (!config->method) {
    if (rank == 0) {
      fprintf(stderr, "Error: Method (-m) is required\n");
    }
    return false;
  }
  if (config->new_rows == 0) {
    if (rank == 0) {
      fprintf(stderr, "Error: Target rows (-r) is required and must be > 0\n");
    }
    return false;
  }
  if (config->new_cols == 0) {
    if (rank == 0) {
      fprintf(stderr, "Error: Target cols (-c) is required and must be > 0\n");
    }
    return false;
  }

  // Validate method
  if (strcmp(config->method, "nearest") != 0 &&
      strcmp(config->method, "bilinear") != 0 && 
      strcmp(config->method, "adaptive") != 0 &&
      strcmp(config->method, "lanczos") != &&
      strcmp(config->method, "fft") != 0 &&
      strcmp(config->method, "wavelet") != 0) ) {
    if (rank == 0) {
      fprintf(stderr, "Error: Invalid method '%s'\n", config->method);
      fprintf(stderr, "Valid methods: 'nearest', 'bilinear', 'adaptive','lanczos', 'fft', 'wavelet'\n");
    if (rank == 0) {
      fprintf(stderr, "Error: Invalid method '%s'\n", config->method);
      fprintf(stderr, "Valid methods: 'nearest', 'bilinear', 'lanczos', 'fft', 'wavelet'\n");
    }
    return false; 
  }
  
  // Lanczos requires square output
  if (strcmp(config->method, "lanczos") == 0 && config->new_rows != config->new_cols) {
    if (rank == 0) {
      fprintf(stderr, "Error: Lanczos scaling requires square output (rows == cols)\n");
    }
    return false;
  }

  // FFT has some limitations
  if (strcmp(config->method, "fft") == 0) {
    if (!matgen_exec_is_available(MATGEN_EXEC_SEQ) && !matgen_exec_is_available(MATGEN_EXEC_PAR_UNSEQ)) {
      if (rank == 0) {
        fprintf(stderr, "Error: FFT scaling requires FFTW3 (sequential) or cuFFT (CUDA) support\n");
      }
      return false;
    }
  }

  return true;
}

/**
 * @brief Compute and print matrix statistics (rank 0 only)
 */
static void print_matrix_stats(const char* label,
                               const matgen_csr_matrix_t* matrix) {
  if (get_mpi_rank() != 0) {
    return;
  }

  if (!matrix) {
    printf("%s: NULL matrix\n", label);
    return;
  }

  // Compute statistics
  matgen_value_t sum = (matgen_value_t)0.0;
  matgen_value_t sum_sq = (matgen_value_t)0.0;
  matgen_value_t min_val = (matgen_value_t)0.0;
  matgen_value_t max_val = (matgen_value_t)0.0;
  bool first = true;

  for (matgen_index_t i = 0; i < matrix->rows; i++) {
    matgen_size_t row_start = matrix->row_ptr[i];
    matgen_size_t row_end = matrix->row_ptr[i + 1];

    for (matgen_size_t idx = row_start; idx < row_end; idx++) {
      matgen_value_t val = matrix->values[idx];
      sum += val;
      sum_sq += val * val;

      if (first) {
        min_val = val;
        max_val = val;
        first = false;
      } else {
        if (val < min_val) {
          min_val = val;
        }

        if (val > max_val) {
          max_val = val;
        }
      }
    }
  }

  matgen_value_t mean =
      matrix->nnz > 0 ? sum / (matgen_value_t)matrix->nnz : (matgen_value_t)0.0;
  matgen_value_t variance =
      matrix->nnz > 0 ? (sum_sq / (matgen_value_t)matrix->nnz) - (mean * mean)
                      : (matgen_value_t)0.0;
  matgen_value_t std_dev = (matgen_value_t)sqrt(variance);
  matgen_value_t frobenius = (matgen_value_t)sqrt(sum_sq);
  matgen_value_t density = (matgen_value_t)matrix->nnz /
                           (matgen_value_t)(matrix->rows * matrix->cols);
  matgen_value_t sparsity = (matgen_value_t)1.0 - density;

  printf("\n%s:\n", label);
  printf("  Dimensions:      %llu × %llu\n", (unsigned long long)matrix->rows,
         (unsigned long long)matrix->cols);
  printf("  Non-zeros (NNZ): %llu\n", (unsigned long long)matrix->nnz);
  printf("  Density:         %.6f (%.2f%%)\n", density, density * 100.0);
  printf("  Sparsity:        %.6f (%.2f%%)\n", sparsity, sparsity * 100.0);

  if (matrix->nnz > 0) {
    printf("  Value range:     [%.6e, %.6e]\n", min_val, max_val);
    printf("  Sum:             %.6e\n", sum);
    printf("  Mean:            %.6e\n", mean);
    printf("  Std Dev:         %.6e\n", std_dev);
    printf("  Frobenius norm:  %.6e\n", frobenius);
  }
}

/**
 * @brief Get collision policy name as string
 */
static const char* collision_policy_name(matgen_collision_policy_t policy) {
  switch (policy) {
    case MATGEN_COLLISION_SUM:
      return "sum";
    case MATGEN_COLLISION_AVG:
      return "average";
    case MATGEN_COLLISION_MAX:
      return "max";
    case MATGEN_COLLISION_MIN:
      return "min";
    case MATGEN_COLLISION_LAST:
      return "last";
    default:
      return "unknown";
  }
}

#ifdef MATGEN_HAS_MPI
/**
 * @brief Broadcast error status from rank 0 to all ranks
 */
static int broadcast_error(int error_code) {
  MPI_Bcast(&error_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return error_code;
}

/**
 * @brief Scatter CSR matrix from rank 0 to all ranks
 *
 * Forward declaration for the internal MPI function
 */
extern matgen_csr_matrix_t* matgen_csr_scatter(
    const matgen_csr_matrix_t* global_matrix);

/**
 * @brief Gather distributed CSR matrix to rank 0
 *
 * Forward declaration for the internal MPI function
 */
extern matgen_csr_matrix_t* matgen_csr_gather(
    const matgen_csr_matrix_t* local_matrix, matgen_index_t global_rows,
    matgen_index_t global_cols);
#endif

// =============================================================================
// Main Function
// =============================================================================

// NOLINTNEXTLINE
int main(int argc, char** argv) {
  int exit_code = 0;

#ifdef MATGEN_HAS_MPI
  // Initialize MPI
  MPI_Init(&argc, &argv);

  int rank = get_mpi_rank();
  int size = get_mpi_size();

  // Safety check: if running with MPI (size > 1), force MPI policy
  // This prevents crashes when users run with mpiexec but forget -p mpi
  bool force_mpi_policy = (size > 1);
#else
  int rank = 0;
  int size = 1;
  bool force_mpi_policy = false;
#endif

  cli_config_t config;

  // Parse arguments (all ranks parse, but only rank 0 prints errors)
  if (!parse_args(argc, argv, &config)) {
    print_usage(argv[0]);
#ifdef MATGEN_HAS_MPI
    MPI_Finalize();
#endif
    return 1;
  }

  // Handle backend info request
  if (config.show_backend_info) {
    print_backend_info();
#ifdef MATGEN_HAS_MPI
    MPI_Finalize();
#endif
    return 0;
  }

  // Configure logging (suppress on non-root ranks unless verbose)
  if (config.quiet || rank != 0) {
    matgen_log_set_level(MATGEN_LOG_LEVEL_ERROR);
  } else if (config.verbose) {
    matgen_log_set_level(MATGEN_LOG_LEVEL_DEBUG);
  } else {
    matgen_log_set_level(MATGEN_LOG_LEVEL_INFO);
  }

  // Parse and resolve execution policy
  matgen_exec_policy_t policy = parse_policy(config.policy_str);

  // Force MPI policy if running under mpiexec with multiple processes
  if (force_mpi_policy && policy != MATGEN_EXEC_MPI) {
    if (rank == 0 && !config.quiet) {
      printf("Warning: Running with %d MPI processes, forcing MPI policy\n",
             size);
    }
    policy = MATGEN_EXEC_MPI;
    config.policy_str = "mpi";
  }

  // For AUTO policy, select based on problem size
  if (policy == MATGEN_EXEC_AUTO && !config.quiet && rank == 0) {
    printf("Auto policy: selecting based on problem size...\n");
  }

  matgen_exec_policy_t resolved_policy = matgen_exec_resolve(policy);

  // Set OpenMP threads if specified
#ifdef MATGEN_HAS_OPENMP
  if (config.num_threads > 0) {
    omp_set_num_threads(config.num_threads);
  }
#endif

  if (!config.quiet && rank == 0) {
    printf(
        "======================================================================"
        "\n");
    printf("MatGen Matrix Scaling Tool\n");
    printf(
        "======================================================================"
        "\n");
    printf("Input:           %s\n", config.input_file);
    printf("Output:          %s\n", config.output_file);
    printf("Method:          %s\n", config.method);
    printf("Target size:     %llu × %llu\n",
           (unsigned long long)config.new_rows,
           (unsigned long long)config.new_cols);
    printf("Policy:          %s (resolved: %s)\n", config.policy_str,
           matgen_exec_policy_name(resolved_policy));

    if (strcmp(config.method, "nearest") == 0) {
      printf("Collision:       %s\n",
             collision_policy_name(config.collision_policy));
    }

    if (strcmp(config.method, "lanczos") == 0) {
      printf("Note:            Lanczos uses kernel width=3\n");
    }

    if (strcmp(config.method, "fft") == 0) {
      printf("Note:            FFT uses frequency-domain interpolation\n");
    }

    if (strcmp(config.method, "wavelet") == 0) {
      printf("Note:            Wavelet uses 2D Haar transform with block processing\n");
    }

#ifdef MATGEN_HAS_OPENMP
    if (resolved_policy == MATGEN_EXEC_PAR) {
      printf("OpenMP threads:  %d\n", matgen_exec_get_num_threads());
    }
#endif

#ifdef MATGEN_HAS_CUDA
    if (resolved_policy == MATGEN_EXEC_PAR_UNSEQ) {
      printf("CUDA devices:    %d\n", matgen_exec_get_num_cuda_devices());
      if (config.cuda_device >= 0) {
        printf("CUDA device ID:  %d\n", config.cuda_device);
      }
    }
#endif

#ifdef MATGEN_HAS_MPI
    if (resolved_policy == MATGEN_EXEC_MPI) {
      printf("MPI processes:   %d\n", size);
    }
#endif

    printf(
        "======================================================================"
        "\n");
  }

  // Step 1: Load input matrix (rank 0 only)
  if (!config.quiet && rank == 0) {
    printf("\n[1/4] Loading input matrix...\n");
  }

  matgen_coo_matrix_t* input_coo = NULL;
  matgen_csr_matrix_t* input_csr = NULL;
  matgen_index_t global_rows = 0;
  matgen_index_t global_cols = 0;
  matgen_size_t global_nnz = 0;

  // Only rank 0 reads the file
  if (rank == 0) {
    input_coo = matgen_mtx_read(config.input_file, NULL);
    if (!input_coo) {
      fprintf(stderr, "Error: Failed to load input matrix from '%s'\n",
              config.input_file);
#ifdef MATGEN_HAS_MPI
      broadcast_error(1);
      MPI_Finalize();
#endif
      return 1;
    }

    if (config.verbose) {
      printf("  Loaded COO matrix: %llu × %llu, nnz = %llu\n",
             (unsigned long long)input_coo->rows,
             (unsigned long long)input_coo->cols,
             (unsigned long long)input_coo->nnz);
    }

    global_rows = input_coo->rows;
    global_cols = input_coo->cols;
    global_nnz = input_coo->nnz;
  }

#ifdef MATGEN_HAS_MPI
  if (resolved_policy == MATGEN_EXEC_MPI) {
    // Broadcast success to other ranks
    int status = 0;
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast global dimensions
#if defined(MATGEN_INDEX_64)
    MPI_Bcast(&global_rows, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_cols, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
#else
    MPI_Bcast(&global_rows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_cols, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
#endif

#if defined(MATGEN_SIZE_64)
    MPI_Bcast(&global_nnz, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
#else
    MPI_Bcast(&global_nnz, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
#endif
  }
#endif

  // Step 2: Convert to CSR (rank 0, then distribute if MPI)
  if (!config.quiet && rank == 0) {
    printf("[2/4] Converting to CSR format...\n");
  }

  if (rank == 0) {
    // Convert COO to CSR on rank 0 using sequential policy for local conversion
    input_csr = matgen_coo_to_csr_with_policy(input_coo, MATGEN_EXEC_SEQ);
    matgen_coo_destroy(input_coo);
    input_coo = NULL;

    if (!input_csr) {
      fprintf(stderr, "Error: Failed to convert matrix to CSR format\n");
#ifdef MATGEN_HAS_MPI
      if (resolved_policy == MATGEN_EXEC_MPI) {
        broadcast_error(1);
        MPI_Finalize();
      }
#endif
      return 1;
    }

    if (config.show_stats) {
      print_matrix_stats("Input Matrix Statistics", input_csr);
    }
  }

#ifdef MATGEN_HAS_MPI
  matgen_csr_matrix_t* local_input_csr = NULL;

  if (resolved_policy == MATGEN_EXEC_MPI) {
    // Broadcast success status
    int status = 0;
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter the input matrix from rank 0 to all ranks
    if (!config.quiet && rank == 0) {
      printf("  Distributing matrix to %d MPI processes...\n", size);
    }

    local_input_csr = matgen_csr_scatter(input_csr);

    if (!local_input_csr) {
      if (rank == 0) {
        fprintf(stderr, "Error: Failed to scatter matrix to MPI processes\n");
      }
      if (input_csr) {
        matgen_csr_destroy(input_csr);
      }
      MPI_Finalize();
      return 1;
    }

    if (config.verbose && rank == 0) {
      printf("  Matrix distributed successfully\n");
    }
  } else {
    // Non-MPI policy: use the input matrix directly (only valid on rank 0)
    local_input_csr = input_csr;
  }
#else
  matgen_csr_matrix_t* local_input_csr = input_csr;
#endif

  // Step 3: Scale matrix
  if (!config.quiet && rank == 0) {
    printf("\n[3/4] Scaling matrix using %s interpolation...\n", config.method);
    printf("  Backend: %s\n", matgen_exec_policy_name(resolved_policy));
  }

  matgen_csr_matrix_t* local_output_csr = NULL;
  matgen_error_t err;

  double start = get_wall_time();

#ifdef MATGEN_HAS_MPI
  // Synchronize before timing for accurate measurement
  if (resolved_policy == MATGEN_EXEC_MPI) {
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif

  if (strcmp(config.method, "nearest") == 0) {
    err = matgen_scale_nearest_neighbor_with_policy_detailed(
        policy, local_input_csr, config.new_rows, config.new_cols,
        config.collision_policy, &local_output_csr);
  } else if (strcmp(config.method, "bilinear") == 0) {
    err = matgen_scale_bilinear_with_policy(policy, local_input_csr,
                                            config.new_rows, config.new_cols,
                                            &local_output_csr);
  } else if (strcmp(config.method, "lanczos") == 0) {
    err = matgen_scale_lanczos_with_policy(policy, local_input_csr,
                                           config.new_rows, config.new_cols,
                                           &local_output_csr);
  } else if (strcmp(config.method, "fft") == 0) {
    err = matgen_scale_fft_with_policy(policy, local_input_csr,
                                       config.new_rows, config.new_cols,
                                       &local_output_csr);
  } else {  // wavelet
    err = matgen_scale_wavelet_with_policy(policy, local_input_csr,
                                           config.new_rows, config.new_cols,
                                           &local_output_csr);
  }

#ifdef MATGEN_HAS_MPI
  // Synchronize after scaling for accurate timing
  if (resolved_policy == MATGEN_EXEC_MPI) {
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif

  double end = get_wall_time();
  double elapsed_sec = end - start;

  if (err != MATGEN_SUCCESS) {
    if (rank == 0) {
      fprintf(stderr, "Error: Matrix scaling failed with error code %d\n", err);
    }
#ifdef MATGEN_HAS_MPI
    if (resolved_policy == MATGEN_EXEC_MPI) {
      if (local_input_csr && local_input_csr != input_csr) {
        matgen_csr_destroy(local_input_csr);
      }
      if (input_csr) {
        matgen_csr_destroy(input_csr);
      }
      MPI_Finalize();
    }
#endif
    if (input_csr) {
      matgen_csr_destroy(input_csr);
    }
    return 1;
  }

  if (!config.quiet && rank == 0) {
    printf("  Scaling completed in %.3f ms (%.6f seconds)\n",
           elapsed_sec * 1000.0, elapsed_sec);
  }

  // Gather output matrix to rank 0 if using MPI
  matgen_csr_matrix_t* output_csr = NULL;

#ifdef MATGEN_HAS_MPI
  if (resolved_policy == MATGEN_EXEC_MPI) {
    if (!config.quiet && rank == 0) {
      printf("  Gathering results from %d MPI processes...\n", size);
    }

    output_csr =
        matgen_csr_gather(local_output_csr, config.new_rows, config.new_cols);

    // Clean up local matrices
    if (local_output_csr) {
      matgen_csr_destroy(local_output_csr);
    }
    if (local_input_csr && local_input_csr != input_csr) {
      matgen_csr_destroy(local_input_csr);
    }

    if (rank == 0 && !output_csr) {
      fprintf(stderr, "Error: Failed to gather matrix from MPI processes\n");
      if (input_csr) {
        matgen_csr_destroy(input_csr);
      }
      MPI_Finalize();
      return 1;
    }
  } else {
    output_csr = local_output_csr;
  }
#else
  output_csr = local_output_csr;
#endif

  if (config.show_stats && rank == 0) {
    print_matrix_stats("Output Matrix Statistics", output_csr);
  }

  // Step 4: Write output matrix (rank 0 only)
  if (!config.quiet && rank == 0) {
    printf("\n[4/4] Writing output matrix...\n");
  }

  if (rank == 0) {
    err = matgen_mtx_write_csr(config.output_file, output_csr);
    if (err != MATGEN_SUCCESS) {
      fprintf(stderr, "Error: Failed to write output matrix to '%s'\n",
              config.output_file);
      exit_code = 1;
    } else if (!config.quiet) {
      printf("  Output written successfully to '%s'\n", config.output_file);
    }
  }

#ifdef MATGEN_HAS_MPI
  // Broadcast exit code to all ranks
  if (resolved_policy == MATGEN_EXEC_MPI) {
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
#endif

  // Summary (rank 0 only)
  if (!config.quiet && rank == 0 && exit_code == 0) {
    printf(
        "\n===================================================================="
        "==\n");
    printf("Scaling Summary\n");
    printf(
        "======================================================================"
        "\n");
    printf("Input:           %llu × %llu, nnz = %llu\n",
           (unsigned long long)global_rows, (unsigned long long)global_cols,
           (unsigned long long)global_nnz);
    printf("Output:          %llu × %llu, nnz = %llu\n",
           (unsigned long long)output_csr->rows,
           (unsigned long long)output_csr->cols,
           (unsigned long long)output_csr->nnz);

    matgen_value_t input_density = (matgen_value_t)global_nnz /
                                   (matgen_value_t)(global_rows * global_cols);
    matgen_value_t output_density =
        (matgen_value_t)output_csr->nnz /
        (matgen_value_t)(output_csr->rows * output_csr->cols);

    printf("Input density:   %.6f (%.2f%%)\n", input_density,
           input_density * 100.0);
    printf("Output density:  %.6f (%.2f%%)\n", output_density,
           output_density * 100.0);
    printf("Backend:         %s\n", matgen_exec_policy_name(resolved_policy));
#ifdef MATGEN_HAS_MPI
    if (resolved_policy == MATGEN_EXEC_MPI) {
      printf("MPI processes:   %d\n", size);
    }
#endif
    printf("Time:            %.3f ms\n", elapsed_sec * 1000.0);
    printf(
        "======================================================================"
        "\n");
    printf("\nDone!\n");
  }

  // Cleanup
  if (rank == 0) {
    if (input_csr) {
      matgen_csr_destroy(input_csr);
    }
    if (output_csr) {
      matgen_csr_destroy(output_csr);
    }
  }

#ifdef MATGEN_HAS_MPI
  MPI_Finalize();
#endif

  return exit_code;
}
