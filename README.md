# Parallel LU Decomposition

Implementation of parallel algorithms for Gaussian elimination to factor a dense N x N matrix into an upper-triangular and a lower-triangular matrix, using `pthread` and `openmp`.

`hello.c` - contains sample code for OpenOP, to serve as a reference

`omp_LU.c` - is where the OpenMP aided parallel LU is being developed

`seq_LU.c` - is where a clean, commented, sequential version of the LU Decomposer is kept to serve as a reference and a baseline for comparison. Also prints `exec_time` and  `L2,1` norm of the residual array to aid in performance analysis.
