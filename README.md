# Parallel LU Decomposition

Implementation of parallel algorithms for Gaussian elimination to factor a dense N x N matrix into an upper-triangular and a lower-triangular matrix, using `pthread` and `openmp`.

`hello.c` - contains sample code for OpenOP, to serve as a reference

`omp_LU.c` - is where the OpenMP aided parallel LU is being developed

`seq_LU.c` - is where a clean, commented, sequential version of the LU Decomposer is kept to serve as a reference and a baseline for comparison. Also prints `exec_time` and  `L2,1` norm of the residual array to aid in performance analysis.


On 1/2/24, after meeting, the Times after shifting to cpp are, 2 threads :

Time taken by the program: 592 ms
Time taken for max: 1 ms
Time taken for swap: 10 ms
Time taken for lu updates: 7 ms
Time taken for a updates: 569 ms

Sequential times : 
Time taken by the program: 840 ms
Time taken for max: 1 ms
Time taken for swap: 3 ms
Time taken for lu updates: 3 ms
Time taken for a updates: 829 ms


After converting u[k][j] -> u_k[j], for 2 threads:

Time taken by the program: 397 ms
Time taken for max: 2 ms
Time taken for swap: 9 ms
Time taken for lu updates: 7 ms
Time taken for a updates: 375 ms

for 4 threads:

Time taken by the program: 254 ms
Time taken for max: 2 ms
Time taken for swap: 17 ms
Time taken for lu updates: 10 ms
Time taken for a updates: 221 ms

for 8 threads:

Time taken by the program: 293 ms
Time taken for max: 2 ms
Time taken for swap: 40 ms
Time taken for lu updates: 23 ms
Time taken for a updates: 224 ms

Seq time :

Time taken by the program: 757 ms
Time taken for max: 1 ms
Time taken for swap: 3 ms
Time taken for lu updates: 3 ms
Time taken for a updates: 747 ms