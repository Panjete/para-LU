# Parallel LU Decomposition

Implementation of parallel algorithms for Gaussian elimination to factor a dense N x N matrix into an upper-triangular and a lower-triangular matrix, using `pthread` and `openmp`.

`hello.c` - contains sample code for OpenOP, to serve as a reference

`omp_LU.c` - is where the OpenMP aided parallel LU is being developed

`seq_LU.c` - is where a clean, commented, sequential version of the LU Decomposer is kept to serve as a reference and a baseline for comparison. Also prints `exec_time` and  `L2,1` norm of the residual array to aid in performance analysis.

To work on :

- Time taken on swaps somehow increases in the parallel OpenMP implementation
- Focus on reducing time in the update_A step


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


After identifying that l[i][k] remains constant in the inner loop, 

2 threads :

Initialised matrices!
Time taken by the program: 296 ms
Time taken for max: 2 ms
Time taken for swap: 10 ms
Time taken for lu updates: 7 ms
Time taken for a updates: 273 ms

4 threads : 

Time taken by the program: 196 ms
Time taken for max: 2 ms
Time taken for swap: 17 ms
Time taken for lu updates: 10 ms
Time taken for a updates: 164 ms

8 threads:

Time taken by the program: 249 ms
Time taken for max: 3 ms
Time taken for swap: 39 ms
Time taken for lu updates: 22 ms
Time taken for a updates: 181 ms


After using pointer increments instead of accesses; storing pointers, and incrementing them, in order to do optimised pointer calculation rather than adding j*8 to base index for every j :

2 threads :

Time taken by the program: 289 ms
Time taken for max: 2 ms
Time taken for swap: 11 ms
Time taken for lu updates: 8 ms
Time taken for a updates: 265 ms

4 threads:

Time taken by the program: 189 ms
Time taken for max: 3 ms
Time taken for swap: 17 ms
Time taken for lu updates: 11 ms
Time taken for a updates: 155 ms

8 threads:

Time taken by the program: 230 ms
Time taken for max: 3 ms
Time taken for swap: 38 ms
Time taken for lu updates: 21 ms
Time taken for a updates: 164 ms

Using upd_A_v3 instead, and using seq code if k too low :

2 threads :

Time taken by the program: 261 ms
Time taken for max: 1 ms
Time taken for swap: 9 ms
Time taken for lu updates: 7 ms
Time taken for a updates: 240 ms

4 threads:

Time taken by the program: 182 ms
Time taken for max: 2 ms
Time taken for swap: 17 ms
Time taken for lu updates: 11 ms
Time taken for a updates: 147 ms

8 threads:

Time taken by the program: 217 ms
Time taken for max: 2 ms
Time taken for swap: 37 ms
Time taken for lu updates: 21 ms
Time taken for a updates: 153 ms