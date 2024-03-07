# LU Decomposition

Implementation of parallel algorithms for Gaussian elimination to factor a dense N x N matrix into an upper-triangular and a lower-triangular matrix, sequentially, with incremental improvements by parallelisation  using `pthread` and `openmp`libraries. 

## Running instructions

The detailed code functionalities and descriptions have been reported in the `report.pdf`. Further, we have commented and explained the modules and helper functions to our best attempts for easy readability.

We submit a `Makefile`, which compiles all the relevant files using the command `make`. The execution instructions are mentioned below. 

## File Structure

### Sequential Files

Housed in `sequential/`.

`seq_LU.cpp` - is where a clean, commented, sequential version of the LU Decomposer is kept to serve as a reference and a baseline for comparison. Also prints `exec_time` and  `L2,1` norm of the residual array to aid in performance analysis.

To run the executable, simply execute `./seq.out <matrixDimension> 1`. Here, the "1" is a vestigial remnant of the numthreads to be developed in the parallel codes.

### OpenMP

Housed in `openmp/`.

`openmp_LU.cpp` - is where the OpenMP aided parallel LU is developed.

And run using `./omp.out <matrixDimension> <numThreads>`

`omp_LU_transpose.cpp` - is an alternative version where we experimented with the Column Major Order Matrices instead.

### pthreads


Housed in `pthreads/`.

`pthread_LU.cpp` and `pthreads_LU_alt.cpp` are the two variations of pthread implementations. The details can be found in the report.

And run using `./pth.out <matrixDimension> <numThreads>`


### Helpers

`/plots` directory houses the code for generating the plots for the times, efficiencies and speedups obtained. It is also were these generated plots are saved.


## Team

Goonjan Saha         
Gurarmaan S. Panjeta <br>
Viraj Agashe <br>
