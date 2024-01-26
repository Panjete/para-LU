#include <stdio.h>
#include <stdlib.h>

/* Sequential Implementation of the LU decomposition as step 1 of the parallelisation procedure */

void LU(int n, int t); /* Wrapper function for the decomposition */

void init_a(int n, double** m); /* Initialising the matrix with random values */
void init_l(int n, double** m); /* Initialising the output L with lower triangular values */
void init_u(int n, double** m); /* Initialising the output u with upper triangular values */


int main(int argc, char* argv[]) {

    int n = strtol(argv[1], NULL, 10); // User input
    int t = strtol(argv[2], NULL, 10); // User input

    LU(n, t); // Pass control to the LU-decomposer

    return 0; 
}   

void LU(int n, int t){
    double *matrix[n];
    double my_data[n*n]; // Where all data is stored
    for (int i = 0; i < n; i++){
        matrix[i] = &(my_data[i * n]); 
    }

    init_a(n, matrix);

    return;
}

void init_a(int n, double** m){

    /* The drand48() function return non-negative, double-precision, floating-point values
      uniformly distributed over the interval [0.0 , 1.0]. */

    for (int i = 0; i < n; i++){ // Row
        for (int j = 0; j < n; j++){ // Column
            m[i][j] = drand48();
        }
    }
    return;
}


/* 

Gurarmaan

Execute using command

clang -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp seq_LU.c -o exec.out

*/