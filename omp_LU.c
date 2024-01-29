#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define SWAP(T, a, b) do {T tmp = a; a = b; b = tmp; } while (0)

/* Parallel Implementation of the LU decomposition , via OpenMP 

Components parallelised :-

1) "parallel for" for setting pointer storing arrays of size n
2) "parallel" at init_a/l/u of size n^2, n^2 and n^2 respectively

*/

void LU(int n, int t); /* Wrapper function for the decomposition */

void init_a(int n, double** m); /* Initialising the matrix with random values */
void init_l(int n, double** m); /* Initialising the output L with lower triangular values */
void init_u(int n, double** m); /* Initialising the output u with upper triangular values */
void init_pi(int n, int* m); /* Initialising the output pi with linear ints */

/* Funtions for verification of Convergence via L2,1 norm */
void mat_mult(double** m1, double** m2, double** m3, int n); // compute L*U and save in a_prime
void mat_rearrange(double** m, int* pi, int n); // compute a_reperm from a and pi
void mat_sub(double** m1, double** m2, int n); // subtract a_reperm - a_prime and save in a_reperm
void mat_print(double** m, int n); // For Debugging and manual verif
double L2c1(double** m, int n); // computing the L2,1 norm of the difference, a_reperm

int main(int argc, char* argv[]) {

    int n = strtol(argv[1], NULL, 10); // User input
    int t = strtol(argv[2], NULL, 10); // User input

    LU(n, t); // Pass control to the LU-decomposer

    return 0; 
}   

void LU(int n, int t){
    // This storage mechanism gives seg fault at n = 590, but a cubic time increase is verifies
    double *a[n], *l[n], *u[n], *pa[n]; // Pointers to the matrice's rows
    double a_data[n*n], pa_data[n*n]; // Where all data is stored
    int pi[n]; /* Here, the vector pi is a compact representation of a permutation matrix p(n,n), 
                  which is very sparse. For the ith row of p, pi(i) stores the column index of
                  the sole position that contains a 1.*/

#pragma parallel for num_threads(t)
    for (int i = 0; i < n; i++){
        a[i] = &(a_data[i * n]); 
        //double l_data[n], u_data[n];
        l[i] = (double*)malloc(n * sizeof(double));
        u[i] = (double*)malloc(n * sizeof(double));
        pa[i] = &(pa_data[i * n]);
    }

#pragma parallel num_threads(t)
{
    init_a(n, a);
}
#pragma parallel num_threads(t)
{
    init_l(n, l);
}
#pragma parallel num_threads(t)
{
    init_u(n, u);
}
    init_pi(n, pi);

    for(int i = 0; i < n*n; i++) {
        pa_data[i] = a_data[i];
    }
    
    // printf("Matrix A at the start = \n");
    // mat_print(a, n);

    // printf("Matrix L at the start = \n");
    // mat_print(l, n);

    // printf("Matrix U at the start = \n");
    // mat_print(u, n);

    clock_t t_clock; 
    t_clock = clock(); 

    // k is the column here
    for(int k = 0; k < n; k++){  
        // Might Confuse indexing
        // Remember we shifted from 1 indexing in pseudo-code to 0-indexing here

        double max = 0;
        int k_prime; // store the row(i) with the max value (pivot) in this column (k)
        for(int i = k ; i < n; i++){
            if(max < fabs(a[i][k])){
                max = fabs(a[i][k]);
                k_prime = i;
            }
        }
        if(max == 0){
            printf("Error : Singular Matrix");
            return;
        }

        // Now that pivot has been discovered, start swapping

        SWAP(int, pi[k], pi[k_prime]);
        for(int jj = 0; jj < n; jj++){
            SWAP(double, a[k][jj], a[k_prime][jj]);
        }
        for(int jj = 0; jj < k; jj++){
            SWAP(double, l[k][jj], l[k_prime][jj]);
        }
        u[k][k] = a[k][k];

        // Swaps Completed. Now re-adjust l and u appropriately

        for(int i = k+1; i < n; i++){
            l[i][k] = a[i][k]/u[k][k];
            u[k][i] = a[k][i];
        }
        for(int i = k+1; i < n; i++){
            for(int j = k+1; j < n; j++){
                a[i][j] = a[i][j] - l[i][k]*u[k][j];
            }
        }
        // Complete for the this iteration of k
    }

    // LU Computed. Now testing for time, and L2,1 convergence.

    t_clock = clock() - t_clock; 
    double time_taken = ((double)t_clock)/CLOCKS_PER_SEC; // in seconds 
    printf("LU for n = %d took %f seconds to execute \n", n, time_taken);

    double *a_prime[n];

    for (int i = 0; i < n; i++){
        a_prime[i] = (double*)malloc(n * sizeof(double));
    }

    // Now use permutation matrix to permute rows of A
    for(int i = 0; i < n; i++) {
        pa[i] = &(pa_data[pi[i]*n]);
    }
    
    mat_mult(l, u, a_prime, n);  // a_prime = LU

    // printf("matrix L = \n");
    // mat_print(l, n);

    // printf("matrix U = \n");
    // mat_print(u, n);

    // printf("Matrix LU  = \n");
    // mat_print(a_prime, n);

    // printf("Pi = \n");
    // for(int i = 0; i < n; i++){printf("pi[%d] = %d \n",i,  pi[i]);}

    mat_sub(pa, a_prime, n);      // a = PA - LU

    // printf("Matrix PA-LU  = \n");
    // mat_print(pa, n);

    double convg_error = L2c1(pa, n);
    printf("LU Convergence error for n = %d is %f  \n", n, convg_error);


    // Freeing space
    for(int i = 0; i < n; i++){
        free(u[i]);
        free(l[i]);
    }

    return;
}

void init_a(int n, double** m){

    /* The drand48() function return non-negative, double-precision, floating-point values
      uniformly distributed over the interval [0.0 , 1.0]. */

    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();

    int per_thread = n/thread_count;
    int i_start = my_rank * per_thread;
    int i_end   = i_start + per_thread;

    if(my_rank == thread_count -1){i_end = n;}


    srand(time(NULL)^my_rank);
    for (int i = i_start; i < i_end; i++){ // Row
        for (int j = 0; j < n; j++){ // Column
            m[i][j] = drand48();
        }
    }
    return;
}

void init_u(int n, double** m){

    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();

    int per_thread = n/thread_count;
    int i_start = my_rank * per_thread;
    int i_end   = i_start + per_thread;

    if(my_rank == thread_count -1){i_end = n;}

    for (int i = i_start; i < i_end; i++){ // Row
        for (int j = 0; j < n; j++){ // Column // REMEMBER : j < i for (below diagonal case), currently setting all to 0
            m[i][j] = 0; // Initialising matrix with 0s below the diagonal
        }
    }
    return;
}

void init_l(int n, double** m){

    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();

    int per_thread = n/thread_count;
    int i_start = my_rank * per_thread;
    int i_end   = i_start + per_thread;

    if(my_rank == thread_count -1){i_end = n;}

    for (int i = i_start; i < i_end; i++){ // Row
        m[i][i] = 1;
        for (int j = 0; j < i; j++){ // Column - this is unnecessary below diagonal zeroing as well
            m[i][j] = 0;
        }
        for (int j = i+1; j < n; j++){ // Column
            m[i][j] = 0;
        }
    }
    return;
}

void init_pi(int n, int* m){

    for (int i = 0; i < n; i++){ // Row
        m[i] = i;
    }
    return;
}

void mat_mult(double** m1, double** m2, double** m3, int n){
    for(int row = 0; row < n; row++){
        for(int col = 0; col < n; col++){
            double cursum = 0;
            for(int k = 0; k < n; k++){
                cursum += m1[row][k] * m2[k][col];
            }
            m3[row][col] = cursum;
        }
    }
    return;
}

void mat_rearrange(double** m, int* pi, int n){
    // Now, goal is computing PA
    // the pi[i] tells what row from m should be the ith row
    // Thus, m[i] <- m[pi[i]]
    double *b[n];
    for(int i = 0; i < n; i++){
        b[i] = m[i]; // Temp array for transfer
    }
    for(int i = 0; i < n; i++){
        m[i] = b[pi[i]]; // store pointers after swapping
    }
    return;
}

void mat_sub(double** m1, double** m2, int n){
    for(int row = 0; row < n; row++){
        for(int col = 0; col < n; col++){
            m1[row][col] -= m2[row][col];
        }
    }
    return;
}

double L2c1(double** m, int n){
    // L2,1 norm of a matrix is the sum of L2 norms of the columns
    double norm = 0;
    for(int j = 0; j < n; j++){ // column j
        double cur_norm = 0;
        for(int i = 0; i < n; i++){
            cur_norm += (m[i][j] * m[i][j]);
        }
        norm += sqrt(cur_norm);
    }
    return norm;
}

void mat_print(double** m, int n){
    for(int row = 0; row < n; row++){
        for(int col = 0; col < n; col++){
            printf("%f ", m[row][col]);
        }
        printf("\n");
    }
    return;
}

/* 

Gurarmaan

Execute using command

clang -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp omp_LU.c -o exec2.out

*/