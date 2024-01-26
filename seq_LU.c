#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SWAP(T, a, b) do {T tmp = a; a = b; b = tmp; } while (0)

/* Sequential Implementation of the LU decomposition as step 1 of the parallelisation procedure */

void LU(int n, int t); /* Wrapper function for the decomposition */

void init_a(int n, double** m); /* Initialising the matrix with random values */
void init_l(int n, double** m); /* Initialising the output L with lower triangular values */
void init_u(int n, double** m); /* Initialising the output u with upper triangular values */
void init_pi(int n, int* m); /* Initialising the output pi with linear ints */


int main(int argc, char* argv[]) {

    int n = strtol(argv[1], NULL, 10); // User input
    int t = strtol(argv[2], NULL, 10); // User input

    LU(n, t); // Pass control to the LU-decomposer

    return 0; 
}   

void LU(int n, int t){
    double *a[n], *l[n], *u[n]; // Pointers to the matrice's rows
    double a_data[n*n], u_data[n*n], l_data[n*n]; // Where all data is stored
    int pi[n]; /* Here, the vector pi is a compact representation of a permutation matrix p(n,n), 
                  which is very sparse. For the ith row of p, pi(i) stores the column index of
                  the sole position that contains a 1.*/


    for (int i = 0; i < n; i++){
        a[i] = &(a_data[i * n]); 
        l[i] = &(l_data[i * n]); 
        u[i] = &(u_data[i * n]); 
    }



    init_a(n, a);
    init_l(n, l);
    init_u(n, u);
    init_pi(n, pi);

    for(int k = 0; k < n; k++){  // k is the column here
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
        for(int jj = 0; jj < k-1; jj++){
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

    // LU Computed. Now test for time, and L1/L2 convergence.


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

void init_u(int n, double** m){

    for (int i = 0; i < n; i++){ // Row
        for (int j = 0; j < i; j++){ // Column
            m[i][j] = 0; // Initialising matrix with 0s below the diagonal
        }
    }
    return;
}

void init_l(int n, double** m){

    for (int i = 0; i < n; i++){ // Row
        m[i][i] = 1;
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


/* 

Gurarmaan

Execute using command

clang -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp seq_LU.c -o exec.out

*/