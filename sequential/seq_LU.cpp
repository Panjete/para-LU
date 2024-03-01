#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <chrono>
#include <iostream>
using namespace std;

#define SWAP(T, a, b) do {T tmp = a; a = b; b = tmp; } while (0)

/* Sequential Implementation of the LU decomposition as step 1 of the parallelisation procedure */

void LU(int n, int t); /* Wrapper function for the decomposition */

void init_a(int n, double** m); /* Initialising the matrix with random values */
void init_l(int n, double** m); /* Initialising the output L with lower triangular values */
void init_u(int n, double** m); /* Initialising the output U with upper triangular values */
void init_pi(int n, int* m); /* Initialising the output pi with linear ints */
void init_pa(int n, double** m1, double** m2); /* Storing original A as pa, for verification of convergence later */

/* Funtions for verification of Convergence via L2,1 norm */
void mat_mult(double** m1, double** m2, double** m3, int n); // compute L*U and save in a_prime
void mat_rearrange(double** m, int* pi, int n); // compute PA from a and pi and save in pa
void mat_sub(double** m1, double** m2, int n); // subtract pa - a_prime and save in pa
void mat_print(double** m, int n); // For Debugging and manual verif
double L2c1(double** m, int n); // computing the L2,1 norm of the difference, a_reperm

int main(int argc, char* argv[]) {

    int n = strtol(argv[1], NULL, 10); // User input
    int t = strtol(argv[2], NULL, 10); // User input

    LU(n, t); // Pass control to the LU-decomposer

    return 0; 
}   

void LU(int n, int t){
    double *a[n], *l[n], *u[n], *pa[n], *a_prime[n]; // Pointers to the matrice's rows
    int pi[n]; /* Here, the vector pi is a compact representation of a permutation matrix p(n,n), 
                  which is very sparse. For the ith row of p, pi(i) stores the column index of
                  the sole position that contains a 1.*/


    for (int i = 0; i < n; i++){
        a[i] = (double*)malloc(n * sizeof(double));
        l[i] = (double*)malloc(n * sizeof(double));
        u[i] = (double*)malloc(n * sizeof(double));
        pa[i] = (double*)malloc(n * sizeof(double));
        a_prime[i] = (double*)malloc(n * sizeof(double));
    }

    init_a(n, a);
    init_l(n, l);
    init_u(n, u);
    init_pi(n, pi);
    init_pa(n, a, pa);
    printf("Initialised matrices!\n");

    clock_t t_clock; 
    t_clock = clock(); 

    std::chrono::microseconds t_max(0);
    std::chrono::microseconds t_swap(0);
    std::chrono::microseconds t_lu(0);
    std::chrono::microseconds t_a(0);

    auto start = std::chrono::high_resolution_clock::now();

    // k is the column here
    for(int k = 0; k < n; k++){  
        // Might Confuse indexing
        // Remember we shifted from 1 indexing in pseudo-code to 0-indexing here

        auto t1 = std::chrono::high_resolution_clock::now();
        double max = 0;
        int k_prime; // store the row(i) with the max value (pivot) in this column (k)
        for(int i = k ; i < n; i++){
            if(max < fabs(a[i][k])){
                max = fabs(a[i][k]);
                k_prime = i;
            }
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
        t_max += duration;


        if(max == 0){
            printf("Error : Singular Matrix");
            return;
        }

        // Now that pivot has been discovered, start swapping

        t1 = std::chrono::high_resolution_clock::now();
        SWAP(int, pi[k], pi[k_prime]);
        double* a_k = a[k], *a_k_prime = a[k_prime]; 
        for(int jj = 0; jj < n; jj++){
            SWAP(double, a_k[jj], a_k_prime[jj]);
        }
        double *l_k = l[k], *l_k_prime = l[k_prime];
        for(int jj = 0; jj < k; jj++){
            SWAP(double, l_k[jj], l_k_prime[jj]);
        }
        u[k][k] = a[k][k];
        t2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
        t_swap += duration;

        // Swaps Completed. Now re-adjust l and u appropriately

        t1 = std::chrono::high_resolution_clock::now();
        double* u_k = u[k];
        for(int i = k+1; i < n; i++){
            l[i][k] = a[i][k]/u[k][k];
            u_k[i] = a_k[i];
        }
        t2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
        t_lu += duration;

        t1 = std::chrono::high_resolution_clock::now();
        for(int i = k+1; i < n; i++){
            for(int j = k+1; j < n; j++){
                a[i][j] = a[i][j] - l[i][k]*u_k[j];
            }
        }
        t2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
        t_a += duration;

        // Complete for the this iteration of k
    }

    // LU Computed. Now testing for time, and L2,1 convergence.

    t_clock = clock() - t_clock; 
    double time_taken = ((double)t_clock)/CLOCKS_PER_SEC; // in seconds 
    printf("LU for n = %d took %f seconds to execute \n", n, time_taken);

    t_clock = clock() - t_clock; 

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Time taken by the program: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << " ms" << std::endl;
    std::cout << "Time taken for max: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_max).count() << " ms" << std::endl;
    std::cout << "Time taken for swap: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_swap).count() << " ms" << std::endl;
    std::cout << "Time taken for lu updates: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_lu).count() << " ms" << std::endl;
    std::cout << "Time taken for a updates: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_a).count() << " ms" << std::endl;

    
    mat_rearrange(pa, pi, n);    // Now use permutation matrix to permute rows of A
    mat_mult(l, u, a_prime, n);  // a_prime = LU
    mat_sub(pa, a_prime, n);     // pa = PA - LU

    //printf("Matrix PA-LU  = \n");
   // mat_print(pa, n);

    double convg_error = L2c1(pa, n);
    printf("LU Convergence error for n = %d is %f  \n", n, convg_error);

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

void init_pa(int n, double** m1, double** m2){

    for(int i = 0; i < n*n; i++) {
        m2[i/n][i%n] = m1[i/n][i%n];
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

clang++ -std=c++11 -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp seq_LU.cpp -o exec_seq.out

*/