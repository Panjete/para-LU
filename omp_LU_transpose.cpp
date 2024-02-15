#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <cstring>
#include <chrono>
#include <iostream>
using namespace std;

#define SWAP(T, a, b) do {T tmp = a; a = b; b = tmp; } while (0)
#define TRANSPOSE true

/* Parallel Implementation of the LU decomposition , via OpenMP 

Components parallelised :-

1) "parallel for" for setting pointer storing arrays of size n
2) "parallel" at init_a/l/u of size n^2, n^2 and n^2 respectively
3) "parallel for" at init_pa of size n^2, which does copying of A
4) "parallel for" at swapping rows of a and l
5) "parallel for" at updating u, l and a

To do : Test for n=8000 - too big, currently does not scale well enough

*/

void LU(int n, int t); /* Wrapper function for the decomposition */

/* Initialisers */
void init_a(int n, double** m); /* Initialising the matrix with random values */
void init_l(int n, double** m); /* Initialising the output L with lower triangular values */
void init_u(int n, double** m); /* Initialising the output U with upper triangular values */
void init_pi(int n, int* m, int t); /* Initialising the output pi with linear ints */
void init_pa(int n, double** m1, double** m2, int t); /* Storing original A as pa, for verification of convergence later */

/* LU Decomposer Modules */
int find_max(double** a, int n, int k);
void swap_row(double* m1, double* m2, int k_upto, int t); /* Swap R1 with R2, upto column k*/
void swap_row_v2(double* m1, double* m2, int k_upto); /* Swap R1 with R2, upto column k*/
void upd_UL(double** a, int n, int k, double* u_k, double** l, int t); /* Fill up the u & l matrices after the global kth iteration */
void upd_UL_v2(double** a, int n, int k, double* u_k, double** l, int t); /* Fill up the u & l matrices after the global kth iteration */
void upd_A(double** a, int n, int k, double* u_k, double** l, int t); /* Update a now that it's kth row has been processed */
void upd_A_v2(double** a, int n, int k, double* u_k, double** l, int t); /* Update a now that it's kth row has been processed */
void upd_A_seq(double** a, int n, int k, double* u_k, double** l, int t); /* Update a now that it's kth row has been processed */
void upd_A_v3(double** a, int n, int k, double* u_k, double** l, int t); /* Update a now that it's kth row has been processed */

int find_max_t(double** a, int n, int k);
void upd_A_seq_v4(double** a, int n, int k, double** u, double** l, int t);

// void swap_row(double* m1, double* m2, int k_upto, int t); 
void upd_UL_v4(double** a, int n, int k, double** u, double** l, int t);
void upd_A_v4(double** a, int n, int k, double** u, double** l, int t);

/* Funtions for verification of Convergence via L2,1 norm */
void mat_mult(double** m1, double** m2, double** m3, int n, int t); // compute L*U and save in a_prime
void mat_rearrange(double** m, int* pi, int n); // compute PA from a and pi and save in pa
void mat_sub(double** m1, double** m2, int n, int t); // subtract pa - a_prime and save in pa
void mat_print(double** m, int n); // For Debugging and manual verif
double L2c1(double** m, int n); // computing the L2,1 norm of the residual matrix

int main(int argc, char* argv[]) {

    int n = strtol(argv[1], NULL, 10); // User input
    int t = strtol(argv[2], NULL, 10); // User input

    if(t == 0){
        printf("Please pass non-zero number of threads!\n");
        return 0;
    }

    LU(n, t); // Pass control to the LU-decomposer

    return 0; 
}   

void LU(int n, int t){
    // This storage mechanism gives seg fault at n = 590, but a cubic time increase is verifies
    double *a[n], *l[n], *u[n], *pa[n], *a_prime[n]; // Pointers to the matrice's rows
    int pi[n]; /* Here, the vector pi is a compact representation of a permutation matrix p(n,n), 
                  which is very sparse. For the ith row of p, pi(i) stores the column index of
                  the sole position that contains a 1.*/

#pragma parallel for num_threads(t)
    for (int i = 0; i < n; i++){
        a[i] = (double*)malloc(n * sizeof(double));
        l[i] = (double*)malloc(n * sizeof(double));
        u[i] = (double*)malloc(n * sizeof(double));
        pa[i] = (double*)malloc(n * sizeof(double));
        a_prime[i] = (double*)malloc(n * sizeof(double));
    }

    #pragma omp parallel num_threads(t)
    {
        init_a(n, a);
    }
    #pragma omp parallel num_threads(t)
    {
        init_l(n, l);
    }
    #pragma omp parallel num_threads(t)
    {
        init_u(n, u);
    }

    init_pi(n, pi, t);    // Internally Parallelised
    init_pa(n, a, pa, t); // Internally Parallelised
    printf("Initialised matrices!\n");

    // Swap area for memcpy
    double* swp = (double*)malloc(n * sizeof(double));

    std::chrono::microseconds t_max(0);
    std::chrono::microseconds t_swap(0);
    std::chrono::microseconds t_lu(0);
    std::chrono::microseconds t_a(0);

    /* Transpose all the matrices here */
    double* a_t[n], *l_t[n], *u_t[n];

    if (TRANSPOSE) {
        for(int i = 0; i < n; i++){
            a_t[i] = (double*)malloc(n * sizeof(double));
            l_t[i] = (double*)malloc(n * sizeof(double));
            u_t[i] = (double*)malloc(n * sizeof(double));
        }
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                a_t[i][j] = a[j][i];
                l_t[i][j] = l[j][i];
                u_t[i][j] = u[i][j];
            }
        }
    }

    /* Print the matrix */
    // for(int i = 0; i < n; i++) {
    //     for(int j = 0; j < n; j++) {
    //         cout << "A" << i << " " << j << " = " << a_t[i][j] << endl;
    //     }
    // }
    
    auto start = std::chrono::high_resolution_clock::now();
    // k is the column here
    for(int k = 0; k < n; k++){  
        auto t1 = std::chrono::high_resolution_clock::now();
        int k_prime;
        if (TRANSPOSE){
            k_prime = find_max_t(a_t, n, k);
        } else {
            k_prime = find_max(a, n, k); // Find the pivot
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
        t_max += duration;

        if(k_prime == -1){ //Happens when all elements in this column == 0
            printf("Error : Singular Matrix");
            return;
        }

        // Now that pivot has been discovered, start swapping
        t1 = std::chrono::high_resolution_clock::now();

        if (TRANSPOSE) {
            SWAP(int, pi[k], pi[k_prime]);

            // SWAP cols of A 
            for(int i = 0; i < n; i++){
                SWAP(double, a_t[i][k], a_t[i][k_prime]);
            }

            // SWAP cols of l
            for(int i = 0; i < k; i++){
                SWAP(double, l_t[i][k], l_t[i][k_prime]);
            }

        } else {
            SWAP(int, pi[k], pi[k_prime]);
            // Swap rows of A
            double* temp = a[k];
            a[k] = a[k_prime];
            a[k_prime] = temp;
            // Swap rows of l
            memcpy(swp, l[k], sizeof(double) * k);
            memcpy(l[k], l[k_prime], sizeof(double) * k);
            memcpy(l[k_prime], swp, sizeof(double) * k);
        }

        t2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
        t_swap += duration;
        
        if (TRANSPOSE) {
            u_t[k][k] = a_t[k][k];
        } else {
            u[k][k] = a[k][k];
        }

        // Swaps Completed. Now re-adjust l, u and a appropriately
        // Updating U and L, and tracking time taken to do so
        t1 = std::chrono::high_resolution_clock::now();
        if (TRANSPOSE) {
            upd_UL_v4(a_t, n, k, u_t, l_t, t); 
        } else {
            upd_UL_v2(a, n, k, u[k], l, t);
        }
        t2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
        t_lu += duration;
        
        // Updating A and adding time taken to do so
        t1 = std::chrono::high_resolution_clock::now();
        if (TRANSPOSE) {
            if(n-k < t){
                upd_A_seq_v4(a_t, n, k, u_t, l_t, t); // Too small a workload to parallelise
            } else {
                upd_A_v4(a_t, n, k, u_t, l_t, t);
            }
        } else {
            if(n-k < t){
                upd_A_seq(a, n, k, u[k], l, t); // Too small a workload to parallelise
            } 
            else{
                #pragma omp parallel num_threads(t)
                {
                    upd_A_v3(a, n, k, u[k], l, t);
                }
            }
        }
        
        t2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
        t_a += duration;

        // Complete for the this iteration of k
    }
    
    if (TRANSPOSE){
        // Take the transpose and copy back a_t, l_t, u_t to a, l, u 
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                a[i][j] = a_t[j][i];
                l[i][j] = l_t[j][i];
                u[i][j] = u_t[i][j];
            }
        }
    }

    // LU Computed. Now testing for time, and L2,1 convergence.

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    cout << "Time taken by the program: " << chrono::duration_cast<chrono::milliseconds>(duration).count() << " ms" << endl;
    cout << "Time taken for max: " << chrono::duration_cast<chrono::milliseconds>(t_max).count() << " ms" << endl;
    cout << "Time taken for swap: " << chrono::duration_cast<chrono::milliseconds>(t_swap).count() << " ms" << endl;
    cout << "Time taken for lu updates: " << chrono::duration_cast<chrono::milliseconds>(t_lu).count() << " ms" << endl;
    cout << "Time taken for a updates: " << chrono::duration_cast<chrono::milliseconds>(t_a).count() << " ms" << endl;

    mat_rearrange(pa, pi, n); // Now use permutation matrix to permute rows of A
    mat_mult(l, u, a_prime, n, t);  // a_prime = LU
    mat_sub(pa, a_prime, n, t);      // pa = PA - LU

    double convg_error = L2c1(pa, n);
    printf("LU Convergence error for n = %d is %f  \n", n, convg_error);

    // Freeing space
    for(int i = 0; i < n; i++){
        free(u[i]); free(l[i]); free(a_prime[i]); free(a[i]); free(pa[i]);
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

void init_pi(int n, int* m, int t){
#pragma omp parallel for num_threads(t)
    for (int i = 0; i < n; i++){ // Row
        m[i] = i;
    }
    return;
}

void init_pa(int n, double** m1, double** m2, int t){
#pragma omp parallel for num_threads(t)
    for(int i = 0; i < n*n; i++) {
        m2[i/n][i%n] = m1[i/n][i%n];
    }
    return;
}

/* LU functions */

int find_max(double** a, int n, int k){
    int k_prime = -1;
    double max = 0;
// #pragma omp parallel for num_threads(t)
    for(int i = k ; i < n; i++){
        // Parallelization should work, because whatever thread encounters global maxima, it will find if true and set k_prime to i
        // What is some other thread also has it's if evaluated to true at this point, but just hasn't set the k_prime? It will now overlap and set the k_prime to the non-maxima index
        if(max < fabs(a[i][k])){
// #pragma omp critical
// {
            max = fabs(a[i][k]);
            k_prime = i;
//}
        }
    }

    return k_prime;
}

void swap_row(double* m1, double* m2, int k_upto, int t){
    
#pragma omp parallel for num_threads(t) 
    for(int jj = 0; jj < k_upto; jj++){
        double temp = m1[jj];
        m1[jj] = m2[jj];
        m1[jj] = temp;
    }

    return;
}

void swap_row_v2(double* m1, double* m2, int k_upto){
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();

    int per_thread = (k_upto+1)/thread_count;
    int i_start = my_rank * per_thread;
    int i_end   = i_start + per_thread;

    if(my_rank == thread_count -1){i_end = k_upto;}

    double temp;
    for (int jj = i_start; jj < i_end; jj++){ // Row
        temp = m1[jj];
        m1[jj] = m2[jj];
        m2[jj] = temp;
    }
    return;
}


void upd_UL(double** a, int n, int k, double* u_k, double** l, int t){

    double x = u_k[k];
    double* a_k = a[k];
#pragma omp parallel for num_threads(t) 
    for(int i = k+1; i < n; i++){ // k and n don't change, can afford to be shared variables
        l[i][k] = a[i][k]/x;
        u_k[i] = a_k[i];
    }

    return;
}; 

void upd_UL_v2(double** a, int n, int k, double* u_k, double** l, int t){

    double x = u_k[k];
    int work = n-k-1;
    if(work == 0){return;}
    if(work < t){
        double* a_k = a[k];
        for(int i = k+1; i < n; i++){ 
            l[i][k] = a[i][k]/x;
            u_k[i] = a_k[i];
        }
        return;
    }

    double* a_k = a[k];
    int per_thread = work/t;
    
#pragma omp parallel for num_threads(t)
    for(int my_rank = 0; my_rank < t; my_rank++){ 
        int i_start = k+1+ my_rank * per_thread;
        int i_end   = i_start + per_thread;
        if(my_rank == t-1){i_end = n;}
        for(int i = i_start; i < i_end; i++){
            l[i][k] = a[i][k]/x;
            u_k[i] = a_k[i];
        }
    }

    return;
}; 


void upd_A(double** a, int n, int k, double* u_k, double** l, int t){

#pragma omp parallel for num_threads(t) 
    for(int i = k+1; i < n; i++){
        for(int j = k+1; j < n; j++){ // j is declared inside the thread - private
            a[i][j] = a[i][j] - l[i][k]*u_k[j]; // k and n don't change, can afford to be shared variables
        }
    }

    return;
}; 


void upd_A_v2(double** a, int n, int k, double* u_k, double** l, int t){

    int work = n-k-1;
    if(work == 0){return;} // Happens when k == n-1
    if(work < t){ // Too small, simply use sequential code
        double* a_i;
        double* l_i;
        for(int i = k+1; i < n; i++){
            a_i = a[i]; l_i = l[i];
            for(int j = k+1; j < n; j++){
                a_i[j] = a_i[j] - l_i[k]*u_k[j];
            }
        }
        return;
    }

    int per_thread = work/t;
    
#pragma omp parallel for num_threads(t)
    for(int my_rank = 0; my_rank < t; my_rank++){ 
        int i_start = k+1+ my_rank * per_thread;
        int i_end   = i_start + per_thread;
        if(my_rank == t-1){i_end = n;}
        double* a_exp, *u_exp;
        double l_ik;
        for(int i = i_start; i < i_end; i++){
            l_ik = l[i][k];
            a_exp = &(a[i][k+1]); // works because k is guarenteed to be < n-1 
            u_exp = &(u_k[k+1]); // works because k is guarenteed to be < n-1 
            for(int j = k+1; j < n; j++){ 
                //a_i[j] -= l_ik * u_k[j];
                *(a_exp) -= l_ik * (*u_exp);
                a_exp++; u_exp++;
            }
        }
    }

    return;
};

void upd_A_seq(double** a, int n, int k, double* u_k, double** l, int t){
    for(int i = k+1; i < n; i++){
        for(int j = k+1; j < n; j++){ // j is declared inside the thread - private
            a[i][j] = a[i][j] - l[i][k]*u_k[j]; // k and n don't change, can afford to be shared variables
        }
    }

    return;
}


void upd_A_v3(double** a, int n, int k, double* u_k, double** l, int t){
    k++; // Incremented because most of the time we were using k+1 anyways
    int per_thread = (n-k)/omp_get_num_threads();
    int i_start = k +  omp_get_thread_num() * per_thread; // my_rank == omp_get_thread_num();
    if(omp_get_thread_num() == omp_get_num_threads()-1){per_thread = n-i_start;}

    double* a_exp, *u_exp = &(u_k[k]), *u_exp_init = &(u_k[k]); // works because k is guarenteed to be < n-1 
    double l_ik;
    for(int i = i_start; i < i_start + per_thread; i++){
        l_ik = l[i][k-1];
        a_exp = &(a[i][k]); // works because k is guarenteed to be < n-1 
        for(int j = k; j < n; j++){ 
            *(a_exp) -= l_ik * (*u_exp);
            a_exp++; u_exp++;
        }
        u_exp = u_exp_init;
        
    }

    return;
}

int find_max_t(double** a, int n, int k) {
    int k_prime = -1;
    double max = 0;
    // #pragma omp parallel for num_threads(t)
    for(int i = k ; i < n; i++){
        if(max < fabs(a[k][i])){
            // #pragma omp critical
            // {
                max = fabs(a[k][i]);
                k_prime = i;
            // }
        }
    }

    return k_prime;

}

void upd_A_seq_v4(double** a, int n, int k, double** u, double** l, int t) {
    double *u_k = u[k];
    for(int i = k+1; i < n; i++){
        for(int j = k+1; j < n; j++){ // j is declared inside the thread - private
            a[j][i] = a[j][i] - l[k][i]*u_k[j]; // k and n don't change, can afford to be shared variables
        }
    }

    return;

}


void upd_UL_v4(double** a, int n, int k, double** u, double** l, int t){
    double x = u[k][k];
    // cout << x << endl;
    double *u_k = u[k];
//     int work = n-k-1;
//     if(work == 0){return;}
//     if(work < t){
//         // double* a_k = a[k];
//         for(int i = k+1; i < n; i++){ 
//             l[k][i] = a[k][i]/x;
//             u_k[i] = a[i][k];
//         }
//         return;
//     }

//     // double* a_k = a[k];
//     int per_thread = work/t;
    
// #pragma omp parallel for num_threads(t)
//     for(int my_rank = 0; my_rank < t; my_rank++){ 
//         int i_start = k+1+ my_rank * per_thread;
//         int i_end   = i_start + per_thread;
//         if(my_rank == t-1){i_end = n;}
//         for(int i = i_start; i < i_end; i++){
//             l[k][i] = a[k][i]/x;
//             u_k[i] = a[i][k];
//         }
//     }
    for(int i = k+1; i < n; i++){ 
        l[k][i] = a[k][i]/x;
        u_k[i] = a[i][k];
    }
    return;

    // return;
}

void upd_A_v4(double** a, int n, int k, double** u, double** l, int t){
    int work = n-k-1;
    if(work == 0){return;} // Happens when k == n-1
    if(work < t){ // Too small, simply use sequential code
        double* a_i;
        double* l_i;
        for(int i = k+1; i < n; i++){
            for(int j = k+1; j < n; j++){
                a[j][i] = a[j][i] - l[k][i]*u[k][j];
                // a_i[j] = a_i[j] - l_i[k]*u_k[j];
            }
        }
        return;
    }

    int per_thread = work/t;
    
#pragma omp parallel for num_threads(t)
    for(int my_rank = 0; my_rank < t; my_rank++){ 
        int i_start = k+1+ my_rank * per_thread;
        int i_end   = i_start + per_thread;
        if(my_rank == t-1){i_end = n;}
        double* a_exp, *u_exp;
        double* l_k = l[k];
        double* u_k = u[k];
        for(int i = i_start; i < i_end; i++){
            double *a_i = a[i];
            for(int j = k+1; j < n; j++){ 
                a_i[j] = a_i[j] - l_k[j]*u_k[i];
            }
        }
    }

    return;


}

/* Verification Functions */

void mat_mult(double** m1, double** m2, double** m3, int n, int t){
#pragma omp parallel for num_threads(t)
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

void mat_sub(double** m1, double** m2, int n, int t){
#pragma omp parallel for num_threads(t)
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

clang++ -std=c++11 -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp omp_LU.cpp -o exec2.out

*/