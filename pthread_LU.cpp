#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <chrono>
#include <cstring>
#include <iostream>
using namespace std;

#define SWAP(T, a, b) do {T tmp = a; a = b; b = tmp; } while (0)

/* Sequential Implementation of the LU decomposition as step 1 of the parallelisation procedure */

void LU(int n, int t); /* Wrapper function for the decomposition */
void malloc_all_matrices();
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

void *thread_lu_set(void *rank);
void *thread_a_set(void *rank);

//functions for parallelisation
pthread_mutex_t mutex;
void* thread_max(void* rank);

//global variables for pthread usageint
int n;
int total_threads;
double** a;
double** l;
double** u;
double** pa;
double** a_prime;
int* pi;
int k; //used in the major for loop
double gmax = 0.0;
int k_prime; //used in the interchange

int main(int argc, char* argv[]) {

    printf("Hello.\n");

    n = strtol(argv[1], NULL, 10); // User input
    total_threads = strtol(argv[2], NULL, 10); // User input

    printf("Arguments Read.\n");

    LU(n, total_threads); // Pass control to the LU-decomposer

    return 0;
}   

void LU(int n, int thread_count){
    pthread_t* thread_handles; //creating thread handles, ig can be reused all throughout
    // thread_handles = (pthread_t*) malloc(thread_count*sizeof(pthread_t));
    thread_handles = new pthread_t[thread_count];
    int thread_ids[thread_count];
    for (int i = 0; i < thread_count; i++) {
        thread_ids[i] = i;
    }

    //double *a[n], *l[n], *u[n], *pa[n], *a_prime[n]; // Pointers to the matrice's rows
    //int pi[n];
    /* Here, the vector pi is a compact representation of a permutation matrix p(n,n), 
       which is very sparse. For the ith row of p, pi(i) stores the column index of
       the sole position that contains a 1.*/

    // below are now made global
    /*
    for (int i = 0; i < n; i++){
        a[i] = (double*)malloc(n * sizeof(double));
        l[i] = (double*)malloc(n * sizeof(double));
        u[i] = (double*)malloc(n * sizeof(double));
        pa[i] = (double*)malloc(n * sizeof(double));
        a_prime[i] = (double*)malloc(n * sizeof(double));
    }
    */
    malloc_all_matrices(); //initialises the global matrices

    init_a(n, a);
    init_l(n, l);
    init_u(n, u);
    init_pi(n, pi);
    init_pa(n, a, pa);

    printf("Matrices Initalised.\n");

    double* swp = new double[n];
    
    //printf("Matrix A at the start = \n");
    //mat_print(a, n);

    std::chrono::microseconds t_max(0);
    std::chrono::microseconds t_swap(0);
    std::chrono::microseconds t_lu(0);
    std::chrono::microseconds t_a(0);

    clock_t t_clock; 
    t_clock = clock();
    pthread_mutex_init(&mutex, NULL); //do outside for loop
    auto start = std::chrono::high_resolution_clock::now();

    // k is the column here
    for(k = 0; k < n; k++){  
        // sequential gmax implementation
        gmax = 0.0;
        
        // paralle gmax implementation
        // gmax = 0;
        int thread;
        auto t1 = std::chrono::high_resolution_clock::now();
        for(int i = k ; i < n; i++){
            if (gmax < fabs(a[i][k])){
                gmax = fabs(a[i][k]);
                k_prime = i;
            }
        }
        // for (thread = 0; thread < thread_count; thread++)
        //     pthread_create(&thread_handles[thread], NULL, thread_max, (void*)&thread_ids[thread]);
        // for (thread = 0; thread < thread_count; thread++)
        //     pthread_join(thread_handles[thread], NULL);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
        t_max += duration;

        
        if (gmax == 0.0) {
            printf("Error: Singular Matrix");
            return;
        }

        // Now that pivot has been discovered, start swapping
        t1 = std::chrono::high_resolution_clock::now();
        // Below not parallelised
        SWAP(int, pi[k], pi[k_prime]);
        double* temp = a[k];
        a[k] = a[k_prime];
        a[k_prime] = temp;

        // Swap rows of l
        memcpy(swp, l[k], sizeof(double) * k);
        memcpy(l[k], l[k_prime], sizeof(double) * k);
        memcpy(l[k_prime], swp, sizeof(double) * k);
        u[k][k] = a[k][k];

        t2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
        t_swap += duration;
        
        t1 = std::chrono::high_resolution_clock::now();

        for (thread = 0; thread < thread_count; thread++)
            pthread_create(&thread_handles[thread], NULL, thread_lu_set, (void*)&thread_ids[thread]);
        for (thread = 0; thread < thread_count; thread++)
            pthread_join(thread_handles[thread], NULL);
        t2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
        t_lu += duration;

        //sequential a setting implementation
        
        t1 = std::chrono::high_resolution_clock::now();
        //parallel a setting implementation
        
        for (thread = 0; thread < thread_count; thread++)
            pthread_create(&thread_handles[thread], NULL, thread_a_set, (void*)&thread_ids[thread]);
        for (thread = 0; thread < thread_count; thread++)
            pthread_join(thread_handles[thread], NULL);
        t2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
        t_a += duration;

        // Complete for the this iteration of k
    }

    pthread_mutex_destroy(&mutex);
    // LU Computed. Now testing for time, and L2,1 convergence.
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    cout << "Time taken by the program: " << chrono::duration_cast<chrono::milliseconds>(duration).count() << " ms" << endl;

    // t_clock = clock() - t_clock; 
    // double time_taken = ((double)t_clock)/CLOCKS_PER_SEC; // in seconds 
    // printf("LU for n = %d took? %f seconds to execute \n", n, time_taken);
    cout << "Time taken for gmax: " << chrono::duration_cast<chrono::milliseconds>(t_max).count() << " ms" << endl;
    cout << "Time taken for swap: " << chrono::duration_cast<chrono::milliseconds>(t_swap).count() << " ms" << endl;
    cout << "Time taken for lu updates: " << chrono::duration_cast<chrono::milliseconds>(t_lu).count() << " ms" << endl;
    cout << "Time taken for a updates: " << chrono::duration_cast<chrono::milliseconds>(t_a).count() << " ms" << endl;

    // Now use permutation matrix to permute rows of A
    mat_rearrange(pa, pi, n);
    mat_mult(l, u, a_prime, n);  // a_prime = LU
    mat_sub(pa, a_prime, n);      // pa = PA - LU

    double convg_error = L2c1(pa, n);
    printf("LU Convergence error for n = %d is %f  \n", n, convg_error);

    free(thread_handles);

    return;
}

void malloc_all_matrices() {
    a = (double**) malloc(n * sizeof(double*));
    l = (double**) malloc(n * sizeof(double*));
    u = (double**) malloc(n * sizeof(double*));
    pa = (double**) malloc(n * sizeof(double*));
    a_prime = (double**) malloc(n * sizeof(double*));

    pi = (int*) malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        a[i] = (double*)malloc(n * sizeof(double));
        l[i] = (double*)malloc(n * sizeof(double));
        u[i] = (double*)malloc(n * sizeof(double));
        pa[i] = (double*)malloc(n * sizeof(double));
        a_prime[i] = (double*)malloc(n * sizeof(double));
    }
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

void* thread_max(void* rank) {
    int t = *((int*)rank);
    int total_i = n - k;
    int i_d = total_i/total_threads;
    int i_r = total_i % total_threads;

    int first_i, last_i;
    if (t < i_r)
        first_i = (i_d*t) + t;
    else
        first_i = (i_d*t) + i_r;

    if (t+1 < i_r)
        last_i = (i_d * (t+1)) + t+1;
    else
        last_i = (i_d * (t+1)) + i_r;


    double my_max = 0;
    int my_k_prime = -1;
    for (int j = k+first_i; j < k+last_i; j++) {
        double a_jk = fabs(a[j][k]);
        if (my_max < a_jk) {
            my_max = a_jk;
            my_k_prime = j;
        }
    }

    pthread_mutex_lock(&mutex); // how to specify where the lock has been put?
    if (my_max > gmax) {
        gmax = my_max;
        k_prime = my_k_prime;
    }
    pthread_mutex_unlock(&mutex);
}

void* thread_swap_a(void* rank) {
    int t = *((int*)rank);
    int j_d = n/total_threads;
    int j_r = n % total_threads;

    int first_j, last_j;
    if (t < j_r)
        first_j = (j_d*t) + t;
    else
        first_j = (j_d*t) + j_r;

    if (t+1 < j_r)
        last_j = (j_d * (t+1)) + t+1;
    else
        last_j = (j_d * (t+1)) + j_r;

    for (int jj = first_j; jj < last_j; jj++) {
        SWAP(double, a[k][jj], a[k_prime][jj]);
    }
}

void* thread_swap_l(void* rank) {
    int t = *((int*)rank);
    int j_d = k/total_threads;
    int j_r = k % total_threads;

    int first_j, last_j;
    if (t < j_r)
        first_j = (j_d*t) + t;
    else
        first_j = (j_d*t) + j_r;

    if (t+1 < j_r)
        last_j = (j_d * (t+1)) + t+1;
    else
        last_j = (j_d * (t+1)) + j_r;

    for (int jj = first_j; jj < last_j; jj++) {
        SWAP(double, l[k][jj], l[k_prime][jj]);
    }
}

void* thread_lu_set(void* rank) {
    int t = *((int*)rank);
    int total_i = n - k - 1;
    int i_d = total_i/total_threads;
    int i_r = total_i % total_threads;

    int first_i, last_i;
    if (t < i_r)
        first_i = (i_d*t) + t;
    else
        first_i = (i_d*t) + i_r;

    if (t+1 < i_r)
        last_i = (i_d * (t+1)) + t+1;
    else
        last_i = (i_d * (t+1)) + i_r;

    double u_kk = u[k][k];
    double *a_k = a[k];
    double *u_k = u[k];
    for (int j = k+1+first_i; j < k+1+last_i; j++) {
        l[j][k] = a[j][k]/u_kk;
        u_k[j] = a_k[j];
    }
}

void* thread_a_set(void* rank) {
    int t = *((int*)rank);
    int total_i = n - k - 1;
    int i_d = total_i/total_threads;
    int i_r = total_i % total_threads;

    int first_i, last_i;
    if (t < i_r)
        first_i = (i_d*t) + t;
    else
        first_i = (i_d*t) + i_r;

    if (t+1 < i_r)
        last_i = (i_d * (t+1)) + t+1;
    else
        last_i = (i_d * (t+1)) + i_r;

    double l_ik;
    double *a_i;
    double *u_k = u[k];
    for (int i = k+1+first_i; i < k+1+last_i; i++) {
        l_ik = l[i][k];
        a_i = a[i];
        for (int j = k+1; j < n; j++) {
            a_i[j] = a_i[j] - l_ik*u_k[j];
        }
    }
}

/* 

Gurarmaan

Execute using command

clang -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp seq_LU.c -o exec.out

*/
