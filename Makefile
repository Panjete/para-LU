CC = g++
CFLAGS = -g -Wall

default: pth.out omp.out

pth.out: pthread_LU.cpp
	$(CC) $(CFLAGS) pthread/pthread_LU.cpp -lpthread -o pthread/pth.out

omp.out: openmp_LU.cpp
	$(CC) $(CFLAGS) openmp/openmp_LU.cpp -fopenmp -o openmp/omp.out

clean:
	rm -f pthread/pth.out openmp/omp.out
