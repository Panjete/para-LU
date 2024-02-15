CC = g++
CFLAGS = -g -Wall

default: pth.out omp.out

pth.out: pthread_LU.cpp
	$(CC) $(CFLAGS) pthread_LU.cpp -lpthread -o pth.out

omp.out: openmp_LU.cpp
	$(CC) $(CFLAGS) openmp_LU.cpp -fopenmp -o omp.out

clean:
	rm -f pth.out omp.out
