#include <iostream>


__global__ void add(int *a, int *b, int *c) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	c[index] = a[index] + b[index];
}


#define N (2048 * 2048)
#define THREADS_PER_BLOCK 512


int main(void) {

	// host copies of a, b, c
	int *a, *b, *c;
	// device copies of a, b, c
	int *d_a, *d_b, *d_c;

	int size = N * sizeof(int);

	// Alloc space and setup
	a = (int *) malloc(size);
	b = (int *) malloc(size);
	c = (int *) malloc(size);

	for(int i = 0; i < N; ++i) {
		a[i] = 2;
		b[i] = 7;
	}

	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU
	add<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	// Cleanup
	free(a);
	free(b);
	free(c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;

}
