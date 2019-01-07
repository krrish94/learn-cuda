#include <iostream>


__global__ void add(int *a, int *b, int *c) {
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}


int main(void) {

	// Number of blocks
	int N = 512;

	// host copies of a, b, c
	int *a, *b, *c;
	// device copies of a, b, c
	int *d_a, *d_b, *d_c;

	int size = N * sizeof(int);

	// Allocate memory to host copies
	a = (int *) malloc(size);
	b = (int *) malloc(size);
	c = (int *) malloc(size);

	// Initialize vectors
	for (int i = 0; i < N; ++i) {
		a[i] = 2;
		b[i] = 7;
	}

	// Allocate memory to device copies
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// Copy inputs from host to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// Launch add() kernel of GPU
	add<<<N,1>>>(d_a, d_b, d_c);

	// Copy result from device to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	// Print
	for(int i = 0; i < N; ++i) {
		std::cout << c[i] << " ";
	}
	std::cout << std::endl;

	// Cleanup
	free(a);
	free(b);
	free(c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;

}
