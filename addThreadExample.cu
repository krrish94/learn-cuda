#include <iostream>


__global__ void add(int *a, int *b, int *c) {
	c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}


#define M 512

int main(void) {

	// host copies of a, b, c
	int *a, *b, *c;
	// device copies of a, b, c
	int *d_a, *d_b, *d_c;

	int size = M * sizeof(int);

	// Initialize host copies
	a = (int *) malloc(size);
	b = (int *) malloc(size);
	c = (int *) malloc(size);

	for(int i = 0; i < M; ++i) {
		a[i] = 2;
		b[i] = 7;
	}

	// Initialize device copies
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// Copy variables from host to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// Launch the kernel
	add<<<1,M>>>(d_a, d_b, d_c);

	// Copy data from device to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	// Print
	for(int i = 0; i < M; ++i) {
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

}
