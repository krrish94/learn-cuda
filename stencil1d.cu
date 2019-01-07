// 1D stencil example using CUDA C++

#include <iostream>


// Global Parameters
#define NUMBLOCKS 8
#define BLOCKSIZE 4
#define RADIUS 1
#define NUMELEMENTS (NUMBLOCKS * BLOCKSIZE)


// Function and macro to handle CUDA errors
static void handleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		std::cout << cudaGetErrorString(err) << " in " << file << " at  line " << line << std::endl;
		exit(EXIT_FAILURE);
	}
}
#define cudaCheck(err) (handleError(err, __FILE__, __LINE__))


// A 1D stencil reads in a block of an array, and adds up all elements within a stencil of fixed raduis 
// and writes them to an output vector
__global__ void stencil1d(int *in, int *out) {

	__shared__ int temp[BLOCKSIZE + 2*RADIUS];

	int gindex = threadIdx.x + blockIdx.x * blockDim.x + RADIUS;
	int lindex = threadIdx.x + RADIUS;

	// Read in data corresponding to the actual block elements
	temp[lindex] = in[gindex];
	// Read in boundary-data ('halo' on either side, with length commensurate to the radius)
	if (threadIdx.x < RADIUS) {
		// Left halo
		temp[lindex - RADIUS] = in[gindex - RADIUS];
		// Right halo
		temp[lindex + BLOCKSIZE] = in[gindex + BLOCKSIZE];
	}

	// Prevent WAR/RAW/WAW conflicts
	__syncthreads();

	// Apply the stencil
	int result = 0;
	for (int offset = -RADIUS; offset <= RADIUS; ++offset) {
		result += temp[lindex + offset];
	}

	// Store the result
	out[gindex-RADIUS] = result;

}


int main(void) {

	// Initialize host copies of in, out
	int in[NUMELEMENTS + 2*RADIUS], out[NUMELEMENTS];
	for (int i = 0; i < NUMELEMENTS + 2*RADIUS; ++i) {
		if (i < RADIUS) {
			in[i] = 0;
		} else if (i < NUMELEMENTS + RADIUS) {
			in[i] = 1;
		} else {
			in[i] = 0;
		}
	}

	// // Verify input by printing
	// for (int i = 0; i < NUMELEMENTS + 2*RADIUS; ++i) {
	// 	std::cout << in[i] << " ";
	// }
	// std::cout << std::endl;

	// Sizes
	int size_in = (NUMELEMENTS + 2*RADIUS) * sizeof(int);
	int size_out = NUMELEMENTS * sizeof(int);

	// Initialize device copies of in, out
	int *d_in, *d_out;
	cudaCheck(cudaMalloc((void **)&d_in, size_in));
	cudaCheck(cudaMalloc((void **)&d_out, size_out));

	// Copy variables from host to device
	cudaCheck(cudaMemcpy(d_in, in, size_in, cudaMemcpyHostToDevice));

	// Launch the kernel
	stencil1d<<<NUMBLOCKS, BLOCKSIZE>>>(d_in, d_out);

	// Check for kernel launch errors
	cudaCheck(cudaPeekAtLastError());

	// Copy variables from device to host
	cudaCheck(cudaMemcpy(out, d_out, size_out, cudaMemcpyDeviceToHost));

	// Print the result
	for (int i = 0; i < NUMELEMENTS; ++i) {
		std::cout << out[i] << " ";
	}
	std::cout << std::endl;

}
