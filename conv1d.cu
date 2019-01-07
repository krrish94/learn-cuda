// 1D convolution example using CUDA C++
// Each block takes in a bunch of elements and computes a 1D convolution using multiple threads

#include <iostream>


// Global parameters
#define NUMBLOCKS 8
#define BLOCKSIZE 4
#define RADIUS 1
#define NUMELEMENTS (NUMBLOCKS * BLOCKSIZE)


// Function and macro to handle CUDA errors
static void handleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		std::cout << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
		exit(EXIT_FAILURE);
	}
}
#define cudaCheck(err) (handleError(err, __FILE__, __LINE__))


// 1D convolution kernel
__global__ void conv1d(float *in, float *out) {

	__shared__ float temp[BLOCKSIZE + 2*RADIUS];

	int gindex = threadIdx.x + blockIdx.x * blockDim.x + RADIUS;
	int lindex = threadIdx.x + RADIUS;

	// Read in data corresponding to the input elements to be processed
	temp[lindex] = in[gindex];
	// Read in boundary data ('halo' on either side of the current filters primary context)
	if (threadIdx.x < RADIUS) {
		// Left side of 'halo'
		temp[lindex - RADIUS] = in[gindex - RADIUS];
		// Right side of 'halo'
		temp[lindex + BLOCKSIZE] = in[gindex + BLOCKSIZE];
	}

	// Ensure thread-safety
	__syncthreads();

	// Perform convolution
	float result = 0.0;
	for (int offset = -RADIUS; offset <= RADIUS; ++offset) {
		result += temp[lindex + offset];
	}
	result = result;

	// Store the result
	out[gindex - RADIUS] = result;

}


int main(void) {

	// Initialize host copies of in, out
	float in[NUMELEMENTS + 2*RADIUS], out[NUMELEMENTS];
	// Zero pad on either side
	for (int i = 0; i < NUMELEMENTS + 2*RADIUS; ++i) {
		if (i < RADIUS) {
			in[i] = 0.0;
		} else if (i < NUMELEMENTS + RADIUS) {
			in[i] = 1.0;
		} else {
			in[i] = 0.0;
		}
	}

	// Sizes
	int size_in = (NUMELEMENTS + 2*RADIUS) * sizeof(float);
	int size_out = NUMELEMENTS * sizeof(float);

	// Initialize device copies
	float *d_in, *d_out;
	cudaCheck(cudaMalloc((void **)&d_in, size_in));
	cudaCheck(cudaMalloc((void **)&d_out, size_out));

	// Copy variables from host to device
	cudaCheck(cudaMemcpy(d_in, in, size_in, cudaMemcpyHostToDevice));

	// Launch the conv1d kernel
	conv1d<<<NUMBLOCKS, BLOCKSIZE>>>(d_in, d_out);

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
