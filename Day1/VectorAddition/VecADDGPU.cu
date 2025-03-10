
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
 

__global__
void vecAddkernel( float* A,  float* B, float* C, int n) {
	int i = threadIdx.x + blockDim.x + blockIdx.x;
	if (i < n) {
		C[i] = A[i] + B[i];
	}
}

int main() {
	const int n = 10;
	float A[n], B[n], C[n];
	float* A_d, * B_d, * C_d;

	int size = n * sizeof(float);
	
	cudaMalloc(&A_d, size);
	cudaMalloc(&B_d, size);
	cudaMalloc(&C_d, size);

	cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);


	vecAddkernel <<<ceil(n/256.0), 256 >> > (A_d, B_d, C_d, n);

	cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
	return 0;
}