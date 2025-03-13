
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <random>
#include <ctime>


__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < Width) && (col < Width)) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row * Width + k] * N[Width * k + col];
        }
        P[row * Width + col] = Pvalue;
    }
}

void printMatrix(const char* name, float* matrix, int width) {
    printf("%s:\n", name);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f\t", matrix[i * width + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Fill a matrix with random integer values
void fillRandomMatrix(float* matrix, int numofelements) {
    for (int i = 0; i < numofelements; i++) {
        // Generate random values between 1 and 10
        matrix[i] = (float)(rand() % 10 + 1);
    }
}

int main() {
    const int Width = 3; // assume its a 3*3 matrix
    int numofelements = Width * Width;
    int size = Width * Width * sizeof(float);

    // Allocate memory for matrices on CPU
    float* h_M = (float*)malloc(size);   // First input matrix
    float* h_N = (float*)malloc(size);   // Second input matrix
    float* h_P = (float*)malloc(size);   // Result matrix

    // Initialize random seed
    srand(time(NULL));

    // Fill matrices with random values
    fillRandomMatrix(h_M, numofelements);
    fillRandomMatrix(h_N, numofelements);

    // Allocate memory for matrices on GPU
    float* d_M, * d_N, * d_P;
    cudaMalloc(&d_M, size);
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_P, size);

    // Copy input matrices from CPU to GPU
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    // Configure thread layout (9 threads in a 3x3 block)
    dim3 blockSize(3, 3);  // 3x3 = 9 threads per block
    dim3 gridSize(1, 1);   // Only need 1 block for a 3x3 matrix

    // Run the kernel on the GPU
    MatrixMulKernel <<<gridSize, blockSize >> > (d_M, d_N, d_P, Width);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

   
    // Copy the result back from GPU to CPU
    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    // Display the matrices
    printMatrix("Matrix M", h_M, Width);
    printMatrix("Matrix N", h_N, Width);
    printMatrix("Result P = M × N", h_P, Width);

    // Clean up
    free(h_M);
    free(h_N);
    free(h_P);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}