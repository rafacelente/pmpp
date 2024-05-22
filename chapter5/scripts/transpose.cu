#include <stdio.h>
#include <stdlib.h>

#define BLOCK_WIDTH 2

__global__ void BlockTranspose(float* A_elements, int A_width, int A_height) {
    __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

    int rows = blockIdx.x * BLOCK_WIDTH + threadIdx.x;
    int cols = blockIdx.y * BLOCK_WIDTH + threadIdx.y;

    blockA[threadIdx.y][threadIdx.x] = A_elements[rows * A_width + cols];

    int transposedX = blockIdx.y * BLOCK_WIDTH + threadIdx.x;
    int transposedY = blockIdx.x * BLOCK_WIDTH + threadIdx.y;

    A_elements[transposedY * A_width + transposedX] = blockA[threadIdx.x][threadIdx.y];
}


int main() {

    int A_width = 8;
    int A_height = 8;

    float* A_h = (float*)malloc(A_width * A_height * sizeof(float));
    float* A_d;

    for (int i = 0; i < A_width * A_height; i++) {
        A_h[i] = i;
    }

    printf("Original matrix:\n");
    for (int i = 0; i < A_width; i++) {
        for (int j = 0; j < A_height; j++) {
            printf("%f ", A_h[i * A_width + j]);
        }
        printf("\n");
    }
    printf("\n");

    cudaMalloc((void**)&A_d, A_width * A_height * sizeof(float));
    cudaMemcpy(A_d, A_h, A_width * A_height * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 gridDim(A_width/blockDim.x, A_height/blockDim.y);
    BlockTranspose<<<gridDim, blockDim>>>(A_d, A_width, A_height);

    cudaMemcpy(A_h, A_d, A_width * A_height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(A_d);

    for (int i = 0; i < A_width; i++) {
        for (int j = 0; j < A_height; j++) {
            printf("%f ", A_h[i * A_width + j]);
        }
        printf("\n");
    }

    free(A_h);
    return 0;
}