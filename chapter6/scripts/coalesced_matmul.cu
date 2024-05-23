#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>


#define N 4096
#define TILE_WIDTH 32

__global__ void coalesced_matmul_kernel(
    float* C,
    float* A,
    float* B, // matrix with column-major layout 
    int width
) {

    //in this example, A is stored in row-major layout and
    // B is stored in column-major layout
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = blockIdx.x; int ty = blockIdx.y;

    int col = bx * TILE_WIDTH + tx;
    int row = by * TILE_WIDTH + ty;

    float Cvalue = 0;
    for (int stride_idx = 0; i < width/TILE_WIDTH; ++stride_idx) {
        Ads[ty][tx] = A[row * width + (stride_idx * TILE_WIDTH) + tx];
        Bds[ty][tx] = B[col * width + (stride_idx * TILE_WIDTH) + ty];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += Ads[ty][k] * Bds[k][tx];
        }
        __syncthreads();
    }
    C[row * width + col] = Cvalue;
}

void coalesced_matmul(float* C_h, float* A_h, float* B_h, int width) {
    int size = width * width * sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(width/TILE_WIDTH), ceil(width/TILE_WIDTH), 1);
    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);

    coalesced_matmul_kernel<<<dimGrid, dimBlock>>>(C_d, A_d, B_d, width);

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    float *A_h, *B_h, *C_h, *expectedResult;
    printf("Running example with N = %d\n", N);

    A_h = (float*)malloc(sizeof(float) * N * N);
    B_h = (float*)malloc(sizeof(float) * N * N);
    C_h = (float*)malloc(sizeof(float) * N * N);
    expectedResult = (float*)malloc(sizeof(float) * N * N);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A_h[j + N*i] = 2.0f;
            B_h[i + N*j] = 1.0f;
            expectedResult[j + N*i] = N * 2.0f;
        }
    }

    tiled_squared_matmul(C_h, A_h, B_h, N);
}