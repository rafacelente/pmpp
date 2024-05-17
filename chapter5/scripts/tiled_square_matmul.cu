#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 8192
#define TILE_WIDTH 32

__global__ void tiled_squared_matmul_kernel(float* C, float* A, float* B, int width) {
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Cvalue = 0;
    for (int stride_idx = 0; stride_idx < width/TILE_WIDTH; stride_idx++) {
        Ads[ty][tx] = A[row * width + stride_idx * TILE_WIDTH + tx];
        Bds[ty][tx] = B[(stride_idx * TILE_WIDTH + ty) * width + col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += Ads[ty][k] * Bds[k][tx];
        }
        __syncthreads();
    }
    C[row * width + col] = Cvalue;
}

void tiled_squared_matmul(float* C_h, float* A_h, float* B_h, int width) {
    int size = width * width * sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(width/TILE_WIDTH), ceil(width/TILE_WIDTH), 1);
    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);

    tiled_squared_matmul_kernel<<<dimGrid, dimBlock>>>(C_d, A_d, B_d, width);

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

bool compare_matrices(float* A, float* B) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (A[i * N + j] != B[i * N + j]) {
            printf("Error: Mismatch at index (%d, %d).\n", i, j);
            return false;
        }
        }
    }
    return true;
}

int min_int(int a, int b) {
    return a < b ? a : b;
}

void calculate_optimal_blocks() {
    int smemSize;
    cudaDeviceGetAttribute(&smemSize, 
        cudaDevAttrMaxSharedMemoryPerBlock, 0);
    int numProcs;
    cudaDeviceGetAttribute(&numProcs,
        cudaDevAttrMultiProcessorCount, 0);
    int maxBlocksPerSM;
    cudaDeviceGetAttribute(&maxBlocksPerSM, 
        cudaDevAttrMaxBlocksPerMultiprocessor, 0);
    int maxSharedMemoryPerMultiprocessor;
    cudaDeviceGetAttribute(&maxSharedMemoryPerMultiprocessor, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
    int maxThreadsPerSM;
    cudaDeviceGetAttribute(&maxThreadsPerSM, cudaDevAttrMaxThreadsPerMultiProcessor, 0);

    printf("OCCUPANCY CALCULATION\n");
    printf("------------------------------------------------\n");
    int threadsPerBlock = TILE_WIDTH * TILE_WIDTH;
    printf("Threads per block: %d\n", threadsPerBlock);
    printf("Warps per block: %d\n", threadsPerBlock / 32);

    int blocksPerSM = min_int(maxThreadsPerSM / threadsPerBlock, maxBlocksPerSM);
    printf("Blocks per SM: %d\n", blocksPerSM);

    int totalThreadsPerSM = threadsPerBlock * blocksPerSM;
    printf("Total threads per SM: %d\n", totalThreadsPerSM);

    printf("Max shared memory per block: %d B\n", smemSize);
    printf("Number of SMs: %d\n", numProcs);
    printf("Max blocks per SM: %d\n", maxBlocksPerSM);
    float occupancy = (float)totalThreadsPerSM / maxThreadsPerSM;
    printf("Occupancy: %.2f\n", occupancy);
    printf("-----------------------------------------------\n");

    // shared memory usage
    int requiredMemoryPerBlock = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);
    printf("Required shared memory per block: %d B\n", requiredMemoryPerBlock);

    int requiredMemoryPerSM = requiredMemoryPerBlock * blocksPerSM;
    printf("Required shared memory per SM: %d B\n", requiredMemoryPerSM);

    printf("Max shared memory per multiprocessor: %d B\n", maxSharedMemoryPerMultiprocessor);

    int memoryConstrainedBlocks = maxSharedMemoryPerMultiprocessor / requiredMemoryPerBlock;
    printf("Memory constrained max blocks: %d\n", memoryConstrainedBlocks);

    if (requiredMemoryPerSM > maxSharedMemoryPerMultiprocessor) {
        printf("ERROR: Not enough shared memory to run this kernel. with %d B needed and %d B available.\n", requiredMemoryPerSM, maxSharedMemoryPerMultiprocessor);
    }
}

void print_matrix(float* matrix, int n) { 
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j == 0) {
                printf("| ");
            }
            printf("%f ", matrix[j + N*i]);
            if (j == (n-1)) {
                printf("|");
            }
        }
        printf("\n");
    }
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
            B_h[j + N*i] = 1.0f;
            expectedResult[j + N*i] = N * 2.0f;
        }
    }

    tiled_squared_matmul(C_h, A_h, B_h, N);
    //print_matrix(C_h, N);
    //printf("\n");
    //print_matrix(expectedResult, N);
    compare_matrices(C_h, expectedResult) ? printf("Success!\n") : printf("Results do not match.\n");
    calculate_optimal_blocks();
}