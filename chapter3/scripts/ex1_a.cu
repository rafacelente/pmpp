#include <stdlib.h>
#include <stdio.h>

#define N 4096

__global__ void matmul_row_kernel(float* C, float* A, float* B, int n, int which_col) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if ((row < n)) {
        float outValue = 0;
        for (int k = 0; k < n; ++k) {
            outValue +=  A[row*n + k]*B[which_col + n*k];
        }
        C[row * n + which_col] = outValue;
    }
}


void matmul(float* C_h, float* A_h, float* B_h, int n) {
    int size = n * n * sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaMalloc((void**) &A_d, size);
    cudaMalloc((void**) &B_d, size);
    cudaMalloc((void**) &C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(1, ceil(n/4.), 1);
    dim3 dimBlock(1,4,1);
    for (int i = 0; i < n; i++) {
        matmul_row_kernel<<<dimGrid, dimBlock>>>(C_d, A_d, B_d, n, i);
    }
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
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

    matmul(C_h, A_h, B_h, N);
    // print_matrix(A_h, N);
    // print_matrix(B_h, N);
    // print_matrix(expectedResult, N);
    // printf("\n");
    // print_matrix(C_h, N);

}
