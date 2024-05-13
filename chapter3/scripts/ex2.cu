#include <stdlib.h>
#include <stdio.h>

#define N 8192

// simple matmul for squared matrices
__global__ void matvec(float* C, float* A, float* B, int n) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if ((col < n)) {
        float outValue = 0;
        for (int k = 0; k < n; ++k) {
            outValue +=  A[col*n + k]*B[k];
        }
        C[col] += outValue;
    }
}

void matmul(float* C_h, float* A_h, float* B_h, int n) {
    int A_size = n * n * sizeof(float);
    int B_size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaMalloc((void**) &A_d, A_size);
    cudaMalloc((void**) &B_d, B_size);
    cudaMalloc((void**) &C_d, B_size);

    cudaMemcpy(A_d, A_h, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, B_size, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(n/4.),1, 1);
    dim3 dimBlock(4,1,1);
    matvec<<<dimGrid, dimBlock>>>(C_d, A_d, B_d, n);

    cudaMemcpy(C_h, C_d, B_size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

bool compare_vectors(float* v, float* u) {
    for (int i = 0; i < N; i++) {
        if (v[i] != u[i]) {
            printf("Error: Mismatch at index %d.\n", i);
            return false;
        }
    }
    return true;
}

void print_vector(float* vector, int n) { 
    for (int i = 0; i < n; i++) {
        if (i == 0) {
            printf("| ");
        }
        printf("%f ", vector[i]);
        if (i == (n-1)) {
            printf("|");
        }
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
    B_h = (float*)malloc(sizeof(float) * N);
    C_h = (float*)malloc(sizeof(float) * N);
    expectedResult = (float*)malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A_h[j + N*i] = 2.0f;
        }
        B_h[i] = 1.0f;
        expectedResult[i] = N * 2.0f;
    }

    matmul(C_h, A_h, B_h, N);
    compare_vectors(C_h, expectedResult) ? printf("Success!\n") : printf("Results do not match.\n");
    // printf("A: ");
    // print_matrix(A_h, N);
    // printf("\nB: ");
    // print_vector(B_h, N);
    // printf("\n\nExpected result: ");
    // print_vector(expectedResult, N);
    // printf("\n\nC: ");
    // print_vector(C_h, N);

}
