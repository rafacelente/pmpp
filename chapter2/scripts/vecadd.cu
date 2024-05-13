#include <stdlib.h>
#include <stdio.h>

#define N 100000000

__global__ void vector_add_kernel(float* out, float* a, float* b, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

void vector_add(float* C_h, float* A_h, float* B_h, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaMalloc((void**) &A_d, size);
    cudaMalloc((void**) &B_d, size);
    cudaMalloc((void**) &C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    vector_add_kernel<<<ceil(n/1024.0), 1024>>>(C_d, A_d, B_d, n);
    
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    float *a, *b, *out;

    a = (float*)malloc(sizeof(float) * N);
    b = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    vector_add(out, a, b, N);
    printf("a[0] = %f\n", a[0]);
    printf("b[0] = %f\n", b[0]);
    printf("out[0] = %f\n", out[0]);
    free(a);
    free(b);
    free(out);
}