# Chapter 3

## 1

A standard matmul kernel for squared matrices in which the threads compute each element of the output matrix can be done like this:

```c
// simple matmul for squared matrices
__global__ void matmul_kernel(float* C, float* A, float* B, int n) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if ((col < n) && (row < n)) {
        float outValue = 0;
        for (int k = 0; k < n; ++k) {
            outValue +=  A[row*n + k]*B[col + n*k];
        }
        C[row * n + col] = outValue;
    }
}
```

In this case, we have a 2 dimensional block, fetching both rows and columns to match the shape of our input and output matrices. This setup makes the computation extremely fast because we can parallelize the computation of the all the elements on a given row for `A`, and all the elements on a given column on `B`. The downside of this approach is that it requires a larger number of threads. This is not a problem for most matrix multiplications, but on larger matrices this might not be ideal.

What we can do to minimize the number of threads being used is use the same thread to compute each row or column. This can be achieved by the following kernel:

```c
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
```

Then, in the host function, we can iterate through all columns of `B`.

```c
dim3 dimGrid(1, ceil(n/4.), 1);
dim3 dimBlock(1,4,1);
for (int i = 0; i < n; i++) {
    matmul_row_kernel<<<dimGrid, dimBlock>>>(C_d, A_d, B_d, n, i);
}
```

Compared to the original approach, this method uses $n \cdot (n-1)$ less threads, where $n$ is the block width. However, this method also is a lot slower. Running both methods with the same input matrices of size 4096, we get an execution time of 0.18 seconds for the first approach and 3.02 seconds for the second approach.

# 2

