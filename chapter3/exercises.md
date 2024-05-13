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

Script on `chapter3/scripts/ex2.cu`.

# 3

Consider the following kernel:

```c
__global__ void foo_kernel(float* a, float* b, unsigned int M, unsigned int N) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y = threadIdx.y;

    if (row < M && col < N) {
        b[row * N + col] = a[row * N + col]/2.1f + 4.8f;
    }
}
void foo(float* a_d, float* b_d) {
    unsigned int M = 150;
    unsigned int N = 300;
    dim3 bd(16,32);
    dim3 gd((N-1)/16 + 1, (M-1)/32 + 1);
    foo_kernel<<<gd, bd>>>(a_d, b_d, M, N);
}

```

### a. What is the number of threads per block?

The block dimension is `(16,32)`, which means that we have `16 * 32 = 512` threads per block.

### b. What is the number of threads per grid?

The grid dimension is `(19, 5)`, which means we have `19 * 5 = 95` blocks per grid. Therefore we have `95 * 512 = 48640` threads per grid.

### c. What is the number of blocks in the grid?

`95`, as answered above.

### d. What is the number of threads that execute the code on line 05?

Our total thread count on the row axis is `16 * 19 = 304`, while on the column axis is `32 * 5 = 160`, which means the total number of threads that will be seen by our kernel is `304 * 160 = 48640`. However, on line 05 we are only processing the threads that fall under the constraints of the matrix we want, which is `300x150`. Therefore, of the 48640 threads, only 45000 will be processed by line 05.

# 4

Consider a 2D matrix with a width of 400 and a height of 500. The matrix is stored as a one-dimensional array. Specify the array index of the matrix element at row 20 and column 10.

## If the matrix is stored in row-major order

If $A$ is stored in row-major order, an element $A_{i,j}$, where $i$ is the i-th row and $j$ is the j-th column, can be mapped to a row-wise vector $P$ such that $A_{i,j} = P_{i\cdot * N + j}$, where $N$ is the number of columns in the matrix $A$. Therefore, for $A$ a `500x400` matrix, we have that $A_{20,10} = P_{20 * 400 + 10} = P_{8010}$.

## If the matrix is stored in column-major order

If $A$ is stored in column-major order, an element $A_{i,j}$, where $i$ is the i-th row and $j$ is the j-th column, can be mapped to a column-wise vector $P$ such that $A_{i,j} = P_{j\cdot * M + i}$, where $M$ is the number of rows in the matrix $A$. Therefore, for $A$ a `500x400` matrix, we have that $A_{20,10} = P_{10 * 500 + 20} = P_{5020}$.

# 5 

For a 3D tensor, we can calulate the row-wise index through $i= z \cdot (width \cdot height) + y \cdot width + x$. Therefore, we have that the array index for `x=10`, `y=20` and `z=5` is equal to 1008010.


