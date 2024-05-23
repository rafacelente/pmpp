# 1

```cpp
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
```


# 2

To avoid completly uncoalesced access to global memory, the block size should take into consideration the distribution of warps. The most favorable access pattern is achieved when all threads in a warp access consecutive global memory locations. Therefore, to completly avoid uncoalesced access, the the block size should be a multiple of 32 so that all warps within the block access consecutive memory locations. Therefore, for a square block BLOCK_SIZE x BLOCK_SIZE, we require that `BLOCK_SIZE * BLOCK_SIZE % 32 == 0`.

# 3

```cpp
__global__ void foo_kernel(float* a, float* b, float* c, float* d, float* e) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ float a_s[256];
    __shared__ float bc_s[4*256];
    a_s[threadIdx.x] = a[i];

    for (unsigned int j = 0; j < 4; ++j) {
        bc_s[j*256 + threadIdx.x] = b[j*blockDim.x * gridDim.x + i] + c[i*4 + j];
    }
    __syncthread();

    d[i+8] = a_s[threadIdx.x];
    e[i*8] = bc_s[threadIdx.x * 4];
}

```

a. Coalesced.
b. Not applicable (shared memory).
c. Coalesced.
d. Not coalesced.
e. Not applicable (shared memory).
f. Not applicable (shared memory).
g. Coalesced.
h. Not applicable (shared memory).
i. Not coalesced.

# 4

### a. 

```cpp
__global__ void matmul(float* C, float* A, float* B, int width) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < width && col < width) {
        float Cvalue = 0;
        for (int i = 0; i < width; ++i) {
            C_value += A[row * width + k] * B[k * width + col];
        } 
        // width^2 operations
        // width^2 global reads, width^2 global writes
        C[row * width + col] = Cvalue;
    }
}
```

Let's consider, for simplicity, the matrix multiplication of the matrices $A$ and $B$, both with dimensions $NxN$. 

In the simple matmul kernel, initially each thread will see the first element of each row of A (or column, in the case of B). Then, in the for loop, it will run over all elements of that same row (or column, in the case of B), fetching its value from global memory and computing the dot product operation. Therefore, for each row of A and column of B, a thread has to read 2 N values from global memory, compute N multiplications and N additions, and then write 1 value to global memory.

Therefore, we do $N$ rows multiplied by $N$ columns multiplied by $2 \cdot N$ operations, giving us a total of $2 \cdot N^3$ operations. Meanwhile, we also do $N$ rows multiplied by $N$ columns multiplied by ($2N$ floating point reads and 1 floating point write), giving us as well a total of $2 \cdot N^3 + 1$ of global memory accesses.

Since each floating point has 8 bytes, we have an arithmetic intensity of 0.125 OP/B.

$$
AI = \frac{2 N^3}{\frac{2 N^2 (8 N) + 1}} \approx 0.125
$$

### b. 

```cpp
__global__ void matmul(float* C, float* A, float* B, int width) {
    __shared__ A_s[BLOCK_WIDTH][BLOCK_WIDTH];
    __shared__ B_s[BLOCK_WIDTH][BLOCK_WIDTH];

    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;

    int row = by * BLOCK_WIDTH + ty;
    int col = bx * BLOCK_WIDTH + tx;

    float Cvalue = 0;
    for (int stride = 0; stride < width/BLOCK_WIDTH; ++stride) {
        if (row < width && BLOCK_WIDTH * stride + tx < width) {
            A_s[ty][tx] = A[row * width + tx + stride * BLOCK_WIDTH];
        }
        else {
            A_s[ty][tx] = 0.0;
        }

        if (col < width && BLOCK_WIDTH * stride + ty < width) {
            B_s[ty][tx] = B[(stride * BLOCK_WIDTH + ty) * width + col];
        }
        else {
            B_s[ty][tx] = 0.0;
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_WIDTH; ++k) {
            Cvalue += A_s[ty][k] * B_s[k][tx];
        }
        __syncthreads();
    }
    C[row * width + col] += Cvalue;
```

In the tiled matmul kernel we still have the same ammount of floating point operations: $N$ rows multiplied by $N$ columns multiplied by $2 \cdot N$ operations, giving us a total of $2 \cdot N^3$ operations.

However, now that for each tile our memory is cached on the shared memory for every tile stride, we reduce significantly the number of global memory accesses. We still do $2N$ global reads and 1 global write per thread, but now that operation doesn't have to repeat for each element in the matrix. Since we're caching the values in each block width $B$, we only have to access these values from global memory every $N/B$ times. Therefore, we have the following expression for our arithmetic intensity:

$$
AI = \frac{2 N^3}{2 (8N)\frac{N^2}{B^2}} = \frac{N^2}{8}
$$


For a block size $B$ of 32, we would have an arithmetic intensity of 128 OP/B.

### c.

If we use a thread coarsening of a factor $C$, each thread will essentially compute $C$ times more values than previously. However, the memory access will also access $C$ more values from global memory. Therefore, the arithmetic intensity will still be the same as the tiled matrix multiplication (128 OP/B).

Note: even though the arithmetic intensity is the same, it doesn't mean that it is not more performant. We're comparing operations per byte and not operations per second, so this metric doesn't answer the question if these bytes are accessed sequentially or in parallel.