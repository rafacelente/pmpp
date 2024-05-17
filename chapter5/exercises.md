# 1

Matrix addition is an element-wise operation, which means each element is only used once per operation. Therefore, using shared memory won't give us any overhead.

# 2

# 3

If we forget the first `__syncthreads()`, some threads may start the dot product computation before loading all values into the shared memory. Therefore, the dot product might be computed with unitialized or previous-iteration variables.

If we forget the the second `__syncthreads()`, we may find a situation where the dot product for a thread is finished and it continues the loop, loading the next tile into the shared memory, disturbing the computation of other threads that hadn't yet finished their computation.

# 4

In CUDA, registers may be an even faster type of memory than shared memory, these types have very different localities. While shared memory is (like its name) shared between threads in a block, registers are local to a thread. This gives a huge advantage to shared memories when the computation is done in such a way that multiple threads re-use the same memory, like in matrix multiplication.

# 5

It can be proven that using a NxN tiled implementation of the matrix multiplication has reduces the global memory access to 1/N over the naive implementation. Therefore, we have a reduction of 31/32 in memory bandwidth usage.

# 6

A local variable in a kernel is a private attribute of the thread that executes that kernel, therefore, for 1000 blocks with 512 threads each, we'd have 512000 versions of that variable created over the lifetime of the execution of the kernel.

# 7

Variables declared on the shared memory are shared between the threads of the same block. Therefore, we'd have 1000 versions of the variable declared over the lifetime of the execution of the kernel.

# 8

### a. No tiling

In a naive matrix multiplication implementation, each element of the matrix is used N times, therefore there are N global memory accesses.

### b. TxT tiling

# 9

The kernel is memory-bound if the time to access the memory per thread is larger than the time to compute all operations in one thread. In our specific thread, we are computing 36 floating-point operations and accessing 28 B (seven 32-bit) of memory per thread.

## a. Peak FLOPS=200 GFLOPS, peak memory bandwidth = 100 GB/seconds

- Time to compute a thread at peak FLOPS: $t_c = \frac{36}{200 * 10^{9}} = 1.8 \cdot 10^{-8}$.
- Time taken to access the memory per thread: $t_m = \frac{28}{100* 10^{9}} = 2.8 \cdot 10^{-8} s$.

Since $t_m > t_c$, this kernel is memory-bound.

## a. Peak FLOPS=300 GFLOPS, peak memory bandwidth = 250 GB/seconds

- Time to compute a thread at peak FLOPS: $t_c = \frac{36}{300 * 10^{9}} = 1.2 \cdot 10^{-8}$.
- Time taken to access the memory per thread: $t_m = \frac{28}{250* 10^{9}} = 1.12 \cdot 10^{-8} s$.

Since $t_c > t_m$, this kernel is compute-bound.

# 10



