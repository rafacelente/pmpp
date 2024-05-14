# 1

Consider the following kernel: 

```c
__global__ void foo_kernel(int* a, int* b) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < 40 || threadIdx.x >= 104) {
        b[i] = a[i] + 1;
    }
    if (i % 2 == 0) {
        a[i] = b[i] * 2;
    }
    for (unsigned int j = 0; j < 5 - (i%3); j++) {
        b[i] += j;
    }
}



void foo(int* a_d, int* b_d) {
    unsigned int N = 1024;
    foo_kernel<<<(N+128-1)/128, 128>>>(a_d, b_d);
}
```

### a. What is the number of warps per block?

Considering a warp size of 32, since we have 128 threads per block, we would have 4 warps per block.

### b. What is the number of warps per grid?

Since `N=1024` and we have `int((N+128-1)/128) = 8` blocks per grid, we will have `8 * 4 = 32` warps per grid.

### c. For the statement on line 04:

#### i. How many warps in the grid are active?

For each block we have 4 warps, each with 32 threads. For a warp to be considered active during an instruction, at least one of the threads within that warp have to execute that instruction. That means that, for a single block, for a warp to be active or partially active, at least of the threads in threads 0 to 31 (Warp 0), 32 to 63 (Warp 1), 64 to 95 (Warp 2) and 96 to 127 (Warp 3) have to active.

On line 04, we are executing all threads on Warp 0, partially executing threads of Warp 1, and partially executing threads of Warp 3. Therefore, we have 3 active warps per block.

With 8 blocks per grid, we have a total of 24 warps active per grid.


#### ii. How many warps in the grid are divergent?

The definition of non-divergent warps is when all threads execute the same instruction within that warp.

On line 04, we have that at any given moment only warps 0 and 3 are executing always the same instruction. Therefore, only `2 * 8 = 16` warps are active per grid.

#### iii. What is the SIMD efficiency (in %) of warp 0 of block 0?

On Warp 0, all threads are active. Therefore the SIMD efficiency is 100%.

#### iv. What is the SIMD efficiency (in %) of warp 1 of block 0?

On Warp 1, threads [32 ... 39] are active, while [40 ... 63] are not. Therefore, the SIMD efficiency of this warp $8/32 = 25\%$

#### v. What is the SIMD efficiency (in %) of warp 3 of block 0?

On Warp 3, threads [96 ... 103] are inactive, while [104 ... 127] are active. Therefore, the SIMD efficiency of this warp $24/32 = 75\%$

### d. For the statement on line 07:

#### i. How many warps in the grid are active?

Since inside each warp at least one warp is executed, all warps in the grid are active.

#### ii. How many warps in the grid are divergent?

Since inside each warp you always have 2 consecutive threads that execute different instructions, all warps in the grid are divergent.

#### iii. What is the SIMD efficiency (%) of warp 0 of block 0?

Half of the threads are active while the other half is inactive, so the SIMD efficiency is $50\%$.

### e. For the loop on line 09:

#### i. How many iterations have no divergence?

In each block we have 128 threads. To compute the iteration cycle, we have to do `[5 - (i % 3) for i in range(0,128)]`, which is a reocurring sequence of `[5,4,3]`. That means that one third of our code will execute 5 times, one third will execute 4 times and the last third will execute 3 times. Since it's the same instruction everytime, we are sure that for at least 3 for loops our code will have no divergence. However, the fourth and fifth loops diverge, since some threads within our warps will execute while others won't. Therefore, our code has no divergence for 3 iterations, and divergence for 2 iterations.

# 2

For a vector addition, assume that the vector length is 2000, each thread calculates one output element, and the thread block size is 512 threads. How many threads will be in the grid?

Response:

You would have 4 blocks of 512 threads, which equates to 2048 threads per grid.

# 3

Since our vector length is 2000 and we have 2048 threads, 1 warp will have divergence (Warp 62, threads 1984 - 2015).

# 4

All threads that are finished before the barrier will stay idle until the last thread is finished. Since our longest thread has an execution time of 3.0 microseconds, the total idle time for all threads is `total_idle_time = (3-torch.Tensor([2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9])).sum() = 4.1`. Our total execution time is `3 * 8 = 24`, therefore our percentage of time spent waiting for the barrier is `4.1/24 = 17.08%`.


# 5

Even with blocks of 32 threads, `__syncthreads()` is still necessary because threads within a warp can have different sets of instructions and read/write times. Depending on the application, the ommition of `__syncthreads()` can lead to race conditions.

# 6

With 512 threads per block, we would have exactly 3 blocks per SM, giving us a total of exactly 1536 threads. Therefore, c is the correct answer.

# 7

a. Possible. Occupancy: 50%.
b. Possible. Occupancy: 50%.
b. Possible. Occupancy: 50%.
b. Possible. Occupancy: 100%.
b. Not possible.

# 8

a. With 128 threads per block, we would need 16 blocks to achieve full occupancy. With 30 registers per thread, we would have `30 * 16 * 128 = 61440` registers in total, so full occupancy is possible.

b. With 32 threads per block, we would need 64 blocks to achieve full occupancy. However, we are only allowed 32 blocks per SM, so we can't achieve full occupancy.

c. With 256 threads per block, we would need 8 blocks to achieve full occupancy. With 34 registers per thread, we would have `34 * 8 * 256 = 69632` registers in total, so the we can't assign this case.

# 9

The student may have been mistaken, since that with 32 x 32 thread blocks, he would not be able to launch a kernel on a device with 512 maximum threads per block.


