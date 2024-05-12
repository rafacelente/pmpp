# Chapter 2

## 1

If we want to use each thread in a grid to calculate one output element of a vector addition, what would be the expression for mapping the thread block indices to the data index (i)?

1. i=threadIdx.x + threadIdx.y;
2. i=blockIdx.x + threadIdx.x;
3. i=blockIdx.x * blockDim.x + threadIdx.x;
4. i=blockIdx.x * threadIdx.x;

Response: For computing a one dimensional vector, considering a one-dimensional grid and one-dimensional block, we have to split the vector into the N blocks of of size `blockDim`. Therefore, to access an element at index `i`, we have to stride by block and thread, so the correct answer is **C**.

## 2

Assume that we want to use each thread to calculate two adjacent elements of a vector addition. What would be the expression for mapping the thread/block indices to the data index (i) of the first element to be processed by a thread?

1. i=blockIdx.x * blockDim.x + threadIdx.x +2;
2. i=blockIdx.x * threadIdx * 2;
3. i=(blockIdx.x * blockDim.x + threadIdx.x) * 2;
4. i=blockIdx.x * blockDim.x * 2 + threadIdx.x;

Response: In the case we have two adjecent elements, we have to multiple the block stride by 2. Therefore, the correct answer is **D**.

## 3

We want to use each thread to calculate two elements of a vector addition. Each thread block processes `2 * blockDim.x` consecutive elements that form two sections. All threads in each block will process a section first, each processing one element. They will then all move to the next section, each processing one element. Assume that variable i should be the index for the first element to be processed by a thread. What would be the expression for mapping the thread/block indices to data index of the first element?

1. i=blockIdx.x * blockDim.x + threadIdx.x +2;
2. i=blockIdx.x * threadIdx.x * 2;
3. i=(blockIdx.x * blockDim.x + threadIdx.x) * 2;
4. i=blockIdx.x * blockDim.x * 2 + threadIdx.x;

Response: In this case, we are interleaving the elements of two vectors and performing the addition on these strided elements. Therefore, the mapping of consecutive elements in memory is doubled. Element 0 is followed by 2, which is followed by 4, and 6 and so on. Taking into consideration the block striding, we arrive at the answer **C**.

## 4

For a vector addition, assume that the vector length is 8000, each thread calculates one output element, and the thread block size is 1024 threads. The programmer configures the kernel call to have a minimum number of thread blocks to cover all output elements. How many threads will be in the grid?
1. 8000
2. 8196
3. 8192
4. 8200

Response: To compute a vector of length 8000 with blocks of size 1024, we have to find the smallest number of blocks that contain enough threads to fit 8000 elements. In this case, the smallest integer `n` that fits this case is `n=8` because 7 * 1024 = 7168 and 8 * 1024 = 8192. Therefore, the correct answer is **C**.


## 5

If we want to allocate an array of v integer elements in the CUDA device global memory, what would be an appropriate expression for the second argument of the cudaMalloc call?
1. n
2. v
3. n * sizeof(int)
4. v * sizeof(int)

Response: We want to allocate a number `v` of integers. Each integer is of size `sizeof(int)`. Therefore, the correct answer is **D**.

## 6

If we want to allocate an array of n floating-point elements and have a floating-point pointer variable A_d to point to the allocated memory, what would be an appropriate expression for the first argument of the `cudaMalloc()` call?
1. n
2. (void* ) A_d
3. *A_d
4. (void** ) &A_d

Response: The first argument of the `cudaMalloc` function is a pointer to the pointer with the address of the variable you want to store in device memory. By definition, this has to be a void double pointer, which is exactly represented in option **D**.

## 7

If we want to copy 3000 bytes of data from host array A_h (A_h is a pointer to element 0 of the source array) to device array A_d (A_d is a pointer to element 0 of the destination array), what would be an appropriate API call for this data copy in CUDA?
1. cudaMemcpy(3000, A_h, A_d, cudaMemcpyHostToDevice);
2. cudaMemcpy(A_h, A_d, 3000, cudaMemcpyDeviceTHost);
3. cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);
4. cudaMemcpy(3000, A_d, A_h, cudaMemcpyHostToDevice);

Response: The `cudaMemcpy` API has the following definition: `cudaMemcpy(destination, source, size, kind)`. Therefore, to copy 3000 bytes of data from A_h on host to A_d on device, we have to call option **C**.

## 8

How would one declare a variable err that can appropriately receive the returned value of a CUDA API call?
1. int err;
2. cudaError err;
3. cudaError_t err;
4. cudaSuccess_t err;

Response: Errors in CUDA are handled by the enum `cudaError_t`, like in the option **C**.

## 9

Consider the following CUDA kernel and the corresponding host function
that calls it:
01

10
```
__global__ void foo_kernel(float * a, float * b, unsigned int N){
    unsigned int i=blockIdx.x * blockDim.x + threadIdx.
    if(i < N) {
        b[i]=2.7f * a[i] - 4.3f;
    }
}
void foo(float * a_d, float * b_d) {
    unsigned int N=200000;
    foo_kernel <<<(N + 128 - 1)/128, 128>>>(a_d,b_d, N);
}
```
1. What is the number of threads per block?
2. What is the number of threads in the grid?
3. What is the number of blocks in the grid?
4. What is the number of threads that execute the code on line 02?
5. What is the number of threads that execute the code on line 04?

Response:

1. The number of threads per block is 128.
2. We have `int((200000 + 128 -1)/128) = 1563` blocks of 128 threads, therefore our grid has `1563 * 128 = 200064` threads.
3. 1563 as calculated above.
4. All threads execute line 2.
5. Only 200000 of the 200064 threads execute line 4 because of the if statement.

## 10

A new summer intern was frustrated with CUD1. He has been complaining that CUDA is very tedious. He had to declare many functions that he plans to execute on both the host and the device twice, once as a host function and once as a device function. What is your response?

Response: CUDA provides the `__host__` and `__device__`decorators that can be used together for functions that are executed both on host and device.

```
__host__ __device__ void myFunction(int x) {
    // ... implementation that works both on host and device
}
```
