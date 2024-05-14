#include <stdlib.h>
#include <stdio.h>

int main() {
    cudaDeviceProp devProp;
    int devCount;
    cudaGetDeviceCount(&devCount);
    for (unsigned int i = 0; i < devCount; i++) {
        cudaGetDeviceProperties(&devProp, i);
        printf("Device number %d: \n", i);
        printf("    Device name: %s\n", devProp.name);\
        printf("    Memory Clock Rate (MHz): %d\n",
           devProp.memoryClockRate/1024);
        printf("    Warp-size: %d\n", devProp.warpSize);
        printf("    Max threads per block: %d\n", devProp.maxThreadsPerBlock);
        printf("    Max blocks per SM: %d\n", devProp.maxBlocksPerMultiProcessor);
        printf("    Number of SMs: %d\n", devProp.multiProcessorCount);
        printf("    Max threads per block: %d\n", devProp.maxThreadsPerBlock);
        printf("    Max thread dimensions per block: (%d, %d, %d)\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
        printf("    Registry size per block: %d\n", devProp.regsPerBlock);
        printf("    L2 cache size: %d\n", devProp.l2CacheSize);
        printf("    Shared memory per block: %lu\n", devProp.sharedMemPerBlock);
        printf("    Global memory size: %lu\n", devProp.totalGlobalMem);
        printf("    Constant memory size: %lu\n", devProp.totalConstMem);
        printf("    Max grid dimensions: (%d, %d, %d)\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
        printf("    Max threads per multiprocessor: %d\n", devProp.maxThreadsPerMultiProcessor);
        printf("    Max shared memory per multiprocessor: %lu\n", devProp.sharedMemPerMultiprocessor);
        printf("    Max threads per multiprocessor: %d\n", devProp.maxThreadsPerMultiProcessor);
    }
}