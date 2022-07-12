#include <stdio.h>
#include <cuda_runtime.h>
 
void printDeviceProp(const cudaDeviceProp &prop)
{
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %zu.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %zu.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %zu.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %zu.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate);
    printf("textureAlignment : %zu.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}
 
bool InitCUDA()
{
    //used to count the device numbers
    int count;
 
    // get the cuda device count
    cudaGetDeviceCount(&count);
    if (count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }
 
    // find the device >= 1.X
    int i;
    for (i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 1) {
                printDeviceProp(prop);
                break;
            }
        }
    }
 
    // if can't find the device
    if (i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }
 
    // set cuda device
    cudaSetDevice(i);
 
    return true;
}
 
int main(int argc, char const *argv[])
{
    if (InitCUDA()) {
        printf("CUDA initialized.\n");
    }
 
    return 0;
}