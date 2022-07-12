#include <stdio.h>
#include <cuda_runtime.h>

//第一个const应该是限制数组内容不可以被改变，第二个const应该是限制指针变量无法改变
__global__ void addVectorKernel(int *const c, const int *const a, const int *const b)
{
    const unsigned int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = {1, 2, 3, 4, 5};
    const int b[arraySize] = {1, 2, 4, 4, 5};
    int c[arraySize] = {0};

    int *dev_a;
    int *dev_b;
    int *dev_c;

    cudaMalloc((void **)&dev_a, arraySize * sizeof(int));
    cudaMalloc((void **)&dev_b, arraySize * sizeof(int));
    cudaMalloc((void **)&dev_c, arraySize * sizeof(int));

    cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    addVectorKernel<<<1, arraySize>>>(dev_c, dev_a, dev_b);

    cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < arraySize; i++)
    {
        printf("%d ", c[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}