#include "cuda_runtime.h"
#include "./test.cuh"

#ifndef __CUDACC__  //Removes error about blockDim, etc...
#include "device_launch_parameters.h"
#endif

__global__ void vectorAdditionKernel(double* A, double* B, double* C, int arraySize) {
    // Get thread ID.
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if thread is within array bounds.
    if (threadID < arraySize) {
        // Add a and b.
        C[threadID] = A[threadID] + B[threadID];
    }
}

__device__ void TestClass::Init()
{
    testVal = new float;
    *testVal = 5.f;
}

__global__ void realKernel(TestClass *testClass)
{
    testClass->Init();
}



/**
 * Wrapper function for the CUDA kernel function.
 * @param A Array A.
 * @param B Array B.
 * @param C Sum of array elements A and B directly across.
 * @param arraySize Size of arrays A, B, and C.
 */
void kernel(TestClass *testClass)
{
    realKernel<<<1,1>>>(testClass);
}