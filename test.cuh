#include <cuda_runtime_api.h>
#include <curand_kernel.h>

class TestClass
{
public:
	float *testVal;
	__device__ void Init();
};

void kernel(TestClass *testClass);