#include "NeuralNetwork.cuh"

#include <corecrt_math_defines.h>

#include <stdio.h>

__device__ NeuralNetworkBase::~NeuralNetworkBase()
{
	//delete[] mLayerSizes;
	//delete[] mWeights;
	//delete[] mOldWeights;
}

__device__ void NeuralNetworkBase::Init(int LayerCount, const int* LayerSizes, int WeightCount, float MaxDiff, curandState* RandState)
{
	mLayerCount = LayerCount;
	mLayerSizes = new int[mLayerCount];
	for (int i = 0; i < mLayerCount; ++i)
		mLayerSizes[i] = LayerSizes[i];

	mWeightCount = WeightCount;
	if (mWeightCount == 0)		//In case I wouldn't pre-calculate it
		for (int i = 1; i < mLayerCount; ++i)
			mWeightCount += mLayerSizes[i] * (mLayerSizes[i - 1] + 1);

	mWeights = new float[mWeightCount];

	for (int i = 0; i < mWeightCount; ++i)
		mWeights[i] = RandState ? (curand_uniform(RandState) - 0.5f) * 2.0f * MaxDiff : 0.f;
}

__device__ void NeuralNetworkBase::Init(int LayerCount, const int *LayerSizes, int WeightCount, float *Weights)
{
	mLayerCount = LayerCount;
	mLayerSizes = new int[mLayerCount];
	for (int i = 0; i < mLayerCount; ++i)
		mLayerSizes[i] = LayerSizes[i];

	mWeightCount = WeightCount;
	mWeights = new float[mWeightCount];

	for (int i = 0; i < mWeightCount; ++i)
		mWeights[i] = Weights[i];
}

__device__ void NeuralNetworkBase::CopyWeights(float *const To) const
{
	for (int i = 0; i < mWeightCount; ++i)
		To[i] = mWeights[i];
}

__device__ void NeuralNetworkBase::Mutate(const float* From, float MaxDiff, curandState* RandState)
{
	for (int i = 0; i < mWeightCount; ++i)
		mWeights[i] = From[i] + (RandState ? (curand_uniform(RandState) - 0.5f) * 2.0f * MaxDiff : 0.f);
}

#define PRINTI(variable) printf("%s: %i\n", #variable, variable)
#define PRINTF(variable) printf("%s: %f\n", #variable, variable)

__device__ void NeuralNetwork::Init(const NeuralNetworkBase& Base)
{
	mLayerCount = Base.mLayerCount;
	mLayerSizes = Base.mLayerSizes;

	mWeights = new float* [mLayerCount - 1];
	int start = 0;
	for (int i = 0; i < mLayerCount - 1; ++i)
	{
		mWeights[i] = &Base.mWeights[start];
		start += (mLayerSizes[i] + 1) * (mLayerSizes[i + 1]);
	}

	mNeurodes = new float* [mLayerCount];
	for (int i = 1; i < mLayerCount - 1; ++i)
	{
		//Input and Output layers will come from parameters
		//The "1" node needs no memory
		mNeurodes[i] = new float[mLayerSizes[i]];
	}
}

__device__ void NeuralNetwork::Reset(const NeuralNetworkBase& Base)
{
	int start = 0;
	for (int i = 0; i < mLayerCount - 1; ++i)
	{
		mWeights[i] = &Base.mWeights[start];
		start += (mLayerSizes[i] + 1) * (mLayerSizes[i + 1]);
	}
}

#if false
__global__ void calculateLayer(float* From, int FromSize, float* To, float* AxonMatrix)
{
	unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	float* axonRow = &AxonMatrix[idx * (FromSize + 1)];

	float sum = 1 * axonRow[FromSize];
	for (int k = 0; k < FromSize; ++k)
		sum += From[k] * axonRow[k];

	//constexpr float K = 1.f;
	To[idx] = tanhf(sum);//(1.f - 2.f / (1.f + expf(-K * sum)));

	/*unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	float *axonRow = &AxonMatrix[idx * (blockDim.x + 1)];
	float sum = 1 * axonRow[FromSize];
	for (int i = 0; i < FromSize; ++i)
		sum += From[i] * axonRow[i];

	//constexpr float K = 1.f;
	To[idx] = tanhf(sum);//(1.f - 2.f / (1.f + expf(-K * sum)));*/
}
#endif

__device__ void NeuralNetwork::Calculate(float* Inputs, float* Outputs)
{
	//ToDo() - Fentebbről leosztani kernelekbe
	mNeurodes[0] = Inputs;
	//PRINTF(mNeurodes[0][0]);
	mNeurodes[mLayerCount - 1] = Outputs;
	for (int i = 1; i < mLayerCount; ++i)
	{
		//calculateLayer <<<1, mLayerSizes[i]>>> (mNeurodes[i - 1], mLayerSizes[i - 1], mNeurodes[i], mWeights[i - 1]);
		//PRINTI(i);
		//PRINTI(mLayerSizes[i]);

		for (int j = 0; j < mLayerSizes[i]; ++j)
		{
			//PRINTI(j);
			//PRINTI(j * (mLayerSizes[i - 1] + 1));
			float* axonRow = &mWeights[i - 1][j * (mLayerSizes[i - 1] + 1)];
			//printf("This point in the code\n");
			//PRINTI(mLayerSizes[i - 1]);
			//PRINTF(axonRow[mLayerSizes[i - 1]]);
			float sum = 1 * axonRow[mLayerSizes[i - 1]];
			//PRINTF(sum);
			for (int k = 0; k < mLayerSizes[i - 1]; ++k)
			{
				//PRINTI(k);
				sum += mNeurodes[i - 1][k] * axonRow[k];
				//PRINTF(sum);
			}

			//constexpr float K = 1.f;
			mNeurodes[i][j] = tanhf(sum) * M_2_PI;//(1.f - 2.f / (1.f + expf(-K * sum)));
			//PRINTF(mNeurodes[i][j]);
		}
	}
}