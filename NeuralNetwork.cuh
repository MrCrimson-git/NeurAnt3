#pragma once

#include <cuda_runtime_api.h>
#include <curand_kernel.h>

class NeuralNetworkBase
{
	friend class NeuralNetwork;
public:
	int mLayerCount;
	int* mLayerSizes;
	float* mWeights;
	int mWeightCount;

public:
	__device__ ~NeuralNetworkBase();
	__device__ void Init(int LayerCount, const int *LayerSizes, int WeightCount = 0, float MaxDiff = 0.0f, curandState* RandState = NULL);
	__device__ void Init(int LayerCount, const int *LayerSizes, int WeightCount, float *Weights);
	__device__ void Cleanup();
	__device__ void CopyWeights(float *const To) const;
	__device__ void Mutate(const float *From, float MaxDiff = 0.0f, curandState* RandState = NULL);
};

class NeuralNetwork
{
public:
	int mLayerCount;
	int *mLayerSizes;
	float **mWeights;
	float **mNeurodes;

public:
	__device__ void Init(const NeuralNetworkBase &Base);
	__device__ void Cleanup();
	__device__ void Reset(const NeuralNetworkBase &Base);

	__device__ void Calculate(float *Inputs, float *Outputs);
};