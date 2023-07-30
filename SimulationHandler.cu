#include "SimulationHandler.h"
#include "GlobalSettings.h"
#include "Colony.cuh"
#include "Ant.cuh"
#include <chrono>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <corecrt_math_defines.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>

__device__ inline float2 rotate(float2 In, float angle)
{
	float sina = sinf(angle);
	float cosa = cosf(angle);
	return { (In.x * cosa - In.y * sina), (In.x * sina + In.y * cosa) };
}

__global__ void InitColonies(Colony *Colonies, Ant *Ants, ColonyState *ColonyStates, AntState *AntStates)
{
	//ToDo: Not to do math, but instead use these IDs to identify what to do where - Less math, more dimensions
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stateId = (idx * GS::gColonyCount) % GS::gAllColonyCount + idx / GS::gEnvironmentCount; //First the even, then the odd numbers
	//unsigned int stateId = (idx % 2) * GS::gEnvironmentCount + idx / 2;

	float diffAngle = 2 * M_PI / GSettings::gColonyCount;
	float startAngle = diffAngle / 2;
	float2 basePosition = { 0.0f, GSettings::gMapSizeX / 4.0f };

	ColonyStates[stateId].mPosition = rotate(basePosition, diffAngle * (stateId % GS::gColonyCount) + startAngle);

	Colonies[idx].Init(
		&Ants[idx * GS::gAntCount],	//Ants
		&ColonyStates[stateId], //State
		&AntStates[stateId * GS::gAntCount],	//AntState
		&Colonies[(idx + GS::gEnvironmentCount) % GS::gAllColonyCount]);	//OtherColony
}

//ToDo: Might be unnecessary, could be merged with Init
__global__ void ResetColonies(Colony *Colonies, Ant *Ants, ColonyState *ColonyStates, AntState *AntStates)
{
	//ToDo: Not do math, but use these IDs to identify what to do where - Less math, more dimensions
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stateId = (idx * GS::gColonyCount) % GS::gAllColonyCount + idx / GS::gEnvironmentCount; //First the even, then the odd numbers
	//unsigned int stateId = (idx % 2) * GS::gEnvironmentCount + idx / 2;

	float diffAngle = 2 * M_PI / GSettings::gColonyCount;
	float startAngle = diffAngle / 2;
	float2 basePosition = { 0.0f, GSettings::gMapSizeX / 4.0f };

	ColonyStates[stateId].mPosition = rotate(basePosition, diffAngle * (stateId % GS::gColonyCount) + startAngle);

	Colonies[idx].Reset(
		&Ants[idx * GS::gAntCount],	//Ants
		&ColonyStates[stateId], //State
		&AntStates[stateId * GS::gAntCount],	//AntState
		&Colonies[(idx + GS::gEnvironmentCount) % GS::gAllColonyCount]);	//OtherColony
}

__global__ void SetupWeights(Colony *Colonies, curandState *RandState, float *Weights = nullptr, int Gen = 0)
{
	unsigned long long offset = GS::gWeightCount * GS::gEnvironmentCount * (Gen + 1ULL);
	curand_init(RAND_SEED, 0ULL, offset, RandState);

	if (!Weights)
	{
		for (int i = 0; i < GS::gEnvironmentCount; ++i)
			Colonies[i].mHiveMind.Init(GS::gLayerCount, GS::gLayers, GS::gWeightCount);

		for (int i = GS::gEnvironmentCount; i < GS::gAllColonyCount; ++i)
			Colonies[i].mHiveMind.Init(GS::gLayerCount, GS::gLayers, GS::gWeightCount, GS::gWeightStartRange, RandState);
	}
	else
	{
		for (int i = 0; i < GS::gEnvironmentCount; ++i)
			Colonies[i].mHiveMind.Init(GS::gLayerCount, GS::gLayers, GS::gWeightCount, &Weights[0]);

		for (int i = GS::gEnvironmentCount; i < GS::gAllColonyCount; ++i)
			Colonies[i].mHiveMind.Init(GS::gLayerCount, GS::gLayers, GS::gWeightCount, &Weights[(i - GS::gEnvironmentCount + 1) * GS::gWeightCount]);
	}
}

__global__ void InitAnts(Ant *Ants, Colony *Colonies, AntState *AntStates)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int colonyId = idx / GS::gAntCount;
	unsigned int colonyStateId = (colonyId * GS::gColonyCount) % GS::gAllColonyCount + colonyId / GS::gEnvironmentCount;
	unsigned int antStateId = colonyStateId * GS::gAntCount + idx % GS::gAntCount;
	Ants[idx].Init(&Colonies[colonyId], &AntStates[antStateId]);
}

//ToDo: Might be unnecessary, could be merged with Init
__global__ void ResetAnts(Ant *Ants, Colony *Colonies, AntState *AntStates)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int colonyId = idx / GS::gAntCount;
	unsigned int colonyStateId = (colonyId * GS::gColonyCount) % GS::gAllColonyCount + colonyId / GS::gEnvironmentCount;
	unsigned int antStateId = colonyStateId * GS::gAntCount + idx % GS::gAntCount;
	Ants[idx].Reset(&Colonies[colonyId], &AntStates[antStateId]);
}

constexpr int threadPerBlock = 128;
constexpr int blockCount = (GS::gEnvironmentCount * GS::gAntCount - 1) / threadPerBlock + 1;
//static_assert(threadPerBlock * blockCount == GS::gEnvironmentCount * GS::gAntCount, "Incorrect threadPerBlock value");

__global__ void AntSteps(Ant *Ants, const int StepCount, const unsigned int From = 0)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < GS::gEnvironmentCount * GS::gAntCount)
		Ants[From + idx].Step(StepCount);
}

void SimulationHandler::AttackerStep()	//ToDo: Merge with DefenderStep
{
	//constexpr int threadPerBlock = 500;//GS::gAllColonyCount;
	//constexpr int blockCount = (GS::gEnvironmentCount * GS::gAntCount - 1) / threadPerBlock + 1; //gColonyCount not needed, because it's called separately for every Colony team.
	AntSteps <<<blockCount, threadPerBlock>>> (mAnts, mStepCounter, GS::gEnvironmentCount * GS::gAntCount);	//kernel call could be simpler, but it might be more understandable this way.
}

void SimulationHandler::DefenderStep()	//ToDo: Merge with AttackerStep
{
	//constexpr int threadPerBlock = 500;//GS::gAllColonyCount;
	//constexpr int blockCount = (GS::gEnvironmentCount * GS::gAntCount - 1) / threadPerBlock + 1; //gColonyCount not needed, because it's called separately for every Colony team.
	AntSteps<<<blockCount, threadPerBlock>>>(mAnts, mStepCounter);	//kernel call could be simpler, but it might be more understandable this way.
}

__global__ void SumWeights(Colony *Colonies, int Amount, float *SumWeights)
{
	//printf("Amount: %i\n", Amount);
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = 0; i < Amount; ++i)
		SumWeights[idx] += Colonies[i].mHiveMind.mWeights[idx];
	SumWeights[idx] /= Amount;
}

__global__ void Evaluate_Device(Colony *Colonies, curandState *RandState)
{
	Colony *Start = &Colonies[GS::gEnvironmentCount];
	Colony *End = &Colonies[GS::gAllColonyCount];
	thrust::device_ptr<Colony> dtp_Colonies(Start);
	thrust::device_ptr<Colony> end_ptr(End);
	thrust::sort(dtp_Colonies, end_ptr, thrust::greater<Colony>());
	int firstNonWinner = 0;
	int highestPointAtt = 0;
	int highestPointDef = 0;
	float biggestWeight = 0.0f;
	float biggestChange = 0.0f;
	int pointGetters = 0;

	for (int i = 0; i < GS::gEnvironmentCount; ++i)
		if (highestPointDef < Colonies[i].mState->mPoints)
			highestPointDef = Colonies[i].mState->mPoints;

	for (int i = 0; i < GS::gEnvironmentCount; ++i)
	{
		if (Start[i].mState->mPoints > 0)
			++pointGetters;
		if (highestPointAtt < Start[i].mState->mPoints)
			highestPointAtt = Start[i].mState->mPoints;
		if (Start[i].mIsWinner && i < GS::gEnvironmentCount / 3) //ToDo: /3 is an arbitary number to force mutation even when everyone wins.
			firstNonWinner = i + 1;
	}

	//Calculate average weights of the loser and winner teams
	const int &weightCount = Colonies[0].mHiveMind.mWeightCount;
	float *avgWinnerWeights = new float[weightCount]();
	if (firstNonWinner != 0)
		SumWeights<<<1, weightCount>>>(Start, firstNonWinner, avgWinnerWeights);	//ToDo: Maybe use Losers as negative weight?
	cudaDeviceSynchronize();	//ToDo: Deprecated to call on device
	//SumWeights<<<1, weightCount>>>(Start, GS::gEnvironmentCount - firstNonWinner, avgLoserWeights);

	for (int i = 0; i < weightCount; ++i)
	{
		if (firstNonWinner == 0)
			avgWinnerWeights[i] = Colonies[0].mHiveMind.mWeights[i];
		if (abs(biggestChange) < abs(avgWinnerWeights[i] - Colonies[0].mHiveMind.mWeights[i]))
			biggestChange = avgWinnerWeights[i] - Colonies[0].mHiveMind.mWeights[i];
		if (abs(biggestWeight) < abs(avgWinnerWeights[i]))
			biggestWeight = avgWinnerWeights[i];
	}
	printf("Winners - losers: %i - %i\n", firstNonWinner, GS::gEnvironmentCount - firstNonWinner);
	printf("Highest points: %i - %i\n", highestPointDef, highestPointAtt);
	printf("Number of point getters: %i\n", pointGetters);
	printf("Biggest weight / change: %f - %f\n", biggestWeight, biggestChange);

	//Set the new defender team to the average of the winners
	//ToDo: make it parallel if necessary for efficiency, since the defenders need no randomization
	for (int i = 0; i < GS::gEnvironmentCount; ++i)
		Colonies[i].mHiveMind.Mutate(avgWinnerWeights);

	for (int i = GS::gEnvironmentCount + 1; i < GS::gAllColonyCount; ++i)
		Colonies[i].mHiveMind.Mutate(avgWinnerWeights, GS::gWeightDiffRange, RandState);

	delete[] avgWinnerWeights;
}

void SimulationHandler::Evaluate()
{
	Evaluate_Device<<<1, 1>>>(mColonies, mRandState);
	CUDA_CHECK;
}

void SimulationHandler::FreeMemory()
{
	cudaFree(mColonies);
	cudaFree(mAnts);
	cudaFree(mDevice_ColonyStates);
	cudaFree(mDevice_AntStates);
	cudaFreeHost(mHost_ColonyStates);
	cudaFreeHost(mHost_AntStates);
}

void SimulationHandler::GenerateStartingState(bool LoadState)
{
	//ToDo: Reset mHost_WeightTransfer to null?
	static std::chrono::time_point<std::chrono::steady_clock> startTime;
	const auto endTime = std::chrono::steady_clock::now();

	if (!LoadState)
		++mGeneration;

	//Calculate the time of one generation
	if (startTime.time_since_epoch() == std::chrono::seconds(0))
		printf("\nGEN #%i\n", mGeneration);
	else
		printf("\nGEN #%i - %lld ms\n", mGeneration, std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count());

	if (mGeneration == 0 || LoadState)
	{
		cudaMalloc(&mColonies, sizeof(Colony) * GSettings::gAllColonyCount);
		cudaMalloc(&mAnts, sizeof(Ant) * GSettings::gAllAntCount);
		cudaMalloc(&mDevice_ColonyStates, sizeof(ColonyState) * GSettings::gAllColonyCount);
		cudaMalloc(&mDevice_AntStates, sizeof(AntState) * GSettings::gAllAntCount);
		cudaMalloc(&mRandState, sizeof(curandState));
		cudaMallocHost(&mHost_ColonyStates, sizeof(ColonyState) * GSettings::gAllColonyCount);
		cudaMallocHost(&mHost_AntStates, sizeof(AntState) * GSettings::gAllAntCount);

		InitColonies<<<1, GS::gAllColonyCount>>>(mColonies, mAnts, mDevice_ColonyStates, mDevice_AntStates);
		SetupWeights<<<1, 1>>>(mColonies, mRandState, LoadState ? mHost_WeightTransfer : nullptr, mGeneration);
		InitAnts<<<GS::gColonyCount * GS::gAntCount, GS::gEnvironmentCount>>>(mAnts, mColonies, mDevice_AntStates);
		CUDA_CHECK;
	}
	else
	{
		ResetColonies<<<1, GS::gAllColonyCount>>>(mColonies, mAnts, mDevice_ColonyStates, mDevice_AntStates);
		//The weights are already set up in Evaluation
		//SetupWeights << <1, 1 >> > (mColonies, mRandState);
		ResetAnts<<<GS::gColonyCount * GS::gAntCount, GS::gEnvironmentCount>>>(mAnts, mColonies, mDevice_AntStates);
	}

	if ((mGeneration % GS::gAutoSavePeriod == 0) && !LoadState)
		SaveState();

	startTime = std::chrono::steady_clock::now();
}

__global__ void CopyDataToHost(const Colony *const Colonies, float *const Weights)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int colonyId = idx + (!idx ? 0 : (GS::gEnvironmentCount - 1));
	Colonies[colonyId].mHiveMind.CopyWeights(Weights + idx * GS::gWeightCount);
}

void SimulationHandler::SaveDeviceToHost() const 
{
	CopyDataToHost<<<1, GS::gEnvironmentCount + 1 >>>(mColonies, mDevice_WeightTransfer);
	cudaMemcpy(mHost_WeightTransfer, mDevice_WeightTransfer, sizeof(float) * GS::gWeightCount * (GS::gEnvironmentCount + 1), cudaMemcpyDeviceToHost);
}