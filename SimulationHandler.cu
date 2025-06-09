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
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stateId = (idx * GS::gColonyCount) % GS::gAllColonyCount + idx / GS::gEnvironmentCount; //First the even, then the odd numbers

	float diffAngle = 2 * M_PI / GSettings::gColonyCount;
	float startAngle = diffAngle / 2;
	float2 basePosition = { 0.0f, GSettings::gMapSizeX / 4.0f };

	ColonyStates[stateId].mPosition = rotate(basePosition, diffAngle * (stateId % GS::gColonyCount) + startAngle);

	Colonies[idx].Init(
		&Ants[idx * GS::gAntCount],	//Ants
		&ColonyStates[stateId], //State
		&Colonies[(idx + GS::gEnvironmentCount) % GS::gAllColonyCount]);	//OtherColony
}

__global__ void InitColonies_Duel(Colony *Colonies, Ant *Ants, ColonyState *ColonyStates, AntState *AntStates)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stateId = (idx * GS::gColonyCount) % 2 + idx;

	float diffAngle = 2 * M_PI / GSettings::gColonyCount;
	float startAngle = diffAngle / 2;
	float2 basePosition = { 0.0f, GSettings::gMapSizeX / 4.0f };

	ColonyStates[stateId].mPosition = rotate(basePosition, diffAngle * (stateId % GS::gColonyCount) + startAngle);

	Colonies[idx].Init(
		&Ants[idx * GS::gAntCount],	//Ants
		&ColonyStates[stateId], //State
		&Colonies[(idx + 1) % 2]);	//OtherColony
}

__global__ void ResetColonies(Colony *Colonies, Ant *Ants, ColonyState *ColonyStates, AntState *AntStates)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stateId = (idx * GS::gColonyCount) % GS::gAllColonyCount + idx / GS::gEnvironmentCount; //First the even, then the odd numbers

	float diffAngle = 2 * M_PI / GSettings::gColonyCount;
	float startAngle = diffAngle / 2;
	float2 basePosition = { 0.0f, GSettings::gMapSizeX / 4.0f };

	ColonyStates[stateId].mPosition = rotate(basePosition, diffAngle * (stateId % GS::gColonyCount) + startAngle);

	Colonies[idx].Reset(
		&Ants[idx * GS::gAntCount],	//Ants
		&ColonyStates[stateId], //State
		&Colonies[(idx + GS::gEnvironmentCount) % GS::gAllColonyCount]);	//OtherColony
}

__global__ void SetupWeights(Colony *Colonies, curandState *RandState, float *Weights = nullptr, int Gen = 0)
{
	unsigned long long offset = GS::gWeightCount * GS::gEnvironmentCount * (Gen + 1); //+1 because the weights have been already generated it just reloads them below
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

__global__ void SetupWeights_Duel(Colony *Colonies, float *Weights)
{
	for (int i = 0; i < 2; ++i)
		Colonies[i].mHiveMind.Init(GS::gLayerCount, GS::gLayers, GS::gWeightCount, &Weights[i * GS::gWeightCount]);
}

__global__ void InitAnts(Ant *Ants, Colony *Colonies, AntState *AntStates)
{
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int colonyId = idx / GS::gAntCount;
	const unsigned int colonyStateId = (colonyId * GS::gColonyCount) % GS::gAllColonyCount + colonyId / GS::gEnvironmentCount;
	const unsigned int antStateId = colonyStateId * GS::gAntCount + idx % GS::gAntCount;
	Ants[idx].Init(&Colonies[colonyId], &AntStates[antStateId]);
}

__global__ void InitAnts_Duel(Ant *Ants, Colony *Colonies, AntState *AntStates, Ant::AntType Type1, Ant::AntType Type2)
{
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int colonyId = idx / GS::gAntCount;
	const unsigned int colonyStateId = (colonyId * GS::gColonyCount) % 2 + colonyId;
	const unsigned int antStateId = colonyStateId * GS::gAntCount + idx % GS::gAntCount;
	const Ant::AntType strategy = ((colonyId ? Type2 : Type1) == Ant::AntType::HALF_HALF) ? ((idx % 2) ? Ant::AntType::ATTACKER : Ant::AntType::GOGETTER) : (colonyId ? Type2 : Type1);
	Ants[idx].Init(&Colonies[colonyId], &AntStates[antStateId], strategy);
}

__global__ void ResetAnts(Ant *Ants, Colony *Colonies, AntState *AntStates)
{
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int colonyId = idx / GS::gAntCount;
	const unsigned int colonyStateId = (colonyId * GS::gColonyCount) % GS::gAllColonyCount + colonyId / GS::gEnvironmentCount;
	const unsigned int antStateId = colonyStateId * GS::gAntCount + idx % GS::gAntCount;
	Ants[idx].Reset(&Colonies[colonyId], &AntStates[antStateId]);
}

__global__ void SumWeights(Colony *Colonies, int Amount, float *SumWeights)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = 0; i < Amount; ++i)
		SumWeights[idx] += Colonies[i].mHiveMind.mWeights[idx];
	SumWeights[idx] /= Amount;
}

__global__ void Evaluate_Duel_Device(Colony *Colonies)
{
	printf("\nFinal results: %i - %i\n", Colonies[0].mState->mPoints, Colonies[1].mState->mPoints);
}

void SimulationHandler::Evaluate_Duel()
{
	CUDA_CHECK
	Evaluate_Duel_Device<<<1, 1>>>(mColonies);
	cudaDeviceSynchronize();
	CUDA_CHECK
}

__global__ void CleanupColonies(Colony *Colonies)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	Colonies[idx].mHiveMind.Cleanup();
	for (int i = 0; i < GS::gAntCount; ++i)
		Colonies[idx].mArmy[i].FreeMemory();
}

void SimulationHandler::FreeMemory()
{
	CUDA_CHECK
	CleanupColonies<<<1, mSimMode != DUEL ? GS::gAllColonyCount : GS::gColonyCount>>>(mColonies);
	CUDA_CHECK
	cudaDeviceSynchronize();
	CUDA_CHECK
	cudaFree(mColonies);
	cudaFree(mAnts);
	cudaFree(mDevice_ColonyStates);
	cudaFree(mDevice_AntStates);
	cudaFree(mRandState);
	free(mHost_ColonyStates);
	free(mHost_AntStates);
}

void SimulationHandler::GenerateStartingState(bool LoadState)
{
	//ToDo: Reset mHost_WeightTransfer to null?
	static std::chrono::time_point<std::chrono::steady_clock> startTime;
	const auto endTime = std::chrono::steady_clock::now();

	if (!LoadState)
		++mIteration;

	//Calculate the time of one generation
	if (startTime.time_since_epoch() == std::chrono::seconds(0))
		printf("\nITER #%i\n", mIteration);
	else
		printf("\nITER #%i - %lld ms\n", mIteration, std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count());

	if (mIteration == 0 || LoadState)
	{
		cudaMalloc(&mColonies, sizeof(Colony) * GSettings::gAllColonyCount);
		cudaMalloc(&mAnts, sizeof(Ant) * GSettings::gAllAntCount);
		cudaMalloc(&mDevice_ColonyStates, sizeof(ColonyState) * GSettings::gAllColonyCount);
		cudaMalloc(&mDevice_AntStates, sizeof(AntState) * GSettings::gAllAntCount);
		cudaMalloc(&mRandState, sizeof(curandState));
		mHost_ColonyStates = (ColonyState *)malloc(sizeof(ColonyState) * GSettings::gAllColonyCount);
		mHost_AntStates = (AntState *)malloc(sizeof(AntState) * GSettings::gAllAntCount);

		InitColonies<<<1, GS::gAllColonyCount>>>(mColonies, mAnts, mDevice_ColonyStates, mDevice_AntStates);
		SetupWeights<<<1, 1>>>(mColonies, mRandState, LoadState ? mHost_WeightTransfer : nullptr, mIteration);
		InitAnts<<<GS::gColonyCount * GS::gAntCount, GS::gEnvironmentCount>>>(mAnts, mColonies, mDevice_AntStates);
		CUDA_CHECK;
	}
	else
	{
		ResetColonies<<<1, GS::gAllColonyCount>>>(mColonies, mAnts, mDevice_ColonyStates, mDevice_AntStates);
		//The weights are already set up in Evaluation
		//SetupWeights<<<1, 1>>>(mColonies, mRandState);
		ResetAnts<<<GS::gColonyCount * GS::gAntCount, GS::gEnvironmentCount>>>(mAnts, mColonies, mDevice_AntStates);
	}

	if ((mIteration % GS::gAutoSavePeriod == 0) && !LoadState)
		SaveState();

	startTime = std::chrono::steady_clock::now();
}

void SimulationHandler::GenerateStartingState_Duel(Ant::AntType Type1, Ant::AntType Type2)
{
	cudaMalloc(&mColonies, sizeof(Colony) * 2);
	cudaMalloc(&mAnts, sizeof(Ant) * 2 * GS::gAntCount);
	cudaMalloc(&mDevice_ColonyStates, sizeof(ColonyState) * 2);
	cudaMalloc(&mDevice_AntStates, sizeof(AntState) * 2 * GS::gAntCount);
	mHost_ColonyStates = (ColonyState *)malloc(sizeof(ColonyState) * 2);
	mHost_AntStates = (AntState *)malloc(sizeof(AntState) * 2 * GS::gAntCount);

	InitColonies_Duel<<<1, 2>>>(mColonies, mAnts, mDevice_ColonyStates, mDevice_AntStates);
	SetupWeights_Duel<<<1, 1>>>(mColonies, mHost_WeightTransfer);
	InitAnts_Duel<<<GS::gColonyCount * GS::gAntCount, 1>>>(mAnts, mColonies, mDevice_AntStates, Type1, Type2);
	CUDA_CHECK;
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

constexpr int threadPerBlock = 128;
constexpr int blockCount = (GS::gEnvironmentCount * GS::gAntCount - 1) / threadPerBlock + 1;

//Unnecessary check, it's not faster to choose smaller threadPerBlock than 2^N
//static_assert(threadPerBlock * blockCount == GS::gEnvironmentCount * GS::gAntCount, "Incorrect threadPerBlock value");

__global__ void AntSteps(Ant *Ants, const unsigned int From = 0)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < GS::gEnvironmentCount * GS::gAntCount)
		Ants[From + idx].Step();
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
		if (Start[i].mIsWinner && i < GS::gEnvironmentCount / 3) // 3 is an arbitary number to force mutation even when everyone wins.
			firstNonWinner = i + 1;
	}

	//Calculate average weights of the loser and winner teams
	const int &weightCount = Colonies[0].mHiveMind.mWeightCount;
	float *avgWinnerWeights = new float[weightCount]();
	if (firstNonWinner != 0)
		SumWeights<<<1, weightCount>>>(Start, firstNonWinner, avgWinnerWeights);	//ToDo: Maybe use Losers as negative weight?
	cudaDeviceSynchronize();	//ToDo: Deprecated to call on device

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

void SimulationHandler::SimulationStep()
{
	CUDA_CHECK;
	//Defenders' turn;
	AntSteps<<<blockCount, threadPerBlock>>>(mAnts);
	//Attackers' turn;
	AntSteps<<<blockCount, threadPerBlock>>>(mAnts, GS::gEnvironmentCount * GS::gAntCount);
	++mStepCounter;

	if (mStepCounter == GS::gSimulationTime)
	{
		cudaDeviceSynchronize();
		Evaluate_Device<<<1, 1>>>(mColonies, mRandState);
		mStepCounter = 0;
		GenerateStartingState();
		cudaDeviceSynchronize();
	}
}

void SimulationHandler::SimulationStep_Duel()
{
	//Defenders' turn;
	AntSteps<<<1, 10>>>(mAnts);
	//Attackers' turn;
	AntSteps<<<1, 10>>>(mAnts, GS::gAntCount);
	++mStepCounter;
}