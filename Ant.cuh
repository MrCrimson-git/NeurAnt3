#pragma once

#include "GlobalSettings.h"
#include <array>
#include "NeuralNetwork.cuh"

struct __align__(16) AntState
{
	float2 mPosition;
	float mRotation;
	bool mHasFlag = false;
};

struct __align__(16) AntRandState
{
	curandState mRandState;
	float mPrevDiff[2]{};
	const float mMaxDiffChange = .001f;
};

class Colony;

class __align__(16) Ant
{
public:
	enum class AntType
	{
		NEURAL,
		RANDOM,
		GOGETTER,
		ATTACKER,
		HALF_HALF
	} mStrategy;

	AntState *mState;
#ifdef INPUT_PERSONALITY
	float mPersonality;
#endif
	Colony *mBase;
	float mInputs[INPUT_COUNT];
	float mOutputs[OUTPUT_COUNT];
	NeuralNetwork mNeuralSystem;
	__device__ void Died();
	__device__ void FreeMemory();
	__device__ void Init(Colony *Base, AntState *State, AntType Type = AntType::NEURAL);
	__device__ void Reset(Colony *Base, AntState *State);
	__device__ void Step();

private:
	__device__ inline void Step_Neural();
	__device__ inline void Step_Random();
	__device__ inline void Step_GoGetter();
	__device__ inline void Step_Attacker();

	//Only used with RANDOM types in duel mode!
	AntRandState *mAntRandState;
public:
};