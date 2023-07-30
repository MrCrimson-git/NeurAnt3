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

class Colony;

class __align__(16) Ant
{
public:
	AntState *mState;
	bool mAlive;
#ifdef INPUT_PERSONALITY
	float mPersonality;
#endif
	Colony *mBase;
	float mInputs[INPUT_COUNT];
	float mOutputs[OUTPUT_COUNT];
	NeuralNetwork mNeuralSystem;
	__device__ void Died();
	__device__ void Init(Colony *Base, AntState *State);
	__device__ void Reset(Colony *Base, AntState *State);
	__device__ void Step(const int StepCount);
};