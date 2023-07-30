#pragma once

#include "NeuralNetwork.cuh"

struct AntState;

struct __align__(16) ColonyState
{
	float2 mPosition;
	int mPoints;
};

class __align__(16) Colony
{
	friend class Ant;
public:
	ColonyState* mState;
	Ant *mArmy;
	Colony *mOtherColony;
	//int mPointDiff;
	int mAlmostPoints;
	int mPointTime;
	int mAge;
	AntState *mArmyStates;	//ToDo: Might be unnecessary
	bool mIsWinner;
	NeuralNetworkBase mHiveMind;
	//curandState* mRandState;
	__device__ void Init(Ant *Ants, ColonyState *State, AntState *AntStates, Colony *OtherColony);
	__device__ void AddPoint(int Type, int Time);
	__device__ void Reset(Ant *Ants, ColonyState* State, AntState* AntStates, Colony *OtherColony);
	//__device__ void Mutate(NeuralNetworkBase& ParentMind, curandState* RandState);
	__device__ void SetIsWinner();
	__host__ __device__ bool operator>(const Colony& other) const;
};
//curandState* mRandState;
