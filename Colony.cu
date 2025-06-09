#include <cuda_runtime_api.h> //ToDo: Might be unnecessary include

#include "Colony.cuh"

__device__ void Colony::Init(Ant *Ants, ColonyState *State, Colony *OtherColony)
{
	mArmy = Ants;
	mOtherColony = OtherColony;
	mState = State;
	mState->mPoints = 0;
	mAlmostPoints = 0;
	mIsWinner = false;
}

__device__ void Colony::AddPoint(int Type)
{
	if (Type == -1)
	{
		atomicAdd(&mAlmostPoints, -1);
	}
	else if (Type == 0)
	{
		atomicAdd(&mAlmostPoints, 1);
	}
	else
	{
		atomicAdd(&mState->mPoints, 1);
		atomicAdd(&mAlmostPoints, -1);
	}
	SetIsWinner();
}

__device__ void Colony::Reset(Ant *Ants, ColonyState *State, Colony *OtherColony)
{
	mArmy = Ants;
	mOtherColony = OtherColony;
	mState = State;
	mState->mPoints = 0;
	mAlmostPoints = 0;
	mIsWinner = false;
}

__device__ void Colony::SetIsWinner()
{
	if (mState->mPoints != mOtherColony->mState->mPoints)
	{
		mIsWinner = mState->mPoints > mOtherColony->mState->mPoints;
		mOtherColony->mIsWinner = !mIsWinner;
	}
	else if (mAlmostPoints != mOtherColony->mAlmostPoints)
	{
		mIsWinner = mAlmostPoints > mOtherColony->mAlmostPoints;
		mOtherColony->mIsWinner = !mIsWinner;
	}
	else
		mIsWinner = mOtherColony->mIsWinner = false;
}

__host__ __device__ bool Colony::operator>(const Colony&other) const
{
	if (mIsWinner != other.mIsWinner)
		return mIsWinner;
	int diff1 = mState->mPoints - mOtherColony->mState->mPoints;
	int diff2 = other.mState->mPoints - other.mOtherColony->mState->mPoints;
	if (diff1 != diff2)
		return diff1 > diff2;
	if (mState->mPoints != other.mState->mPoints)
		return mState->mPoints > other.mState->mPoints;
	if (mAlmostPoints != other.mAlmostPoints)
		return mAlmostPoints > other.mAlmostPoints;
	return false;
}