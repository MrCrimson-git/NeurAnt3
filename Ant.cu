#include "GlobalSettings.h"
#include "Ant.cuh"
#include "Colony.cuh"

#define _USE_MATH_DEFINES
#include <math.h>
#include <float.h>

#include <stdio.h>

__device__ inline float distanceToInput(const float &In)
{
	return -1 + 2 / (1 + In * GSettings::g4_MapSizeX);
}

__device__ float length(const float2 &In)
{
	return hypotf(In.x, In.y);
}

__device__ float length2(const float2 &In)
{
	return In.x*In.x + In.y*In.y;
}

__device__ float normalize(float rotation)
{
	if (rotation > 1.0f)
		rotation -= 2.0f;
	else if (rotation < -1.0f)
		rotation += 2.0f;
	return rotation;
}

__device__ float2 operator - (const float2 &lhs, const float2 &rhs)
{
	return { lhs.x - rhs.x, lhs.y - rhs.y };
}

__device__ float2 operator + (const float2 &lhs, const float2 &rhs)
{
	return { lhs.x + rhs.x, lhs.y + rhs.y };
}

__device__ void operator += (float2 &lhs, const float2 &rhs)
{
	lhs.x += rhs.x;
	lhs.y += rhs.y;
}

__device__ inline float2 rotate(const float Length, const float Angle)
{
	float realAngle = Angle * M_PI;
	float sina = sinf(realAngle);
	float cosa = cosf(realAngle);
	return { Length * cosa, Length * sina };
}

__device__ inline float rotationFrom(const float2 In, float Angle)
{
	float tang = atan2f(In.y, In.x) * M_1_PI;
	return normalize(Angle - tang);
}

__device__ void Ant::Died()
{
	mState->mRotation = -1.f + 2.f / GS::gAntCount * (this - mBase->mArmy);
	mState->mPosition = mBase->mState->mPosition;
	if (mState->mHasFlag)
		mBase->AddPoint(-1);
	mState->mHasFlag = false;
}

__device__ void Ant::FreeMemory()
{
	mNeuralSystem.Cleanup();

	if (mAntRandState)
	{
		delete mAntRandState;
		mAntRandState = nullptr;
	}
}

__device__ void Ant::Init(Colony *Base, AntState *State, AntType Type)
{
	mBase = Base;
	mNeuralSystem.Init(mBase->mHiveMind);
	
	mState = State;
	mStrategy = Type;
	mState->mPosition = mBase->mState->mPosition;
	mState->mRotation = -1.f + 2.f / GS::gAntCount * (this - mBase->mArmy);
	mState->mHasFlag = false;
	mAntRandState = nullptr;

	if (Type == AntType::RANDOM)
	{
		mAntRandState = new AntRandState;
		curand_init(RAND_SEED, this - mBase->mArmy, 0ULL, &mAntRandState->mRandState);
		mOutputs[0] = 0.f;
		mOutputs[1] = 0.f;
	}
}

__device__ void Ant::Reset(Colony *Base, AntState *State)
{
	mBase = Base;
	mNeuralSystem.Reset(mBase->mHiveMind);

	mState = State;
	mState->mPosition = mBase->mState->mPosition;
	mState->mRotation = -1.f + 2.f / GS::gAntCount * (this - mBase->mArmy);
	mState->mHasFlag = false;
}

__device__ void Ant::Step()
{
	const Colony &otherBase = *mBase->mOtherColony;
	const float distanceToBase = length(mState->mPosition - mBase->mState->mPosition);
	const float rotationToBase = rotationFrom(mBase->mState->mPosition - mState->mPosition, mState->mRotation);
	const float distanceToOtherBase = length(mState->mPosition - otherBase.mState->mPosition);
	const float rotationToOtherBase = rotationFrom(otherBase.mState->mPosition - mState->mPosition, mState->mRotation);

	Ant *enemyAnt_1 = nullptr;
	Ant *enemyAnt_2 = nullptr;
	Ant const *allyAnt = nullptr;
	float enemyDistance2_1 = FLT_MAX;
	float enemyDistance2_2 = FLT_MAX;
	float allyDistance2 = FLT_MAX;

	for (int i = 0; i < GS::gAntCount; ++i)
	{
		Ant &otherAnt = otherBase.mArmy[i];
		const float curDist2 = length2(mState->mPosition - otherAnt.mState->mPosition);
		/*if (curDist2 < GS::gAntSize2)
		{
			const float enemyRot = rotationFrom(otherAnt.mState->mPosition - mState->mPosition, mState->mRotation);
			if (enemyRot < GS::gAntAttackAngle && enemyRot > -GS::gAntAttackAngle)
				otherAnt.Died();
		}*/

		if (curDist2 < enemyDistance2_1)
		{
			enemyAnt_2 = enemyAnt_1;
			enemyDistance2_2 = enemyDistance2_1;
			enemyAnt_1 = &otherAnt;
			enemyDistance2_1 = curDist2;
		}
		else if (curDist2 < enemyDistance2_2)
		{
			enemyAnt_2 = &otherAnt;
			enemyDistance2_2 = curDist2;
		}
	}

	for (int i = 0; i < GS::gAntCount; ++i)
	{
		const Ant &otherAnt = mBase->mArmy[i];
		if (&otherAnt == this)
			continue;
		const float curDist2 = length2(mState->mPosition - otherAnt.mState->mPosition);
		if (curDist2 < allyDistance2)
		{
			allyAnt = &otherAnt;
			allyDistance2 = curDist2;
		}
	}

	const float rotationToEnemy_1 = enemyAnt_1 ? rotationFrom(enemyAnt_1->mState->mPosition - mState->mPosition, mState->mRotation) : 0.0f;
	const float rotationOfEnemy_1 = normalize(enemyAnt_1->mState->mRotation - mState->mRotation);
	const float rotationToEnemy_2 = enemyAnt_2 ? rotationFrom(enemyAnt_2->mState->mPosition - mState->mPosition, mState->mRotation) : 0.0f;
	const float rotationOfEnemy_2 = normalize(enemyAnt_2->mState->mRotation - mState->mRotation);
	const float rotationToAlly = allyAnt ? rotationFrom(allyAnt->mState->mPosition - mState->mPosition, mState->mRotation) : 0.0f;
	const float rotationOfAlly = normalize(allyAnt->mState->mRotation - mState->mRotation);

	mInputs[__COUNTER__] = mState->mHasFlag ? 1.0f : -1.0f;

	mInputs[__COUNTER__] = distanceToInput(distanceToBase);
	mInputs[__COUNTER__] = rotationToBase;
	mInputs[__COUNTER__] = distanceToInput(distanceToOtherBase);
	mInputs[__COUNTER__] = rotationToOtherBase;

	mInputs[__COUNTER__] = rotationToEnemy_1;
	mInputs[__COUNTER__] = rotationOfEnemy_1;
	mInputs[__COUNTER__] = distanceToInput(sqrtf(enemyDistance2_1));
	mInputs[__COUNTER__] = enemyAnt_1->mState->mHasFlag ? 1.0f : -1.0f;
	mInputs[__COUNTER__] = rotationToEnemy_2;
	mInputs[__COUNTER__] = rotationOfEnemy_2;
	mInputs[__COUNTER__] = distanceToInput(sqrtf(enemyDistance2_2));
	mInputs[__COUNTER__] = enemyAnt_2->mState->mHasFlag ? 1.0f : -1.0f;

	mInputs[__COUNTER__] = rotationToAlly;
	mInputs[__COUNTER__] = rotationOfAlly;
	mInputs[__COUNTER__] = distanceToInput(sqrtf(allyDistance2));
	mInputs[__COUNTER__] = allyAnt->mState->mHasFlag ? 1.0f : -1.0f;

#if __COUNTER__ != INPUT_COUNT
#error INPUT_COUNT has to be adjusted
#endif

	//Make decision
	switch (mStrategy)
	{
	case AntType::NEURAL: Step_Neural(); break;
	case AntType::GOGETTER: Step_GoGetter(); break;
	case AntType::ATTACKER: Step_Attacker(); break;
	default: Step_Random();
	}

	mState->mRotation = normalize(mState->mRotation + mOutputs[0] * GS::gAntMaxRotation);
	mState->mPosition += rotate(GSettings::gAntMaxVelocity * (mOutputs[1] + 1.0f) * 0.5f, mState->mRotation);

	//Attack the closer enemy ant if they are close enough and in direction
	const float newEnemyDistance2_1 = length2(mState->mPosition - enemyAnt_1->mState->mPosition);
	const float newEnemyDistance2_2 = length2(mState->mPosition - enemyAnt_2->mState->mPosition);

	bool enemy1IsValid = false;
	bool enemy2IsValid = false;
	float enemyRot;

	if (newEnemyDistance2_1 < GS::gAntSize2) {
		enemyRot = rotationFrom(enemyAnt_1->mState->mPosition - mState->mPosition, mState->mRotation);
		if (enemyRot < GS::gAntAttackAngle && enemyRot > -GS::gAntAttackAngle) {
			enemy1IsValid = true;
		}
	}

	if (newEnemyDistance2_2 < GS::gAntSize2) {
		enemyRot = rotationFrom(enemyAnt_2->mState->mPosition - mState->mPosition, mState->mRotation);
		if (enemyRot < GS::gAntAttackAngle && enemyRot > -GS::gAntAttackAngle) {
			enemy2IsValid = true;
		}
	}

	if (enemy1IsValid && enemy2IsValid) {
		if (newEnemyDistance2_1 < newEnemyDistance2_2) {
			enemyAnt_1->Died();
	}
		else {
			enemyAnt_2->Died();
		}
	}
	else if (enemy1IsValid) {
		enemyAnt_1->Died();
	}
	else if (enemy2IsValid) {
		enemyAnt_2->Died();
	}

	if (mState->mHasFlag)
	{
		if (distanceToBase < GSettings::gNestRadius)
		{
			mBase->AddPoint(1);
			mState->mHasFlag = false;
		}
	}
	else if (distanceToOtherBase < GSettings::gNestRadius)
	{
		mBase->AddPoint(0);
		mState->mHasFlag = true;
	}
}

__device__ void Ant::Step_Neural()
{
	mNeuralSystem.Calculate(mInputs, mOutputs);
}

__device__ void Ant::Step_Random()
{
	for (int i = 0; i < 2; ++i)
	{
		//Adds random number between +/- mMaxDiffChange
		mAntRandState->mPrevDiff[i] += (curand_uniform(&mAntRandState->mRandState) - 0.5f) * 2.0f * mAntRandState->mMaxDiffChange;
		mOutputs[i] += mAntRandState->mPrevDiff[i];

		//Clamp
		if (abs(mOutputs[i]) > 1.f)
		{
			//Reset mPrevDiff so it won't keep trying to change in the wrong direction as much
			mAntRandState->mPrevDiff[i] = 0.f;
			mOutputs[i] = mOutputs[i] > 1.f ? 1.f : -1.f;
		}
	}
}

__device__ void Ant::Step_GoGetter()
{
	const float rotationToNextBase = mState->mHasFlag ? mInputs[2] : mInputs[4]; // rotationToBase : rotationToOtherBase
	
	mOutputs[0] = fmaxf(fminf(-rotationToNextBase / GS::gAntMaxRotation, 1.f), -1.f);
	mOutputs[1] = 1.0;
}

__device__ void Ant::Step_Attacker()
{
	mOutputs[0] = fmaxf(fminf(-mInputs[5] / GS::gAntMaxRotation, 1.f), -1.f);	// rotationToEnemy_1
	mOutputs[1] = 1.0;
}