#pragma once

#include <cuda_runtime_api.h>
#include <assert.h>
#define BINARY_SAVE

//#define CUDA_ERROR_CHECK
#define CUDA_CHECK_FORCE \
{ \
    cudaError_t error = cudaGetLastError(); \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        assert(error == cudaSuccess); \
    } \
}

#ifdef CUDA_ERROR_CHECK
#define CUDA_CHECK CUDA_CHECK_FORCE
#else
#define CUDA_CHECK
#endif //CUDA_ERROR_CHECK

#define INPUT_COUNT 17
#define MIDDLE_COUNT 15
#define OUTPUT_COUNT 2

#define BIASED false

#define RAND_SEED 5ULL

namespace GSettings
{
	constexpr int gScreenWidth = 1280;
	constexpr int gScreenHeight = 800;

	constexpr int gAutoSavePeriod = 100;

	constexpr int gMapSizeX = 1440;
	constexpr float g4_MapSizeX = 4.0f / gMapSizeX;
	constexpr int gMapSizeY = 900;

	constexpr int gEnvironmentCount = 500;
	constexpr int gColonyCount = 2;
	constexpr int gAntCount = 10;

	constexpr int gAllColonyCount = gEnvironmentCount * gColonyCount;
	constexpr int gAllAntCount = gEnvironmentCount * gColonyCount * gAntCount;

	constexpr float gNestRadius = 25.0f;

	constexpr float gAntSize = 10.0f;
	constexpr float gAntSize_2 = gAntSize * .5f;
	constexpr float gAntSize2 = gAntSize * gAntSize;
	constexpr float gAntAttackAngle = 0.25f;
	constexpr float gAntMaxVelocity = 0.7f;
	constexpr float gAntMaxRotation = 0.05f;

	constexpr int gSimulationTime = 6 * 60 * 60;

	__constant__ constexpr int gLayers[] = { INPUT_COUNT, MIDDLE_COUNT, OUTPUT_COUNT };
	__constant__ constexpr int gLayerCount = sizeof(gLayers) / sizeof(gLayers[0]);
	__device__ constexpr int CalcWeightCount()
	{
		return INPUT_COUNT * MIDDLE_COUNT + MIDDLE_COUNT * (MIDDLE_COUNT - 1) / 2 + MIDDLE_COUNT * OUTPUT_COUNT;

		int ret = 0;
		for (int i = 0; i < gLayerCount - 1; ++i)
		{
			ret += (gLayers[i] + BIASED) * gLayers[i + 1];
			if (i != 0)
				ret += gLayers[i] * (gLayers[i] - 1) / 2;
		}


		/*int ret = 0;
		for (int i = 1; i < gLayerCount; ++i)
			ret += gLayers[i] * (gLayers[i - 1] + 1);
		return ret;*/
	}
	__constant__ constexpr int gWeightCount = CalcWeightCount(); //CUDA gives warning, but works as intended.

	constexpr float gWeightStartRange = 0.8f;
	constexpr float gWeightDiffRange = 0.35f;
}

namespace GVars
{
	extern int gAntsToDraw;
}

namespace GS = GSettings;
namespace GV = GVars;