#include "GlobalSettings.h"

namespace GSettings
{
	//__constant__ int gLayers[] = { INPUT_COUNT, MIDDLE_COUNT, OUTPUT_COUNT };
	//__constant__ int gLayerCount = sizeof(gLayers) / sizeof(gLayers[0]);
}

namespace GVars
{
	int gAntsToDraw = GSettings::gAllAntCount;
}