#pragma once

//Using SDL and standard IO
//#include <SDL.h>
#include <thread>
//#include "GlobalSettings.h" //Unnecessary as long as Colony.cuh->Ant.cuh has it
//#include "Colony.cuh"
#include <Windows.h>
#include <d2d1.h>

#include <sstream> //std::wstream

//Avoiding unnecessary includes
class Ant;
struct AntState;
class Colony;
struct ColonyState;
struct curandStateXORWOW;
typedef struct curandStateXORWOW curandState;
struct IDWriteFactory;

class SimulationHandler
{
public:
    
    //__host__
    bool mCloseWindow = false;  //ToDo: Can this variable be removed?
    int mGeneration = -1;
    AntState *mHost_AntStates = nullptr;
    ColonyState *mHost_ColonyStates = nullptr;
    float *mHost_WeightTransfer = nullptr;
    bool mPaused = false;
    bool mSlowMode = false;
    int mStepCounter = 0;
    bool mStopSimulation = false;
    std::wstring mStrBuffer;
    //SDL_Surface *mScreenSurface = NULL;
    std::thread *mSimulationThread = nullptr;
    //SDL_Window *mWindow = NULL;
    std::thread *mWindowThread = nullptr;

    HWND mHwnd;
    ID2D1Factory *mDirect2dFactory = nullptr;
    IDWriteFactory *mDWriteFactory = nullptr;
    IDWriteTextFormat *mTextFormat = nullptr;
    ID2D1HwndRenderTarget *mRenderTarget = nullptr;
    ID2D1SolidColorBrush *mBrush = nullptr;

    //__device__
    Ant *mAnts = nullptr;
    Colony *mColonies = nullptr;
    AntState *mDevice_AntStates = nullptr;
    ColonyState *mDevice_ColonyStates = nullptr;
    float *mDevice_WeightTransfer = nullptr;
    curandState *mRandState = nullptr;

    static SimulationHandler *sSimulationHandler;

    SimulationHandler();
    ~SimulationHandler();

    void ChangeSpeed();
    void CleanUpWindowThread();
    void LoadState();
    void OpenWindow();
    void PauseSimulation();
    void SaveState() const;
    void StartSimulation();
    void StopSimulation();

protected:
    void AttackerStep();
    void DefenderStep();
    void Evaluate();
    void FreeMemory();
    void GenerateLoadedState();
    void GenerateStartingState(bool LoadState = false);
    void OnRender();
    void OnResize(UINT width, UINT height);
    void RefreshStringBuffer();
    void SaveDeviceToHost() const;
    void SimulationStep();
    void SimulationThread(bool LoadState);
    void WindowThread();

    static LRESULT CALLBACK WndProc(HWND HWnd, UINT Message, WPARAM WParam, LPARAM LParam);
};