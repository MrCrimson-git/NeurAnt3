#pragma once

#include <thread>
//#include "GlobalSettings.h" //Unnecessary as long as Colony.cuh->Ant.cuh has it
#include "Ant.cuh"
#include <Windows.h>
#include <d2d1.h>

#include <sstream> //std::wstream

//Avoiding unnecessary includes
class Colony;
struct ColonyState;
struct curandStateXORWOW;
typedef struct curandStateXORWOW curandState;
struct IDWriteFactory;

#define TEXT_RENDER false

class SimulationHandler
{
public:
    enum SimMode { NEW, LOAD, DUEL };

    //__host__
    bool mCloseWindow = false;
    int mIteration = -1;
    AntState *mHost_AntStates = nullptr;
    ColonyState *mHost_ColonyStates = nullptr;
    float *mHost_WeightTransfer = nullptr;
    bool mPaused = false;
    SimMode mSimMode = NEW;
    bool mSlowMode = false;
    int mStepCounter = 0;
    bool mStopSimulation = false;
#if TEXT_RENDER
    std::wstring mStrBuffer;
#endif //TEXT_RENDER
    std::thread *mSimulationThread = nullptr;
    std::thread *mWindowThread = nullptr;

    HWND mHwnd;
    ID2D1Factory *mDirect2dFactory = nullptr;
    IDWriteFactory *mWriteFactory = nullptr;
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
    bool StartDuel();
    void StartSimulation();
    void StopSimulation();

protected:
    void AttackerStep_Duel();
    void DefenderStep_Duel();
    void DuelThread(int First, int Second);
    void Evaluate_Duel();
    void FreeMemory();
    void GenerateStartingState_Duel(Ant::AntType Type1, Ant::AntType Type2);
    void GenerateLoadedState();
    void GenerateStartingState(bool LoadState = false);
    void OnRender();
#if TEXT_RENDER
    void RefreshStringBuffer();
#endif //TEXT_RENDER
    void SaveDeviceToHost() const;
    void SimulationStep();
    void SimulationStep_Duel();
    void SimulationThread(SimMode LoadState);
    void WindowThread();

    static LRESULT CALLBACK WndProc(HWND HWnd, UINT Message, WPARAM WParam, LPARAM LParam);

private:
    inline bool ReadSaveFile(const std::string &SaveFile, const unsigned short Slot);
};