//#define TEST

#include <iostream>

#include <stdio.h>
#include <string>

#include "SimulationHandler.h"


class Menu
{
public:
    enum Action { EXIT = -1, STAY, MAIN, SIMULATION, SETTINGS };
    virtual void ToScreen() = 0;
    virtual Action HandleInput(std::string Input) { return STAY; };
    virtual ~Menu() {};
};

class MainMenu : public Menu
{
    virtual void ToScreen()
    {
        system("CLS");
        std::cout << "MAIN MENU\n" << std::endl
            << "1 - Start" << std::endl
            << "2 - Settings" << std::endl
            << "0 - Exit program" << std::endl
            << "Here will go other options" << std::endl
            << "That all end up in separate lines" << std::endl << std::endl;
    }

    virtual Action HandleInput(std::string Input)
    {
        if (Input == "1")
            return Menu::SIMULATION;
        else if (Input == "2")
            return Menu::SETTINGS;
        else if (Input == "0")
            return Menu::EXIT;
        return STAY;
    };
};

class SimulationMenu : public Menu
{
public:
    std::thread *mWindowThread = NULL;

    virtual ~SimulationMenu()
    {
        //StopSimulation();
    }

#define SIM_HANDLER_FUNCTION(FuncNam) void FuncNam() \
    { if (SimulationHandler::sSimulationHandler) SimulationHandler::sSimulationHandler->FuncNam(); }

    void StartSimulation()
    {
        if (!SimulationHandler::sSimulationHandler)
            SimulationHandler::sSimulationHandler = new SimulationHandler;
        //else
            SimulationHandler::sSimulationHandler->StartSimulation();
    }

    SIM_HANDLER_FUNCTION(OpenWindow)
    SIM_HANDLER_FUNCTION(ChangeSpeed)
    SIM_HANDLER_FUNCTION(PauseSimulation)
    SIM_HANDLER_FUNCTION(SaveState)
    void LoadState()
    {
        if (!SimulationHandler::sSimulationHandler)
            SimulationHandler::sSimulationHandler = new SimulationHandler;
        //else
            SimulationHandler::sSimulationHandler->LoadState();
    }

    void StopSimulation()
    {
        if (SimulationHandler::sSimulationHandler)
        {
            delete SimulationHandler::sSimulationHandler;
            SimulationHandler::sSimulationHandler = NULL;
        }
    }

    virtual void ToScreen()
    {
        //system("CLS");
        std::cout << "SIMULATION\n" << std::endl
            << "start - Start a new simulation" << std::endl
            << "load - Load a previous state" << std::endl
            << "save - Save the current state" << std::endl
            << "open - Open the visual window" << std::endl
            << "slow - Change speed mode" << std::endl
            << "pause - Pause/unpause the simulation" << std::endl
            << "stop - Stop the simulation" << std::endl
            << "0 - Back to Main Menu" << std::endl << std::endl;
    }

    virtual Action HandleInput(std::string Input)
    {
        if (Input == "start")
            StartSimulation();
        else if (Input == "load")
            LoadState();
        else if (Input == "save")
            SaveState();
        else if (Input == "open")
            OpenWindow();
        else if (Input == "slow")
            ChangeSpeed();
        else if (Input == "pause")
            PauseSimulation();
        else if (Input == "stop")
            StopSimulation();
        else if (Input == "0")
            return Menu::MAIN;
        return STAY;
    };
};

#include <cuda_runtime_api.h>
class SettingsMenu : public Menu
{
    virtual void ToScreen()
    {
        int device;
        cudaDeviceProp deviceProp;

        cudaGetDevice(&device);
        cudaGetDeviceProperties(&deviceProp, device);

        int concurrentManagedAccess;
        cudaDeviceGetAttribute(&concurrentManagedAccess, cudaDevAttrConcurrentManagedAccess, device);

        size_t stackSize, heapSize, newHeapSize;
        cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
        cudaDeviceGetLimit(&heapSize, cudaLimitMallocHeapSize);
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, 4U * heapSize);
        cudaDeviceGetLimit(&newHeapSize, cudaLimitMallocHeapSize);

        system("CLS");
        std::cout << "SETTINGS\n" << std::endl
            << "Device: " << device << " - " << deviceProp.name << std::endl
            << "Total global memory: " <<  deviceProp.totalGlobalMem / 1'048'056 << " MB" << std::endl
            << "Shared memory / block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl
            << "Total constant memory: " << deviceProp.totalConstMem / 1024 << " KB" << std::endl
            << "Number of multiprocessors: " << deviceProp.multiProcessorCount << std::endl
            << "Concurrent managed access: " << deviceProp.concurrentManagedAccess << std::endl
            << std::endl
            << "Stack size limit: " << stackSize << std::endl
            << "Heap size limit: " << heapSize << std::endl
            << "New heap size limit: " << newHeapSize << std::endl
            << "\n0 - Back to Main Menu" << std::endl << std::endl;
    }

    virtual Action HandleInput(std::string Input)
    {
        if (Input == "0")
            return Menu::MAIN;
        return STAY;
    };
};

void NextMenu(Menu *&MenuPtr, Menu::Action Action)
{
    if (Action != Menu::STAY && MenuPtr)
    {
        delete MenuPtr;
        MenuPtr = NULL;
    }

    switch (Action)
    {
    case Menu::MAIN: MenuPtr = new MainMenu; break;
    case Menu::SIMULATION: MenuPtr = new SimulationMenu; break;
    case Menu::SETTINGS: MenuPtr = new SettingsMenu; break;
    case Menu::STAY: //NOOP
    default: break;//NOOP
    }

}

#ifndef TEST
int main(int argc, char* args[])
{
    Menu *currentMenu = NULL;

    Menu::Action nextAction = Menu::MAIN;

    while (nextAction != Menu::EXIT)
    {
        NextMenu(currentMenu, nextAction);
        currentMenu->ToScreen();
        std::string input;
        std::cin >> input;
        nextAction = currentMenu->HandleInput(input);
    }

    return 0;
}

#else
#include <iomanip>
#include "./test.cuh"

void CudaTest()
{
    TestClass *testObject_host, *testObject_device;

    cudaMallocHost(&testObject_host, sizeof(TestClass));
    cudaMalloc(&testObject_device, sizeof(TestClass));

    kernel(testObject_device);

    cudaMemcpy(testObject_host, testObject_device, sizeof(TestClass), cudaMemcpyDeviceToHost);

    // Print out result.
    std::cout << std::fixed << std::setprecision(6) << *(testObject_host->testVal) << std::endl;
    system("pause");
}

int main()
{
    CudaTest();
    return 0;
}
#endif //TEST