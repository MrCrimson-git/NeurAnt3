#pragma once

#include <iostream>
#include <stdio.h>
#include <string>

#include <cuda_runtime_api.h>

#include "SimulationHandler.h"


class Menu
{
public:
    enum Action { EXIT = -1, STAY, MAIN, SIMULATION, INFORMATION };
    virtual Action HandleInput(std::string Input) { return STAY; };
    virtual void ToScreen() = 0;
};

class MainMenu : public Menu
{
    virtual void ToScreen()
    {
        system("CLS");
        std::cout << "MAIN MENU\n" << std::endl
            << "1 - Start" << std::endl
            << "2 - Device Informations" << std::endl
            << "0 - Exit program" << std::endl << std::endl;
    }

    virtual Action HandleInput(std::string Input)
    {
        if (Input == "1")
            return Menu::SIMULATION;
        else if (Input == "2")
            return Menu::INFORMATION;
        else if (Input == "0")
            return Menu::EXIT;
        return STAY;
    };
};

class SimulationMenu : public Menu
{
public:
#define SIM_HANDLER_FUNCTION(FuncName) void FuncName() \
    { if (SimulationHandler::sSimulationHandler) SimulationHandler::sSimulationHandler->FuncName(); }

    void StartSimulation()
    {
        if (!SimulationHandler::sSimulationHandler)
            SimulationHandler::sSimulationHandler = new SimulationHandler;

        SimulationHandler::sSimulationHandler->StartSimulation();
    }

    void StartDuel()
    {
        if (!SimulationHandler::sSimulationHandler)
            SimulationHandler::sSimulationHandler = new SimulationHandler;

         if (!SimulationHandler::sSimulationHandler->StartDuel())
             StopSimulation();
    }

    void LoadState()
    {
        if (!SimulationHandler::sSimulationHandler)
            SimulationHandler::sSimulationHandler = new SimulationHandler;

        SimulationHandler::sSimulationHandler->LoadState();
    }

    SIM_HANDLER_FUNCTION(SaveState);
    SIM_HANDLER_FUNCTION(OpenWindow);
    SIM_HANDLER_FUNCTION(ChangeSpeed);
    SIM_HANDLER_FUNCTION(PauseSimulation);

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
        bool onGoing = SimulationHandler::sSimulationHandler;
        bool isDuel = onGoing && (SimulationHandler::sSimulationHandler->mSimMode == SimulationHandler::SimMode::DUEL);

        if (!onGoing)
            system("CLS");
        std::cout << "\nSIMULATION";
            if (onGoing)
            {
                std::cout << " - Current mode: "
                    << (isDuel ? "DUEL" : "SIM");
                if (SimulationHandler::sSimulationHandler->mPaused)
                    std::cout << ", PAUSED";
            }
            std::cout << std::endl << std::endl;

        if (!onGoing)
            std::cout << "start - Start a new simulation" << std::endl
            << "load - Load a previous state" << std::endl
            << "duel - Create a new duel" << std::endl;
        else
        {
            if (!isDuel)
                std::cout << "save - Save the current state" << std::endl;
            std::cout << "open - Open the visual window" << std::endl
            << "slow - Change speed mode" << std::endl
            << "pause - Pause/unpause the simulation" << std::endl
            << "stop - Stop the simulation" << std::endl;
        }
        std::cout << "\n0 - Back to Main Menu" << std::endl << std::endl;
    }

    virtual Action HandleInput(std::string Input)
{
    if (Input == "0")
        return Menu::MAIN;

    if (!SimulationHandler::sSimulationHandler)
    {
        if (Input == "start")
            StartSimulation();
        else if (Input == "duel")
        {
            StartDuel();
        }
        else if (Input == "load")
            LoadState();
    }
    else
    {
        if (Input == "save" && SimulationHandler::sSimulationHandler->mSimMode != SimulationHandler::SimMode::DUEL)
            SaveState();
        else if (Input == "open")
            OpenWindow();
        else if (Input == "slow")
            ChangeSpeed();
        else if (Input == "pause")
            PauseSimulation();
        else if (Input == "stop")
            StopSimulation();
    }

    return STAY;
}
};

class InformationMenu : public Menu
{
    virtual void ToScreen()
    {
        int device;
        cudaDeviceProp deviceProp;

        cudaGetDevice(&device);
        cudaGetDeviceProperties(&deviceProp, device);

        size_t stackSize, heapSize;
        cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
        cudaDeviceGetLimit(&heapSize, cudaLimitMallocHeapSize);

        system("CLS");
        std::cout << "DEVICE INFORMATIONS\n" << std::endl
            << "Device: " << device << " - " << deviceProp.name << std::endl
            << "Total global memory: " << deviceProp.totalGlobalMem / 1'048'056 << " MB" << std::endl
            << "Shared memory / block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl
            << "Total constant memory: " << deviceProp.totalConstMem / 1024 << " KB" << std::endl
            << "Number of multiprocessors: " << deviceProp.multiProcessorCount << std::endl
            << "Concurrent managed access: " << (deviceProp.concurrentManagedAccess ? "Supported" : "Not supported") << std::endl
            << std::endl
            << "Stack size limit: " << stackSize << std::endl
            << "Heap size limit: " << heapSize << std::endl
            << std::endl
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
    case Menu::INFORMATION: MenuPtr = new InformationMenu; break;
    case Menu::STAY: //NOOP
    default: break; //NOOP
    }

}