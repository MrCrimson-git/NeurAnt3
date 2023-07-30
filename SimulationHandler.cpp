#include "SimulationHandler.h"
#include "GlobalSettings.h"
#include <iostream>
#include "Colony.cuh"
#include "Ant.cuh"
#include <sstream>
#include <filesystem>
#include <fstream>

#define _USE_MATH_DEFINES

// C RunTime Header Files:
#include <stdlib.h>
#include <malloc.h>
#include <memory.h>
#include <wchar.h>
#include <math.h>

#include <d2d1.h>
#include <d2d1helper.h>
#include <dwrite.h>
#pragma comment(lib, "dwrite") //links dwrite.lib in a more controlled manner
#include <wincodec.h>

SimulationHandler* SimulationHandler::sSimulationHandler = NULL;

template<class Interface>
inline void SafeRelease(
    Interface **ppInterfaceToRelease)
{
    if (*ppInterfaceToRelease != NULL)
    {
        (*ppInterfaceToRelease)->Release();
        (*ppInterfaceToRelease) = NULL;
    }
}

#ifndef HINST_THISCOMPONENT
EXTERN_C IMAGE_DOS_HEADER __ImageBase;
#define HINST_THISCOMPONENT ((HINSTANCE)&__ImageBase)
#endif

SimulationHandler::SimulationHandler() : mHwnd(NULL),
    mDirect2dFactory(NULL),
    mRenderTarget(NULL),
    mBrush(NULL)
{
    cudaMalloc(&mDevice_WeightTransfer, sizeof(float) * GS::gWeightCount * (GS::gEnvironmentCount + 1));
    cudaMallocHost(&mHost_WeightTransfer, sizeof(float) * GS::gWeightCount * (GS::gEnvironmentCount + 1));
}

SimulationHandler::~SimulationHandler()
{
    StopSimulation();
    cudaFree(mDevice_WeightTransfer);
    cudaFreeHost(mHost_WeightTransfer);
    //while (mWindowThread != NULL) {} //ToDo: Any check that window thread stopped
}

void SimulationHandler::ChangeSpeed()
{
    mSlowMode ^= true;
}

void SimulationHandler::CleanUpWindowThread()
{
    if (mWindowThread)
    {
        SafeRelease(&mRenderTarget);
        SafeRelease(&mBrush);

        mWindowThread->join();
        delete mWindowThread;
        mWindowThread = NULL;
    }
    mCloseWindow = false;
}

void SimulationHandler::GenerateLoadedState()
{
    std::string path = std::format("{}-{}-{}", INPUT_COUNT, MIDDLE_COUNT, OUTPUT_COUNT);
    std::filesystem::path SaveDir("SaveFiles\\" + path);
    int maxGen = 0;
    for(auto const &file : std::filesystem::directory_iterator{SaveDir})
    {
        int gen;
        sscanf_s(file.path().filename().string().c_str(), "%*d-%d", &gen);
        if (gen > maxGen) maxGen = gen;
    }

    std::ifstream saveFile(std::format("{}/{}-{}.nan", SaveDir.string(), RAND_SEED, maxGen), std::ios::in | std::ios::binary);
    saveFile.read(reinterpret_cast<char *>(mHost_WeightTransfer), sizeof(float) * GS::gWeightCount * (GS::gEnvironmentCount + 1));
    saveFile.close();
    
    mGeneration = maxGen;

    GenerateStartingState(true);
}

void SimulationHandler::LoadState()
{
    if (!mSimulationThread)
    {
        mSimulationThread = new std::thread(&SimulationHandler::SimulationThread, this, true);
        mPaused = true;
    }
}

void SimulationHandler::OpenWindow()
{
    if (!mWindowThread)
        mWindowThread = new std::thread(&SimulationHandler::WindowThread, this);
}

void SimulationHandler::PauseSimulation()
{
    mPaused ^= true;
}

void SimulationHandler::SaveState() const
{
    cudaDeviceSynchronize();
    CUDA_CHECK

    SaveDeviceToHost();
    cudaDeviceSynchronize();
    CUDA_CHECK

    std::string path = std::format("{}-{}-{}", INPUT_COUNT, MIDDLE_COUNT, OUTPUT_COUNT);
    std::filesystem::path SaveDir("SaveFiles\\" + path);
    if (!std::filesystem::exists(SaveDir))
        try
        {
            std::filesystem::create_directory(SaveDir);
        } catch (const std::exception &ex)
        {
            std::cerr << "Error creating directory: " << ex.what() << std::endl;
        }

#ifdef BINARY_SAVE
    std::ofstream saveFile(std::format("{}/{}-{}.nan", SaveDir.string(), RAND_SEED, mGeneration), std::ios::out | std::ios::binary);
    saveFile.write(reinterpret_cast<const char *>(mHost_WeightTransfer), sizeof(float) * GS::gWeightCount * (GS::gEnvironmentCount + 1));
#else
    std::ofstream saveFile(SaveDir.string() + "/savefile.nan");
    for (int i{ 0 }; i < GS::gWeightCount * (GS::gEnvironmentCount + 1); ++i)
        saveFile << std::fixed << std::setprecision(6) << managedWeights[i] << ",";
#endif // BINARY_SAVE 

    saveFile.flush();   //Maybe fixes a rare bug?
    saveFile.close();

    cudaDeviceSynchronize();
    CUDA_CHECK
}

void SimulationHandler::StartSimulation()
{
    if (!mSimulationThread)
    {
        mSimulationThread = new std::thread(&SimulationHandler::SimulationThread, this, false);
        mPaused = true;
    }
}

void SimulationHandler::StopSimulation()
{
    //Close visual window
    if (mWindowThread)
    {
        mCloseWindow = true;
        SendMessage(mHwnd, WM_CLOSE, 0, 0);
    }

    //Stop simulation
    if (mSimulationThread)
    {
        mStopSimulation = true;
        mSimulationThread->join();
        FreeMemory();
        delete mSimulationThread;
        mSimulationThread = NULL;
    }
}

void SimulationHandler::OnRender()
{
    RECT rc;
    GetClientRect(mHwnd, &rc);

    D2D1_SIZE_U size = D2D1::SizeU(
        rc.right - rc.left,
        rc.bottom - rc.top
    );

    //Create device resources
    if (!mRenderTarget)
    {
        // Create a Direct2D render target.
        mDirect2dFactory->CreateHwndRenderTarget(D2D1::RenderTargetProperties(), D2D1::HwndRenderTargetProperties(mHwnd, size), &mRenderTarget);

        // Create the brush.
        mRenderTarget->CreateSolidColorBrush( D2D1::ColorF(D2D1::ColorF::LightSlateGray), &mBrush);
    }

    //Copy variable so it won't change during render:
    const auto antsToDraw = GV::gAntsToDraw;

    cudaMemcpyAsync(mHost_ColonyStates, mDevice_ColonyStates, sizeof(ColonyState) * GS::gColonyCount, cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(mHost_AntStates, mDevice_AntStates, sizeof(AntState) * antsToDraw, cudaMemcpyDeviceToHost);

    CUDA_CHECK

    D2D1::Matrix3x2F baseTransform(D2D1::Matrix3x2F::Translation(size.width * .5f, size.height * .5f));
    mRenderTarget->BeginDraw();
    mRenderTarget->SetTransform(baseTransform);
    mRenderTarget->Clear(D2D1::ColorF(D2D1::ColorF::White));

    D2D1_SIZE_F rtSize = mRenderTarget->GetSize();

    // Draw a grid background.
    int width = static_cast<int>(rtSize.width);
    int height = static_cast<int>(rtSize.height);

    mBrush->SetColor(D2D1::ColorF(D2D1::ColorF::LightSlateGray));

    //Drawing vertical lines
    for (int x = -(width / 40) * 20; x < width / 2; x += 20)
    {
        mRenderTarget->DrawLine(
            D2D1::Point2F(static_cast<FLOAT>(x), -rtSize.height * .5f),
            D2D1::Point2F(static_cast<FLOAT>(x), rtSize.height * .5f),
            mBrush,
            0.5f
        );
    }

    //Drawing horizontal lines
    for (int y = -(height / 40) * 20; y < height / 2; y += 20)
    {
        mRenderTarget->DrawLine(
            D2D1::Point2F(-rtSize.width * .5f, static_cast<FLOAT>(y)),
            D2D1::Point2F(rtSize.width * .5f, static_cast<FLOAT>(y)),
            mBrush,
            0.5f
        );
    }

    //Colorpicker
#if 0
    //ToDo: Not quite working
    auto getColonyColor = [](int I) -> D2D1::ColorF
    {
        float remainder;
        float ratio = modf(I * 3.f / GS::gColonyCount, &remainder);
        int firstColor = int (0xFF * (1 - ratio));
        int secondColor = int (0xFF * ratio);
        UINT32 color = firstColor * 0x100 ^ (int)remainder + secondColor * 0x100 ^ (((int) remainder + 1) % 3);
        return D2D1::ColorF(color);
    };
#endif

    //Drawing bases
    for (int i = 0; i < GS::gColonyCount; ++i)
    {
        float x = mHost_ColonyStates[i].mPosition.x;
        float y = mHost_ColonyStates[i].mPosition.y;
        mBrush->SetColor(D2D1::ColorF(i ? D2D1::ColorF::LightPink : D2D1::ColorF::LightBlue, 0.5f));
        D2D1_ELLIPSE ellipse = D2D1::Ellipse(D2D1::Point2F(x, y), 25, 25);
        mRenderTarget->FillEllipse(ellipse, mBrush);
        mBrush->SetColor(D2D1::ColorF(i ? D2D1::ColorF::Red : D2D1::ColorF::Blue));
        mRenderTarget->DrawEllipse(ellipse, mBrush);
        const std::wstring points = std::to_wstring(mHost_ColonyStates[i].mPoints);
        const D2D1_RECT_F rect = D2D1::RectF(-rtSize.width * 0.4f, 0.f, rtSize.width * 0.4f, 0.f);
        mTextFormat->SetTextAlignment(DWRITE_TEXT_ALIGNMENT(i));
        mTextFormat->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_CENTER);
        mRenderTarget->DrawText(points.c_str(), (UINT32)points.size(), mTextFormat, rect, mBrush);
    }

    //Drawing ants
    for (int i = 0; i < antsToDraw; ++i)
    {
        float x = mHost_AntStates[i].mPosition.x;
        float y = mHost_AntStates[i].mPosition.y;
        if (abs(y) > size.height * .5f || abs(x) > size.width * .5f)
            continue;
        mBrush->SetColor(D2D1::ColorF((i / 10) % 2 ? D2D1::ColorF::Red : D2D1::ColorF::Blue));
        D2D1_ELLIPSE ellipse = D2D1::Ellipse(D2D1::Point2F(0, 0), GS::gAntSize_2, GS::gAntSize_2);
        mRenderTarget->SetTransform(D2D1::Matrix3x2F::Rotation(mHost_AntStates[i].mRotation * 180) * D2D1::Matrix3x2F::Translation(x, y) * baseTransform);
        mRenderTarget->DrawLine(D2D1::Point2F(0.f, 0.f), D2D1::Point2F(GS::gAntSize_2, 0), mBrush, .7f * (float)pow(1.75f, mHost_AntStates[i].mHasFlag));
        mRenderTarget->DrawEllipse(ellipse, mBrush, (float)pow(1.75f, mHost_AntStates[i].mHasFlag));
    }

    //Drawing loading bar
    mRenderTarget->SetTransform(D2D1::Matrix3x2F::Translation(0.f, 5.f));
    mBrush->SetColor(D2D1::ColorF(D2D1::ColorF::Green));
    mRenderTarget->DrawLine(D2D1::Point2F(0.f, 0.f), D2D1::Point2F((float)size.width * mStepCounter / GS::gSimulationTime, 0.0f), mBrush, 5.0f);

    //Drawing text
    mRenderTarget->SetTransform(D2D1::Matrix3x2F::Translation(5.f, 15.f));
    mBrush->SetColor(D2D1::ColorF(D2D1::ColorF::Black));
    //RefreshStringBuffer();
    mRenderTarget->DrawText(mStrBuffer.c_str(), (UINT32)mStrBuffer.size(), mTextFormat, D2D1::RectF(0.f, 0.f, rtSize.width, rtSize.height), mBrush);

    mRenderTarget->EndDraw();
}

void SimulationHandler::OnResize(UINT width, UINT height)
{
    if (mRenderTarget)
    {
        // Note: This method can fail, but it's okay to ignore the
        // error here, because the error will be returned again
        // the next time EndDraw is called.
        mRenderTarget->Resize(D2D1::SizeU(width, height));
    }
}

void SimulationHandler::RefreshStringBuffer()
{
    static long lastTime = 0, curTime;
    curTime = std::clock();
    mStrBuffer = std::format(L"{} {} {} this thing also. Click-clock: {}", L"TEST", 2, L"test", curTime - lastTime);
    lastTime = curTime;
}

void SimulationHandler::SimulationStep()
{

    DefenderStep();
    CUDA_CHECK
    AttackerStep();
    CUDA_CHECK
    ++mStepCounter;

    if (mStepCounter == GS::gSimulationTime)
    {
        cudaDeviceSynchronize();
        Evaluate();
        CUDA_CHECK
        mStepCounter = 0;
        GenerateStartingState();
    }
    //RefreshStringBuffer();
}

void SimulationHandler::SimulationThread(bool LoadState)
{
    LoadState ? GenerateLoadedState() : GenerateStartingState();

    while (!mStopSimulation || mCloseWindow)
    {
        if (mCloseWindow)
            CleanUpWindowThread();

        if (!mPaused && !mSlowMode)
            SimulationStep();
    }
}

void SimulationHandler::WindowThread()
{
    HeapSetInformation(NULL, HeapEnableTerminationOnCorruption, NULL, 0);

    if (SUCCEEDED(CoInitialize(NULL)))
    {
        D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &mDirect2dFactory);

        DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(mDWriteFactory), reinterpret_cast<IUnknown **>(&mDWriteFactory));
        mDWriteFactory->CreateTextFormat(L"Arial", NULL, DWRITE_FONT_WEIGHT_SEMI_BOLD, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 32, L"" /*locale*/, &mTextFormat);

        WNDCLASSEX wcex = { sizeof(WNDCLASSEX) };
        wcex.style = CS_HREDRAW | CS_VREDRAW;
        wcex.lpfnWndProc = SimulationHandler::WndProc;
        wcex.cbClsExtra = 0;
        wcex.cbWndExtra = sizeof(LONG_PTR);
        wcex.hInstance = HINST_THISCOMPONENT;
        wcex.hbrBackground = NULL;
        wcex.lpszMenuName = NULL;
        wcex.hCursor = LoadCursor(NULL, IDI_APPLICATION);
        wcex.lpszClassName = L"SimulationWindow";

        RegisterClassEx(&wcex);

        mHwnd = CreateWindow(L"SimulationWindow", L"Neurant 3", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 0, 0, NULL, NULL, HINST_THISCOMPONENT, this);

        if (mHwnd)
        {
            // Because the SetWindowPos function takes its size in pixels, we
            // obtain the window's DPI, and use it to scale the window size.
            int dpi = GetDpiForWindow(mHwnd);

            SetWindowPos(mHwnd, NULL, NULL, NULL, static_cast<int>(ceil(GS::gMapSizeX * dpi / 96.f)), static_cast<int>(ceil(GS::gMapSizeY * dpi / 96.f)), SWP_NOMOVE | SWP_NOZORDER);  //TODO: Research what does what
            ShowWindow(mHwnd, SW_SHOWNORMAL);
            UpdateWindow(mHwnd); 

            MSG msg;
            msg.message = WM_NULL;
            while (msg.message != WM_QUIT)
            {
                if (mSlowMode && !mPaused)
                    SimulationStep();
                if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
                {
                    TranslateMessage(&msg);
                    DispatchMessage(&msg);
                }
                else
                {
                    OnRender();
                    ValidateRect(mHwnd, NULL);
                }
            }
        }

        CoUninitialize();
    }
}

LRESULT CALLBACK SimulationHandler::WndProc(HWND HWnd, UINT Message, WPARAM WParam, LPARAM LParam)
{
    LRESULT result = 0;

    if (Message == WM_CREATE)
    {
        LPCREATESTRUCT pcs = (LPCREATESTRUCT)LParam;
        SimulationHandler *pSimulationHandler = (SimulationHandler *)pcs->lpCreateParams;

        ::SetWindowLongPtrW(
            HWnd,
            GWLP_USERDATA,
            reinterpret_cast<LONG_PTR>(pSimulationHandler)
        );

        result = 1;
    }
    else
    {
        SimulationHandler *pSimulationHandler = reinterpret_cast<SimulationHandler *>(static_cast<LONG_PTR>(
            ::GetWindowLongPtrW(
                HWnd,
                GWLP_USERDATA
            )));

        bool wasHandled = false;

        if (pSimulationHandler)
        {
            switch (Message)
            {
            case WM_DESTROY:
            {
                pSimulationHandler->mCloseWindow = true;
                PostQuitMessage(0);
            }
            result = 1;
            wasHandled = true;
            break;

            case WM_SIZE:
            {
                UINT width = LOWORD(LParam);
                UINT height = HIWORD(LParam);
                pSimulationHandler->OnResize(width, height);
            }
            result = 0;
            wasHandled = true;
            break;

            case WM_PAINT:
            {
                pSimulationHandler->OnRender();
                ValidateRect(HWnd, NULL);
            }
            result = 0;
            wasHandled = true;
            break;

            case WM_DISPLAYCHANGE:
            {
                InvalidateRect(HWnd, NULL, FALSE);
            }
            result = 0;
            wasHandled = true;
            break;

            case WM_KEYDOWN:
                switch (WParam)
                {
                case VK_ADD:
                    GV::gAntsToDraw = min((GV::gAntsToDraw / 1000 + 1) * 1000, GS::gAllAntCount);
                    break;
                case VK_SUBTRACT:
                    GV::gAntsToDraw = max((GV::gAntsToDraw / 1000 - 1) * 1000, GS::gColonyCount * GS::gAntCount);
                }
                result = 0;
                wasHandled = true;
                break;

            }
        }

        if (!wasHandled)
        {
            result = DefWindowProc(HWnd, Message, WParam, LParam);
        }
    }

    return result;

}
