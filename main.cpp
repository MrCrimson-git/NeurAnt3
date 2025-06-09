#include "Menu.h"

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