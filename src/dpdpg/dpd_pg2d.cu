#include <iostream>
#include "System.h"

int main(int argc, char **argv)
{
    Pscf::Pspg::DPDpg::System<2> sys2d;

    sys2d.setOptions(argc, argv);

    sys2d.readParam();

    sys2d.readCommands();

    return 0;
}