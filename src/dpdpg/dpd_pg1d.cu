#include <iostream>
#include "System.h"

int main(int argc, char **argv)
{
    Pscf::Pspg::DPDpg::System<1> sys1d;

    sys1d.setOptions(argc, argv);

    sys1d.readParam();

    sys1d.readCommands();

    return 0;
}

