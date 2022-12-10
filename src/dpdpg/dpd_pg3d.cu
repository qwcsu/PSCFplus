#include <iostream>
#include "System.h"

int main(int argc, char **argv)
{
    Pscf::Pspg::DPDpg::System<3> sys3d;

    sys3d.setOptions(argc, argv);

    sys3d.readParam();

    sys3d.readCommands();

    return 0;
}