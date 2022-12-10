#include "System.h"

int main(int argc, char **argv)
{
    Pscf::Pspg::DPD::System<3> sys3d;

    sys3d.setOptions(argc, argv);

    sys3d.readParam();

    sys3d.readCommands();

    return 0;
}