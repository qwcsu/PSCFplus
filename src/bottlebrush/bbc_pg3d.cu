#include <iostream>
#include <bottlebrush/System.h>

int main(int argc, char **argv)
{
    Pscf::Pspg::BBCpg::System<3> bbc_system;

    bbc_system.setOptions(argc, argv);

    bbc_system.readParam();

    return 0;
}