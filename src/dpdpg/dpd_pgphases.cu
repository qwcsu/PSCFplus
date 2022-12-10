#include <iostream>
#include "System.h"
#include "Ridder.h"
#include "Phases.h"

using namespace Pscf::Pspg::DPDpg;

int main(int argc, char **argv)
{
    System<3> sys1;

    sys1.setOptions(argc, argv);

    sys1.readParam();

    sys1.readCommands();
    
    Phases<System<3>, System<3>> phs(sys1, sys1, argc, argv);

    // Ridder<Phases<System<3>, System<3>>> r(50, 1e-5);   

    // double root = r.solve(phs, -1.0, 1.0); 

    // std::cout << std::endl
    //           << "The root is " 
    //           << root << std::endl;

    return 0;
}