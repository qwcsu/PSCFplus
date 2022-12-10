#include <iostream>
#include <bbc/System.h>

int main(int argc, char **argv)
{

    Pscf::BBC::System<1> system;

    system.setOptions(argc, argv);
   
    system.readParam();
}