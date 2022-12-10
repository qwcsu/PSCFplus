#ifndef BBC_MIXTURE_H
#define BBC_MIXTURE_H

#include "Polymer.h"
#include <pspg/solvers/Solvent.h>
#include <pscf/solvers/MixtureTmpl.h>

namespace Pscf
{
    namespace BBC
    {
        template <int D>
        class Mixture 
        {

        public:

            Mixture()  = default;
            ~Mixture() = default;
        };
    }
}

#endif