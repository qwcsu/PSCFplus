#ifndef PSPG_BOTTLEBRUSH_H
#define PSPG_BOTTLEBRUSH_H

#include "BottlebrushTmpl.h"

#include <pspg/solvers/Polymer.h>
#include <pspg/solvers/Block.h>

namespace Pscf{
namespace Pspg{
    
        using namespace Util;

        template <int D>
        class Bottlebrush : public BottlebrushTmpl<Block<D>>
        {

        public:

            Bottlebrush();

            ~Bottlebrush();

        };
        
    
}
}

#endif // !PSPG_BOTTLEBRUSH_H

