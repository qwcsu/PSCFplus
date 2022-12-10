#ifndef DPD_JOINT_TPP
#define DPD_JOINT_TPP

#include "Joint.h"

namespace Pscf
{
namespace Pspg
{
    namespace DPD
    {
        using namespace Util;

        template <int D>
        Joint<D>::Joint()
         : meshPtr_(0),
           kMeshDimensions_(0)
        {
            // propagator(0).setBlock(*this);
            // propagator(1).setBlock(*this);
        }

        template <int D>
        Joint<D>::~Joint()
        {
            //...
        }
    }
}
}

#endif