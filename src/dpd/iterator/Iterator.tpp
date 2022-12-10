#include "Iterator.h"

namespace Pscf{
namespace Pspg{
    namespace DPD
    {
        using namespace Util;
        
        template <int D>
        Iterator<D>::Iterator()
        {
            setClassName("Iterator");
        }

        template <int D>
        Iterator<D>::Iterator(System<D>* system)
        : systemPtr_(system)
        {
            setClassName("Iterator");
        }

        template <int D>
        Iterator<D>::~Iterator()
        {}
        
    }
}
}