#include "AmIterator.tpp"

namespace Pscf {
namespace Pspg {
    namespace DPD
    {
        using namespace Util;

        // Explicit instantiation of relevant class instances
        template class AmIterator<1>;
        template class AmIterator<2>;
        template class AmIterator<3>;
    }
}
}
