#include "System.tpp"

namespace Pscf {
namespace Pspg {
    namespace DPD
    {
        using namespace Util;

        // Explicit instantiation of relevant class instances
        template class System<1>;
        template class System<2>;
        template class System<3>;
    }
}
}
