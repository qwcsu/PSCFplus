#include "System.tpp"

namespace Pscf {
namespace Pspg {
    namespace BBCpg{
        using namespace Util;

        // Explicit instantiation of relevant class instances
        template class System<1>;
        template class System<2>;
        template class System<3>;

    }
}
}
