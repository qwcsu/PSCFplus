#include "DPDmixture.tpp"

namespace Pscf {
namespace Pspg {
    namespace DPD
    {
        using namespace Util;

        // Explicit instantiation of relevant class instances
        template class DPDmixture<1>;
        template class DPDmixture<2>;
        template class DPDmixture<3>;
    }
}
}
