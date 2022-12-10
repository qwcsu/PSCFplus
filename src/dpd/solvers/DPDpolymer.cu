#include "DPDpolymer.tpp"

namespace Pscf {
namespace Pspg {
    namespace DPD
    {
        using namespace Util;

        // Explicit instantiation of relevant class instances
        template class DPDpolymer<1>;
        template class DPDpolymer<2>;
        template class DPDpolymer<3>;
    }
}
}
