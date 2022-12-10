#include "DPDpropagator.tpp"

namespace Pscf {
namespace Pspg {
    namespace DPD{

        using namespace Util;

        // Explicit instantiation of relevant class instances
        template class DPDpropagator<1>;
        template class DPDpropagator<2>;
        template class DPDpropagator<3>;
    }
}
}
