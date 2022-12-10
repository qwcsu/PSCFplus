#include "DPDBlock.tpp"

namespace Pscf {
namespace Pspg {
    namespace DPD{

        using namespace Util;

        // Explicit instantiation of relevant class instances
        template class DPDBlock<1>;
        template class DPDBlock<2>;
        template class DPDBlock<3>;
    }
}
}
