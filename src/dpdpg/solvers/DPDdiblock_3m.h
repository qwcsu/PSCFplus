#ifndef DPDDIBLOCK_3M_H
#define DPDDIBLOCK_3M_H

#include <dpdpg/solvers/Block.h>

#include <pspg/field/FieldIo.h>            // member
#include <pspg/field/RDField.h>           // member
#include <pspg/field/RDFieldDft.h>        // member
#include <pspg/solvers/WaveList.h>
#include <pspg/GpuResources.h>
#include <pspg/field/FFT.h>               // member
#include <pspg/field/FFTBatched.h>        // member

#include <pscf/crystal/UnitCell.h>
#include <pscf/chem/Monomer.h>
#include <pscf/mesh/Mesh.h>
#include <pscf/crystal/Basis.h>            // member

#include <util/param/ParamComposite.h>
#include <util/containers/DArray.h>

namespace Pscf {
namespace Pspg {
    namespace DPDpg
    {
        using namespace Util;

        template <int D>
        class DPDdiblock_3m : public ParamComposite
        {

        public:
            /**
             * Constructor.
             */
            DPDdiblock_3m();

            /**
             * Destructor
             */
            ~DPDdiblock_3m();

        private:

        };
    }
}
}

#endif