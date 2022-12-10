#ifndef BBCPG_SYSTEM_H
#define BBCPG_SYSTEM_H

#include <pspg/iterator/AmIterator.h>
#include <pspg/field/FieldIo.h>            // member
#include <pspg/solvers/Mixture.h>          // member
#include <pspg/solvers/WaveList.h>         // member
#include <pspg/field/RDField.h>            // typedef
#include <pspg/field/RDFieldDft.h>         // typedef

#include <pscf/crystal/Basis.h>            // member
#include <pscf/mesh/Mesh.h>                // member
#include <pscf/crystal/UnitCell.h>         // member
#include <pscf/homogeneous/Mixture.h>      // member
#include <pscf/inter/ChiInteraction.h>     // member

#include <util/param/ParamComposite.h>     // base class
#include <util/misc/FileMaster.h>          // member
#include <util/containers/DArray.h>        // member template
#include <util/containers/Array.h>         // function parameter

namespace Pscf{
namespace Pspg{
    namespace BBCpg{

        using namespace Util;

        template <int D>
        class System : public ParamComposite
        {
        public:

            System();

            ~System();

            void setOptions(int argc, char **argv);

            void readParam();

            virtual void readParam(std::istream& in);

            virtual void readParameters(std::istream& in);

            Mixture<D>& mixture();

            FileMaster& fileMaster();

        private:

            Mixture<D> mixture_;

            FileMaster fileMaster_;

            bool hasMixture_; 

        };

        template <int D>
        inline Mixture<D>& System<D>::mixture()
        { 
            return mixture_; 
        }

        template <int D>
        inline
        FileMaster& System<D>::fileMaster()
        {
            return fileMaster_;
        }



        #ifndef BBCPG_SYSTEM_TPP
        extern template class System<1>;
        extern template class System<2>;
        extern template class System<3>;
        #endif
    }
}
}



#endif // ! BBCPG_SYSTEM_H