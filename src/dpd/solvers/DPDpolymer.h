#ifndef DPD_POLYMER_H
#define DPD_POLYMER_H

#include "DPDBlock.h"
#include "Joint.h"
#include <pspg/field/RDField.h>
#include <dpd/solvers/DPDpolymerTmpl.h>
#include <util/containers/FArray.h>
namespace Pscf
{
namespace Pspg
{
    namespace DPD
    {
        using namespace Util;

        template <int D>
        class DPDpolymer : public DPDpolymerTmpl< DPDBlock<D>, Joint<D> >
        {
        public:

            typedef DPDpolymerTmpl<DPDBlock<D>, Joint<D>> Base;

            typedef typename DPDBlock<D>::WField WField;

            DPDpolymer();

            ~DPDpolymer()=default;

            void setupUnitCell(UnitCell<D> const & unitCell, 
                               const WaveList<D>& wavelist);

            void compute(DArray<WField> const & wFields);

            void computeStress(WaveList<D>& wavelist);

            using Base::nBlock;
            using Base::block;
            using Base::ensemble;
            using Base::solve;
            using Base::N;
            
        protected:

            // protected inherited function with non-dependent names
            using ParamComposite::setClassName;

        private:

            int nBlock_;

            int nParam_;

        };
    }
}}

#endif  // !DPD_POLYMER_H