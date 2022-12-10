#ifndef DPD_POLYMER_TPP
#define DPD_POLYMER_TPP

#include "DPDpolymer.h"

namespace Pscf
{
namespace Pspg
{
    namespace DPD
    {
        template <int D>
        DPDpolymer<D>::DPDpolymer()
        {
            setClassName("DPDpolymer"); 
        }

        template <int D>
        void DPDpolymer<D>::setupUnitCell(UnitCell<D> const & unitCell,
                                          const WaveList<D> & wavelist)
        {
            nParam_ = unitCell.nParameter();

            for (int j = 0; j < nBlock(); ++j)
            {
                block(j).setupUnitCell(unitCell, wavelist);
            }
        }

        template <int D>
        void DPDpolymer<D>::compute(DArray<WField> const & wFields)
        {
            int monomerId;

            for (int j = 0; j < nBlock(); ++j)
            {
                monomerId = block(j).monomerId();
                block(j).setupSolver(wFields[monomerId],N());
            }
            
            solve();
        }

        template <int D>
        void DPDpolymer<D>::computeStress(WaveList<D> & wavelist)
        {

        }

    }
}}

#endif  // !DPD_POLYMER_TPP