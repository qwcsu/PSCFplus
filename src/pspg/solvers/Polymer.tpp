#ifndef PSPG_POLYMER_TPP
#define PSPG_POLYMER_TPP

/*
* PSCF - Polymer Self-Consistent Field Theory
*
* Copyright 2016 - 2019, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*/

#include "Polymer.h"
#include <pspg/GpuResources.h>

namespace Pscf {
    namespace Pspg {

        template <int D>
        Polymer<D>::Polymer()
        {  setClassName("Polymer"); }

        template <int D>
        Polymer<D>::~Polymer()
        = default;

        template <int D>
        void Polymer<D>::setPhi(double phi)
        {
            UTIL_CHECK(ensemble() == Species::Closed)
            UTIL_CHECK(phi >= 0.0)
            UTIL_CHECK(phi <= 1.0)
            phi_ = phi;
        }

        template <int D>
        void Polymer<D>::setMu(double mu)
        {
            UTIL_CHECK(ensemble() == Species::Open)
            mu_ = mu;
        }

        /*
        * Set unit cell dimensions in all solvers.
        */
        template <int D>
        void Polymer<D>::setupUnitCell(UnitCell<D> const & unitCell, const WaveList<D>& wavelist)
        {
            nParams_ = unitCell.nParameter();
            for (int j = 0; j < nBlock(); ++j) {
                block(j).setupUnitCell(unitCell, wavelist);
            }
        }

        /*
        * Compute solution to MDE and concentrations.
        */
        template <int D>
        void Polymer<D>::compute(DArray<WField> const & wFields)
        {
            // Setup solvers for all blocks
            int monomerId;
            for (int j = 0; j < nBlock(); ++j) {
                monomerId = block(j).monomerId();
                block(j).setupSolver(wFields[monomerId]);
            }

            // Call generic solver() method base class template.
            solve();
        }

        /*
        * Compute stress from a polymer chain.
        */

        template <int D>
        void Polymer<D>::computeStress(WaveList<D>& wavelist)
        {
            double prefactor;

            // Initialize stress_ to 0
            for (int i = 0; i < nParams_; ++i)
            {
                stress_ [i] = 0.0;
            }
            
            for (int i = 0; i < nBlock(); ++i)
            {
                prefactor = exp(mu_)/length();
//                std::cout << "prefactor= "<< prefactor <<"\n";
                block(i).computeStress(wavelist, prefactor);

                for (int j=0; j < nParams_; ++j)
                {
                    stress_ [j] += block(i).stress(j);
//                     std::cout<<"nParams_ "<<nParams_<<std::endl;
                }
            }
            // exit(1);
        }

    }
}

#endif
