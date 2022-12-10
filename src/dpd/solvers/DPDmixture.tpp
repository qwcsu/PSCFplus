#ifndef PSPG_MIXTURE_TPP
#define PSPG_MIXTURE_TPP
// #define double float
/*
* PSCF - Polymer Self-Consistent Field Theory
*
* Copyright 2016 - 2019, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*/

#include "DPDmixture.h"
#include <pspg/GpuResources.h>

#include <cmath>

static __global__ void accumulateConc(cudaReal* result, double uniform, const cudaReal* cField, int size)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = startID; i < size; i += nThreads)
    {
        result[i] += (double)uniform * cField[i];
    }
}

namespace Pscf
{
namespace Pspg
{
    namespace DPD
    {
        template <int D>
        DPDmixture<D>::DPDmixture()
        {  setClassName("DPDmixture"); }

        template <int D>
        void DPDmixture<D>::readParameters(std::istream& in)
        {
            MixtureTmpl< Pscf::Pspg::DPD::DPDpolymer<D>, 
                         Pscf::Pspg::Solvent<D> >::readParameters(in);
            
            double kuhn;
            int monomerId;
            for (int i = 0; i < nPolymer(); ++i)
            {
                for (int j = 0; j < polymer(i).nJoint(); ++j)
                {
                    monomerId = polymer(i).joint(j).monomerId();
                    kuhn = monomer(monomerId).step();
                    polymer(i).joint(j).setKuhn(kuhn);
                }
            }

            read(in, "sigma", sigma_);
            read(in, "kappa", kappa_);

            int total_nblock = 0;
            for(int i = 0; i < nPolymer(); ++i)
            {
                total_nblock += polymer(i).nBlock();
            }

            UTIL_CHECK(nMonomer() > 0)
            UTIL_CHECK(nPolymer()+ nSolvent() > 0)
            UTIL_CHECK(sigma_ > 0)
            UTIL_CHECK(kappa_ > 0)
        }

        template <int D>
        void DPDmixture<D>::setMesh(Mesh<D> const & mesh)
        {
            UTIL_CHECK(nMonomer() > 0)
            UTIL_CHECK(nPolymer()+ nSolvent() > 0)

            meshPtr_ = &mesh;

            // Set discretization for all blocks
            for (int i = 0; i < nPolymer(); ++i)
            {
                for (int j = 0; j < polymer(i).nBlock(); ++j)
                {
                    polymer(i).block(j).setDiscretization(mesh);
                }
            }
        }

        template <int D>
        void DPDmixture<D>::setupUnitCell(const UnitCell<D> & unitCell,
                                          const WaveList<D> & wavelist)
        {
            nParams_ = unitCell.nParameter();
            for (int i = 0; i < nPolymer(); ++i)
            {
                polymer(i).setupUnitCell(unitCell, wavelist);
            }
        }

        template <int D>
        void DPDmixture<D>::compute(DArray<WField> const & wFields, 
                                    DArray<CField>& cFields)
        {
            UTIL_CHECK(meshPtr_)
            UTIL_CHECK(mesh().size() > 0)
            UTIL_CHECK(nMonomer() > 0)
            UTIL_CHECK(nPolymer() + nSolvent() > 0)
            UTIL_CHECK(wFields.capacity() == nMonomer())
            UTIL_CHECK(cFields.capacity() == nMonomer())

            int nx = mesh().size();
            int nm = nMonomer();
            int i, j;

            for (i = 0; i < nm; ++i) 
            {
                UTIL_CHECK(cFields[i].capacity() == nx)
                UTIL_CHECK(wFields[i].capacity() == nx)
                //cFields[i][j] = 0.0;
                assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (cFields[i].cDField(), 0.0, nx);
            }

            for (i = 0; i < nPolymer(); ++i) 
            {
                polymer(i).compute(wFields);
            }

            // Accumulate monomer concentration fields
            for (i = 0; i < nPolymer(); ++i) 
            {
                for (j = 0; j < polymer(i).nBlock(); ++j) 
                {
                    int monomerId = polymer(i).block(j).monomerId();
                    UTIL_CHECK(monomerId >= 0)
                    UTIL_CHECK(monomerId < nm)
                    CField& monomerField = cFields[monomerId];
                    CField& blockField = polymer(i).block(j).cField();
                    accumulateConc<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                    (monomerField.cDField(),
                    polymer(i).phi(), 
                    blockField.cDField(), 
                    nx);
                }
            }
        }

        template <int D>
        void DPDmixture<D>::computeStress(WaveList<D>& wavelist)
        {
        }
    }
}
}

#endif


