#ifndef PSPG_MIXTURE_TPP
#define PSPG_MIXTURE_TPP
// #define double float
/*
* PSCF - Polymer Self-Consistent Field Theory
*
* Copyright 2016 - 2019, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*/

#include "Mixture.h"
#include <pspg/GpuResources.h>

#include <cmath>

//in propagator.h
//static __global__ void assignUniformReal(cudaReal* result, cudaReal uniform, int size);

//theres a precision mismatch here. need to cast properly.
static __global__ void accumulateConc(cudaReal* result, double uniform, const cudaReal* cField, int size)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = startID; i < size; i += nThreads)
    {
        result[i] += (double)uniform * cField[i];
    }
}

namespace Pscf {
    namespace Pspg
    {

        template <int D>
        Mixture<D>::Mixture()
                : vMonomer_(1.0),
                  ds_(-1.0),
                  meshPtr_(0)
        {  setClassName("Mixture"); }

        template <int D>
        Mixture<D>::~Mixture()
        = default;

        template <int D>
        void Mixture<D>::readParameters(std::istream& in)
        {
            MixtureTmpl< Pscf::Pspg::Polymer<D>, Pscf::Pspg::Solvent<D> >::readParameters(in);
            vMonomer_ = 1.0; // Default value
            readOptional(in, "vMonomer", vMonomer_);
            read(in, "ns", ns_);
            // read(in, "ds", ds_);
            ds_ = 1.0/ns_;

            int total_nblock = 0;
            for(int i = 0; i < nPolymer(); ++i)
            {
                total_nblock += polymer(i).nBlock();
            }
            // std::cout << "total nblock = " << total_nblock <<std::endl;

            UTIL_CHECK(nMonomer() > 0)
            UTIL_CHECK(nPolymer()+ nSolvent() > 0)
            UTIL_CHECK(ds_ > 0)
        }

        template <int D>
        void Mixture<D>::setMesh(Mesh<D> const& mesh)
        {
            UTIL_CHECK(nMonomer() > 0)
            UTIL_CHECK(nPolymer()+ nSolvent() > 0)
            UTIL_CHECK(ds_ > 0)

            meshPtr_ = &mesh;

            // Set discretization for all blocks
            int i, j;
            for (i = 0; i < nPolymer(); ++i) {
                for (j = 0; j < polymer(i).nBlock(); ++j) {
                    polymer(i).block(j).setDiscretization(ds_, mesh);
                }
            }

        }

        template <int D>
        void Mixture<D>::setupUnitCell(const UnitCell<D>& unitCell, const WaveList<D>& wavelist)
        {
            nParams_ = unitCell.nParameter();
            for (int i = 0; i < nPolymer(); ++i) {
                polymer(i).setupUnitCell(unitCell, wavelist);
            }
        }

        /*
        * Compute concentrations (but not total free energy).
        */
        template <int D>
        void Mixture<D>::compute(DArray<Mixture<D>::WField> const & wFields,
                                 DArray<Mixture<D>::CField>& cFields)
        {
            UTIL_CHECK(meshPtr_)
            UTIL_CHECK(mesh().size() > 0)
            UTIL_CHECK(nMonomer() > 0)
            UTIL_CHECK(nPolymer() + nSolvent() > 0)
            UTIL_CHECK(wFields.capacity() == nMonomer())
            UTIL_CHECK(cFields.capacity() == nMonomer())

            int nx = 1;
            for (int d = 0; d < D; ++d)
                nx *= mesh().dimension(d);
            
            int nm = nMonomer();
            int i, j;

            // Clear all monomer concentration fields
            for (i = 0; i < nm; ++i) {
                // std::cout << "nx = " << nx << std::endl;
                // std::cout << "wsize =  " << wFields[i].capacity() << std::endl;
                UTIL_CHECK(cFields[i].capacity() == nx)
                UTIL_CHECK(wFields[i].capacity() == nx)
                //cFields[i][j] = 0.0;
                assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(cFields[i].cDField(), 0.0, nx);
            }

            // Solve MDE for all polymers
            for (i = 0; i < nPolymer(); ++i) {
                polymer(i).compute(wFields);
            }

            // Accumulate monomer concentration fields
            for (i = 0; i < nPolymer(); ++i) {
                for (j = 0; j < polymer(i).nBlock(); ++j) {
                    int monomerId = polymer(i).block(j).monomerId();
                    UTIL_CHECK(monomerId >= 0)
                    UTIL_CHECK(monomerId < nm)
                    CField& monomerField = cFields[monomerId];
                    CField& blockField = polymer(i).block(j).cField();
                    //monomerField[k] += polymer(i).phi() * blockField[k];
                    accumulateConc<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(monomerField.cDField(),
                            polymer(i).phi(), blockField.cDField(), nx);
                }
            }
            // cudaReal c_c[nx];
            // cudaMemcpy(c_c, cFields[1].cDField(), nx*sizeof(cudaReal), cudaMemcpyDeviceToHost);
            // for(int i = 0; i < 4; ++i)
            //     std::cout << c_c[i] << "\n";
        }

        /*
        * Compute Total Stress.
        */
        template <int D>
        void Mixture<D>::computeStress(WaveList<D>& wavelist)
        {
            int i, j;

            // Compute stress for each polymer.
            for (i = 0; i < nPolymer(); ++i) {
                polymer(i).computeStress(wavelist);
            }

            // Accumulate total stress
            for (i = 0; i < nParams_; ++i) {
                stress_[i] = 0.0;
                for (j = 0; j < nPolymer(); ++j) {
                    stress_[i] += polymer(j).stress(i);
                }
            }
        }

    } // namespace Pspg
} // namespace Pscf
#endif
