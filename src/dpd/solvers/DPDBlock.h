#ifndef DPD_BLOCK_H
#define DPD_BLOCK_H

#include <dpd/solvers/DPDpropagator.h>
#include "DPDBlockTmpl.h" 
#include <pspg/field/RDField.h>           
#include <pspg/field/RDFieldDft.h>        
#include <pspg/field/FFT.h>               
#include <pspg/field/FFTBatched.h>        
#include <util/containers/FArray.h>
#include <pscf/crystal/UnitCell.h>
#include <pspg/solvers/WaveList.h>

namespace Pscf {
    template <int D> class Mesh;
}

namespace Pscf{
namespace Pspg{
    namespace DPD
    {
        using namespace Util;

        template <int D>
        class DPDBlock : public DPDBlockTmpl<DPDpropagator<D>>
        {
        public:

            typedef typename Pscf::Pspg::DPD::DPDpropagator<D>::Field  Field;
            typedef typename Pscf::Pspg::DPD::DPDpropagator<D>::WField WField;
            typedef typename Pscf::Pspg::DPD::DPDpropagator<D>::QField QField;

            DPDBlock();

            ~DPDBlock();

            void setDiscretization(const Mesh<D>& mesh);

            void setupUnitCell(const UnitCell<D>& unitCell,
                               const WaveList<D>& wavelist);

            void setupSolver(WField const & w, int N);

            void setupFFT();

            void step(const cudaReal* q, cudaReal* qNew);

            void computeConcentration(double prefactor);

            RDField<D> expW();

            RDField<D> expWp();

            // void computeStress(WaveList<D>& wavelist);

            Mesh<D> const & mesh() const;

            using DPDBlockDescriptor::length;

            using DPDBlockTmpl<Pscf::Pspg::DPD::DPDpropagator<D>>::isJoint;

            using DPDBlockTmpl<Pscf::Pspg::DPD::DPDpropagator<D>>::kuhn;

            using DPDBlockTmpl<Pscf::Pspg::DPD::DPDpropagator<D>>::propagator;

            using DPDBlockTmpl<Pscf::Pspg::DPD::DPDpropagator<D>>::cField;

        private:
            
            FFT<D> fft_;

            // FFTBatched<D> fftBatched_;
            
            Mesh<D> const * meshPtr_;

            IntVec<D> kMeshDimensions_;

            int kSize_;

            RDField<D> expKsq_;

            cudaReal* expKsq_host;

            RDField<D> expW_; // exp(-w/N)

            RDField<D> expWp_; // exp(+w/N)

            RDField<D> qr_;

            RDFieldDft<D> qk_;

            int nParams_;
            
        };

        template <int D>
        inline
        Mesh<D> const & DPDBlock<D>::mesh() const
        {
            UTIL_ASSERT(meshPtr_);
            return *meshPtr_;
        }

        template <int D>
        inline
        RDField<D> DPDBlock<D>::expW()
        {
            return expW_;
        }

        template <int D>
        inline
        RDField<D> DPDBlock<D>::expWp()
        {
            return expWp_;
        }

#ifndef DPD_BLOCK_TPP
extern template class DPDBlock<1>;
extern template class DPDBlock<2>;
extern template class DPDBlock<3>;
#endif 
    }
}}

#endif