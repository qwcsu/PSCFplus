#ifndef DPD_JOINT_H
#define DPD_JOINT_H

#include <dpd/solvers/DPDpropagator.h>
#include "JointTmpl.h" 
#include <pspg/field/RDField.h>           
#include <pspg/field/RDFieldDft.h>        
#include <pspg/field/FFT.h>               
#include <pspg/field/FFTBatched.h>        
#include <util/containers/FArray.h>
#include <pscf/crystal/UnitCell.h>
#include <pspg/solvers/WaveList.h>


namespace Pscf
{
    template<int D> class Mesh;
}

namespace Pscf
{
namespace Pspg
{
    namespace DPD
    {
        using namespace Util;
        
        template <int D>
        class Joint : public JointTmpl<DPDpropagator<D>>
        {

        public:

            typedef typename Pscf::Pspg::DPD::DPDpropagator<D>::QField QField;

            Joint();

            ~Joint();

            /// inline

            double stress(int n);

            Mesh<D> const & mesh() const;
        

        private:

            FFT<D> fft_;

            FArray<double, 6> stress_;

            RDField<D> Psi_;

            RDField<D> qr_;

            RDFieldDft<D> qk_;

            Mesh<D> const * meshPtr_;

            IntVec<D> kMeshDimensions_;

            int kSize_;

            int nParams_;

        };

        template <int D>
        inline
        double Joint<D>::stress(int n)
        {
            return stress_[n];
        }

        template <int D>
        inline
        Mesh<D> const & Joint<D>::mesh() const
        {
            UTIL_ASSERT(meshPtr_)
            return *meshPtr_;
        }

    #ifndef DPD_JOINT_TPP
    // Suppress implicit instantiation
    extern template class Joint<1>;
    extern template class Joint<2>;
    extern template class Joint<3>;
    #endif
        
    }
   
    
} 
}

#endif
