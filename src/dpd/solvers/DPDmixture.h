#ifndef DPD_MIXTURE_H
#define DPD_MIXTURE_H

#include "DPDpolymer.h"
#include <pspg/solvers/Solvent.h>
#include <pscf/solvers/MixtureTmpl.h>
#include <pscf/inter/Interaction.h>
#include <util/containers/DArray.h>

namespace Pscf {
    template <int D> class Mesh;
}

namespace Pscf{
namespace Pspg{
    namespace DPD{
        template <int D> 
        class DPDmixture : public MixtureTmpl<DPDpolymer<D>, Solvent<D>>
        {
        public:

            typedef typename DPDpropagator<D>::WField WField;

            typedef typename DPDpropagator<D>::CField CField;

            DPDmixture();

            ~DPDmixture()=default;

            void readParameters(std::istream& in);

            void setMesh(Mesh<D> const & mesh);

            void setupUnitCell(const UnitCell<D>& unitCell, 
                               const WaveList<D>& wavelist);

            void compute(DArray<WField> const & wFields, 
                         DArray<CField>& cFields);

            void computeStress(WaveList<D>& wavelist);

            double stress(int n)
            {  return stress_[n]; }


            using MixtureTmpl< Pscf::Pspg::DPD::DPDpolymer<D>, 
                               Pscf::Pspg::Solvent<D> >::nPolymer;

            using MixtureTmpl< Pscf::Pspg::DPD::DPDpolymer<D>, 
                               Pscf::Pspg::Solvent<D> >::nSolvent;

            using MixtureTmpl< Pscf::Pspg::DPD::DPDpolymer<D>, 
                               Pscf::Pspg::Solvent<D> >::nMonomer;

            using MixtureTmpl< Pscf::Pspg::DPD::DPDpolymer<D>, 
                               Pscf::Pspg::Solvent<D> >::polymer;

            using MixtureTmpl< Pscf::Pspg::DPD::DPDpolymer<D>, 
                               Pscf::Pspg::Solvent<D> >::monomer;

        protected:

            using MixtureTmpl< Pscf::Pspg::DPD::DPDpolymer<D>, 
                               Pscf::Pspg::Solvent<D> >::setClassName;
            using ParamComposite::read;
            using ParamComposite::readOptional;

        private:

            double sigma_;

            double kappa_;

            Mesh<D> const * meshPtr_;
            
            Mesh<D> const & mesh() const;

            int nParams_;

            FArray<double, 6> stress_;
        };

        template <int D>
        inline Mesh<D> const & DPDmixture<D>::mesh() const
        {
            UTIL_ASSERT(meshPtr_)
            return *meshPtr_;
        }
    #ifndef PSPG_MIXTURE_TPP
    // Suppress implicit instantiation
    extern template class DPDmixture<1>;
    extern template class DPDmixture<2>;
    extern template class DPDmixture<3>;
    #endif
    }
}
}

#endif