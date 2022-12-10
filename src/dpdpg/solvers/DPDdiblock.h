#ifndef DPDDIBLOCK_H
#define DPDDIBLOCK_H

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
        class DPDdiblock : public ParamComposite
        {

        public:

            /**
             * Constructor.
             */
            DPDdiblock();

            /**
             * Destructor
             */
            ~DPDdiblock();

            /**
             * Read parameters from file and initialize.
             *
             * \param in input parameter file
             */
            virtual void readParameters(std::istream& in);

            void setMesh(Mesh<D> const& mesh);

            void setupUnitCell(const UnitCell<D>& unitCell,
                               const WaveList<D>& wavelist);
            
            void allocate(Basis<D> const& basis);

            void setBasis(Basis<D> const& basis, UnitCell<D>& unitCell);

            void compute(DArray<RDField<D>> const & wFields, DArray<RDField<D>>& cFields);

            void computeStress(Basis<D> const& basis, UnitCell<D> & unitCell, FieldIo<D> & fieldIo,
                               DArray<RDField<D>>& cFields);

            void computeStress_incmp(Basis<D> const& basis, UnitCell<D> & unitCell, FieldIo<D> & fieldIo,
                                     DArray<RDField<D>>& cFields);

            /**
             * Get number of monomer types.
             */
            int nMonomer() const;

            /**
             * Get number of block.
             */
            int nBlock() const;

            /**
             * Get number of cell params.
             */
            int nParam() const;

            /**
             * Get length of copolymer
             */
            int getN() const;

            int getNA() const;

            double sigma() const;

            double kpN();

            double chiN();

            Mesh<D> const & mesh() const;

            double eps() const;

            RDField<D> & bu0(); 

            double Q() const;

            double * q();

            double * qt();

            double stress(int n)
            {  return stress_[n]; }

            void setChi(double chi);

            void setEps(double eps);

            bool compressibility();

            double idemp(int i, int j); 

        private:

            // Fourier transform plan
            FFT<D> fft_;

            FFTBatched<D> fftBatched_;

            /// Stress exerted by a polymer chain of a block.
            FArray<double, 6> stress_;

            RDField<D> expA_;
            RDField<D> expAB_;
            RDField<D> expB_;
            RDField<D> bu0_;

            RDField<D> expWA_;
            RDField<D> expWB_;
            RDField<D> expWAp_;
            RDField<D> expWBp_;

            cudaReal* qFields_d;
            cudaReal* qtFields_d;

            double *q_;
            double *qt_;

            RDField<D> qr_;
            RDFieldDft<D> qk_;

            cudaReal* expA_host;
            cudaReal* expAB_host;
            cudaReal* expB_host;
            cudaReal* bu0_host;

            double Q_;

            FieldIo<D> fieldIo_;

            /**
             * Array of monomer type descriptors.
             */
            DArray<Monomer> monomers_;

            /**
             * Array of block descriptors.
             */
            DArray<Block> blocks_;

            /**
             * Number of monomer types.
             */
            int nMonomer_;

            /**
             * Number of monomer types.
             */
            int nBlock_;

            /**
             * Length of copolymer
             */
            int N_;

            int NA_;

            double sigma_;

            double kpN_;

            double chiN_;

            double eps_;

            bool comp_;

            Mesh<D> const * meshPtr_;

            IntVec<D> kMeshDimensions_;

            Basis<D> const * basisPtr_;

            int kSize_;

            int nParams_;

            cudaReal *dk_d, *dbu0_d, *dPhiA_d, *dPhiAB_d, *dPhiB_d;

            DMatrix<double> chiInverse_;

            DMatrix<double> idemp_;
        };

        template <int D>
        inline 
        int DPDdiblock<D>::nMonomer() const
        {
            return nMonomer_;
        }

        template <int D>
        inline 
        int DPDdiblock<D>::nBlock() const
        {
            return nBlock_;
        }

        template <int D>
        inline 
        int DPDdiblock<D>::nParam() const
        {
            return nParams_;
        }

        template <int D>
        inline 
        int DPDdiblock<D>::getN() const
        {
            return N_;
        }

        template <int D>
        inline 
        int DPDdiblock<D>::getNA() const
        {
            return NA_;
        }

        template <int D>
        inline 
        double DPDdiblock<D>::sigma() const
        {
            return sigma_;
        }

        template <int D>
        inline 
        double DPDdiblock<D>::kpN()
        {
            return kpN_;
        }

        template <int D>
        inline 
        double DPDdiblock<D>::chiN()
        {
            return chiN_;
        }

        template <int D>
        inline
        Mesh<D> const & DPDdiblock<D>::mesh() const
        {
            UTIL_ASSERT(meshPtr_)
            return *meshPtr_;
        }

        template <int D>
        inline
        double DPDdiblock<D>::eps() const
        {
            return eps_;
        }

        template <int D>
        inline
        RDField<D>& DPDdiblock<D>::bu0()
        {
            return bu0_;
        }
        
        template <int D>
        inline
        double DPDdiblock<D>::Q() const
        {
            return Q_;
        }

        template <int D>
        inline
        double* DPDdiblock<D>::q() 
        {
            int NA = this->getNA();
            int nx = this->mesh().size();
            cudaMemcpy(q_, qFields_d + (NA - 1)*nx, nx * sizeof(cudaReal),cudaMemcpyDeviceToHost);
            return q_;
        }
        
        template <int D>
        inline
        double* DPDdiblock<D>::qt() 
        {
            int NA = this->getNA();
            int N = this->getN();
            int nx = this->mesh().size();
            cudaMemcpy(qt_, qtFields_d + (N - NA)*nx, nx * sizeof(double),cudaMemcpyDeviceToHost);
            return qt_;
        }

        template <int D>
        inline 
        double DPDdiblock<D>::idemp(int i, int j)
        {  
            return idemp_(i, j); 
        }

        template <int D>
        void DPDdiblock<D>::setChi(double chiN)
        {
            chiN_ = chiN;
        }

        template <int D>
        void DPDdiblock<D>::setEps(double eps)
        {
            eps_ = eps;
        }

        template <int D>
        bool DPDdiblock<D>::compressibility()
        {
            return comp_;
        }

        #ifndef DPDDIBLOCK_TPP
        // Suppress implicit instantiation
        extern template class DPDdiblock<1>;
        extern template class DPDdiblock<2>;
        extern template class DPDdiblock<3>;
        #endif
    }
}
}

#endif