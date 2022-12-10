#ifndef DPDPG_AM_ITERATOR_H
#define DPDPG_AM_ITERATOR_H


#include <dpdpg/iterator/Iterator.h>
#include <pspg/solvers/Mixture.h>
#include <pscf/math/LuSolver.h>
#include <util/containers/DArray.h>
#include <util/containers/DMatrix.h>
#include <util/containers/RingBuffer.h>
#include <pspg/iterator/HistMat.h>
#include <pspg/field/RDField.h>
#include <util/containers/FArray.h>


namespace Pscf{
namespace Pspg{
    namespace DPDpg
    {
        using namespace Util;

        template <int D>
        class AmIterator : public Iterator<D>
        {
            
        public:

            /**
             * Default constructor
             */
            AmIterator();

            explicit AmIterator(System<D>* system);

            /**
             * Destructor
             */
            ~AmIterator();

            
            /**
             * Read all parameters and initialize.
             *
             * \param in input filestream
             */
            void readParameters(std::istream& in);

            double epsilon();

            int maxHist();

            int maxItr();

            void allocate();

            int solve();

            int solve_3m();

            void computeDeviation();

            void computeDeviation_incmp();

            bool isConverged(int itr);

            int minimizeCoeff(int itr);

            int minimizeCoeff_incmp(int itr);
            
            void buildOmega(int itr);

            void buildOmega_incmp(int itr);

            double final_error;

        private:

            int maxItr_;
            double epsilon_;
            int maxHist_;
            int nHist_;
            bool isFlexible_;
            double lambda_;

            RingBuffer< DArray < RDField<D> > > devHists_;
            RingBuffer< DArray < RDField<D> > > omHists_;

            RingBuffer< FArray <double, 6> > devCpHists_;
            RingBuffer< FSArray<double, 6> > CpHists_;


            FSArray<double, 6> cellParameters_;

            /// Umn, matrix to be minimized
            DMatrix<double> invertMatrix_;

            /// Cn, coefficient to convolute previous histories with
            DArray<double> coeffs_;

            DArray<double> vM_;

            /// bigW, blended omega fields
            DArray<RDField<D> > wArrays_;

            /// bigWcP, blended parameter
            FArray <double, 6> wCpArrays_;

            /// bigDCp, blened deviation parameter. new wParameter = bigWCp + lambda * bigDCp
            FArray <double, 6> dCpArrays_;

            /// bigD, blened deviation fields. new wFields = bigW + lambda * bigD
            DArray<RDField<D> > dArrays_;

            DArray<RDField<D> > tempDev;

            HistMat <cudaReal> histMat_;

            cudaReal innerProduct(const RDField<D>& a, const RDField<D>& b, int size);
            cudaReal reductionH(RDField<D>& a, int size);
            cudaReal* d_temp_;
            cudaReal* temp_;

            using Iterator<D>::setClassName;
            using Iterator<D>::systemPtr_;
            using ParamComposite::read;
            using ParamComposite::readOptional;
        };

        template <int D>
        inline  
        double AmIterator<D>::epsilon()
        {
            return epsilon_;
        }

        template <int D>
        inline
        int AmIterator<D>::maxHist()
        {
            return maxHist_;
        }

        template <int D>
        inline
        int AmIterator<D>::maxItr()
        {
            return maxItr_;
        }

        #ifndef DPDPG_AM_ITERATOR_TPP
        // Suppress implicit instantiation
        extern template class AmIterator<1>;
        extern template class AmIterator<2>;
        extern template class AmIterator<3>;
        #endif
    }
}
}

#endif