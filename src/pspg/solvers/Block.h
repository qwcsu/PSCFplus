#ifndef PSPG_BLOCK_H
#define PSPG_BLOCK_H

/*
* PSCF - Polymer Self-Consistent Field Theory
*
* Copyright 2016 - 2019, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*/

#include "Propagator.h"                   // base class argument
#include <pscf/solvers/BlockTmpl.h>       // base class template
#include <pspg/field/RDField.h>           // member
#include <pspg/field/RDFieldDft.h>        // member
#include <pspg/field/FFT.h>               // member
#include <pspg/field/FFTBatched.h>        // member
#include <pspg/field/fct.h>
#include <util/containers/FArray.h>
#include <pscf/crystal/UnitCell.h>
#include <pspg/solvers/WaveList.h>

namespace Pscf {
    template <int D> class Mesh;
}

namespace Pscf {
    namespace Pspg {

        using namespace Util;

        /**
        * Block within a branched polymer.
        *
        * Derived from BlockTmpl< Propagator<D> >. A BlockTmpl< Propagator<D> >
        * has two Propagator<D> members and is derived from BlockDescriptor.
        *
        * \ingroup Pspg_Solvers_Module
        */
        template <int D>
        class Block : public BlockTmpl< Propagator<D> >
        {

        public:

            /**
            * Generic field (base class)
            */
            typedef typename Pscf::Pspg::Propagator<D>::Field Field;

            /**
            * Monomer chemical potential field.
            */
            typedef typename Pscf::Pspg::Propagator<D>::WField WField;

            /**
            * Constrained partition function q(r,s) for fixed s.
            */
            typedef typename Pscf::Pspg::Propagator<D>::QField QField;

            // Member functions

            /**
            * Constructor.
            */
            Block();

            /**
            * Destructor.
            */
            ~Block();

            /**
            * Initialize discretization and allocate required memory.
            *
            * \param ds desired (optimal) value for contour length step
            * \param mesh spatial discretization mesh
            */
            void setDiscretization(double ds, const Mesh<D>& mesh);

            /**
            * Setup parameters that depend on the unit cell.
            *
            * \param unitCell unit cell, defining cell dimensions
            * \param waveList container for properties of recip wavevectors
            */
            void setupUnitCell(const UnitCell<D>& unitCell,
                               const WaveList<D>& wavelist);

            /**
            * Set solver for this block.
            *
            * \param w  chemical potential field for this monomer type
            */
            void setupSolver(WField const & w);

            /**
            * Initialize FFT and batch FFT classes.
            */
            void setupFCT();

            /**
            * Compute step of integration loop, from i to i+1.
            */
            void step(const cudaReal* q, cudaReal* qNew);

            /**
            * Compute unnormalized concentration for block by integration.
            *
            * Upon return, grid point r of array cField() contains the
            * integral int ds q(r,s)q^{*}(r,L-s) times the prefactor,
            * where q(r,s) is the solution obtained from propagator(0),
            * and q^{*} is the solution of propagator(1),  and s is
            * a contour variable that is integrated over the domain
            * 0 < s < length(), where length() is the block length.
            *
            * \param prefactor multiplying integral
            */
            void computeConcentration(double prefactor);

            /**
            * Compute derivatives of free energy with respect to cell parameters.
            *
            * \param waveList container for properties of recip. latt. wavevectors.
            */
            void computeStress(WaveList<D>& wavelist, double prefactor);

            /**
            * Get derivative of free energy with respect to a unit cell parameter.
            *
            * \param n  unit cell parameter index
            */
            double stress(int n);

            /**
            * Return associated spatial Mesh by reference.
            */
            Mesh<D> const & mesh() const;

            /**
            * Contour length step size.
            */
            double ds() const;

            /**
            * Number of contour length steps.
            */
            int ns() const;

            FCT<D> & fct();

            // Functions with non-dependent names from BlockTmpl< Propagator<D> >
            using BlockTmpl< Pscf::Pspg::Propagator<D> >::setKuhn;
            using BlockTmpl< Pscf::Pspg::Propagator<D> >::propagator;
            using BlockTmpl< Pscf::Pspg::Propagator<D> >::cField;
            using BlockTmpl< Pscf::Pspg::Propagator<D> >::length;
            using BlockTmpl< Pscf::Pspg::Propagator<D> >::kuhn;

            // Functions with non-dependent names from BlockDescriptor
            using BlockDescriptor::setId;
            using BlockDescriptor::setVertexIds;
            using BlockDescriptor::setMonomerId;
            using BlockDescriptor::setLength;
            using BlockDescriptor::id;
            using BlockDescriptor::monomerId;
            using BlockDescriptor::vertexIds;
            using BlockDescriptor::vertexId;
            using BlockDescriptor::length;

        private:

            cudaReal reductionH(RDField<D>& a, int size);

            // Fourier transform plan
            FFT<D> fft_;

            FCT<D> fct_;

            FFTBatched<D> fftBatched_;

            /// Stress exerted by a polymer chain of a block.
            FArray<double, 6> stress_;

            // Array of elements containing exp(-K^2 b^2 ds/6)
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1 || REPS ==0
            RDField<D> expKsq_;
#endif  
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1
            RDField<D> expKsq2_;
#endif 
#if REPS == 4 || REPS == 3 || REPS == 2 
            RDField<D> expKsq3_;
#endif 
#if REPS == 4 || REPS == 3
            RDField<D> expKsq4_;
#endif     
#if REPS == 4
            RDField<D> expKsq5_;
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1 || REPS ==0
            RDField<D> expW_;
#endif  
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1
            RDField<D> expW2_;
#endif 
#if REPS == 4 || REPS == 3 || REPS == 2 
            RDField<D> expW3_;
#endif 
#if REPS == 4 || REPS == 3
            RDField<D> expW4_;
#endif
#if REPS == 4
            RDField<D> expW5_;
#endif

            // Work array for real-space field.
            RDField<D> qr_;

#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1 || REPS ==0
            RDField<D> q1_;
#endif  
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1
            RDField<D> q2_;
#endif 
#if REPS == 4 || REPS == 3 || REPS == 2 
            RDField<D> q3_;
#endif
#if REPS == 4 || REPS == 3
            RDField<D> q4_;
#endif
#if REPS == 4
            RDField<D> q5_;
#endif

            cudaReal* d_temp_;
            cudaReal* temp_;
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1 || REPS ==0
            cudaReal* expKsq_host;
#endif  
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1
            cudaReal* expKsq2_host;
#endif 
#if REPS == 4 || REPS == 3 || REPS == 2 
            cudaReal* expKsq3_host;
#endif 
#if REPS == 4 || REPS == 3
            cudaReal* expKsq4_host;
#endif
#if REPS == 4
            cudaReal* expKsq5_host;
#endif


            /// Pointer to associated Mesh<D> object.
            Mesh<D> const * meshPtr_;

            Mesh<D> * meshPtr_ph_;

            int size_ph_;

            /// Contour length step size.
            double ds_;

            /// Number of contour length steps = # grid points - 1.
            int ns_;

            int nParams_;

        };

        // Inline member functions

        /// Get number of contour steps.
        template <int D>
        inline int Block<D>::ns() const
        {  return ns_; }

        /// Get number of contour steps.
        template <int D>
        inline double Block<D>::ds() const
        {  return ds_; }

        /// Get derivative of free energy w/ respect to a unit cell parameter.
        template <int D>
        inline double Block<D>::stress(int n)
        {  return stress_[n]; }

        /// Get Mesh by reference.
        template <int D>
        inline Mesh<D> const & Block<D>::mesh() const
        {
            UTIL_ASSERT(meshPtr_)
            return *meshPtr_;
        }

        template <int D>
        inline
        FCT<D>& Block<D>::fct()
        {
            return fct_;
        }

   #ifndef PSPG_BLOCK_TPP
   // Suppress implicit instantiation
   extern template class Block<1>;
   extern template class Block<2>;
   extern template class Block<3>;
   #endif

}
}
//#include "Block.tpp"
#endif
