#ifndef DPD_BLOCK_TMPL_H
#define DPD_BLOCK_TMPL_H

#include <pscf/chem/DPDBlockDescriptor.h>
#include <util/containers/Pair.h>   

#include <cmath>

namespace Pscf
{
namespace Pspg
{
    namespace DPD
    {
        using namespace Util;

        template <class TP>
        class DPDBlockTmpl : public DPDBlockDescriptor
        {
        public:

            typedef TP DPDpropagator;

            typedef typename TP::CField CField;

            typedef typename TP::WField WField;

            DPDBlockTmpl();

            virtual ~DPDBlockTmpl();

            virtual void setKuhn (double kuhn);

            virtual void setJoint ();

            TP& propagator(int directionId);

            TP const & propagator(int directionId) const;

            typename TP::CField& cField();

            double kuhn() const;

            bool isJoint() const;

        private:

            Pair<DPDpropagator> propagators_;

            CField cField_;

            double kuhn_;

            bool isJoint_;
        };

        template <class TP>
        DPDBlockTmpl<TP>::DPDBlockTmpl()
                : propagators_(),
                  cField_(),
                  kuhn_(0.0),
                  isJoint_(false)
        {
            propagator(0).setDirectionId(0);
            propagator(1).setDirectionId(1);
            propagator(0).setPartner(propagator(1));
            propagator(1).setPartner(propagator(0));
        }

        template <class TP>
        DPDBlockTmpl<TP>::~DPDBlockTmpl() = default;

        /// non-inline
        template <class TP>
        void DPDBlockTmpl<TP>::setKuhn (double kuhn)
        {
            kuhn_ = kuhn;
        }

        template <class TP>
        void DPDBlockTmpl<TP>::setJoint ()
        {
            if (this->length() == -1)
            {
                isJoint_ = true;
                this->setLength(1);
            }    
        }
        
        /// inline
        template <class TP>
        inline
        TP & DPDBlockTmpl<TP>::propagator(int directionId)
        {
            return propagators_[directionId];
        }

        template <class TP>
        inline
        TP const & DPDBlockTmpl<TP>::propagator(int directionId) const
        {
            return propagators_[directionId];
        }

        template <class TP>
        inline
        typename TP::CField & DPDBlockTmpl<TP>::cField()
        {
            return cField_;
        }

        template <class TP>
        inline
        double DPDBlockTmpl<TP>::kuhn() const
        {
            return kuhn_;
        }
        
        template <class TP>
        inline
        bool DPDBlockTmpl<TP>::isJoint() const
        {
            return isJoint_;
        }

    }
}
}

#endif
