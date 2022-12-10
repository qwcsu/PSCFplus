#ifndef PSCF_LINEAR_BLOCK_TMPL_H
#define PSCF_LINEAR_BLOCK_TMPL_H

#include <pscf/chem/BlockDescriptor.h> // base class
#include <util/containers/Pair.h>      // member template

#include <cmath>

namespace Pscf
{
    using namespace Util;

    template <class TP> 
    class LinearBlockTmpl : public BlockDescriptor
    {

    public:

        typedef TP Propagator;

        typedef typename TP::CField CField;

        typedef typename TP::WField WField;

        LinearBlockTmpl();

        virtual ~LinearBlockTmpl() = default;

        virtual void setKuhn(double kuhn);

        TP& propagator(int directionId);

        TP const & propagator(int directionId) const;

        typename TP::CField& cField();

        double kuhn() const;

    private:

        Pair<Propagator> propagators_;

        CField cField_;

        double kuhn_;
    }; 

    // Inline member functions
    template <class TP>
    inline
    TP& LinearBlockTmpl<TP>::propagator(int directionId)
    {
        return propagators_[directionId];
    }

    template <class TP>
    inline
    TP const & LinearBlockTmpl<TP>::propagator(int directionId) const
    {  
        return propagators_[directionId]; 
    }

    template <class TP>
    inline double LinearBlockTmpl<TP>::kuhn() const
    {  
        return kuhn_; 
    }

    template <class TP>
    inline
    typename TP::CField& LinearBlockTmpl<TP>::cField()
    {  
        return cField_; 
    }

    // Non-inline functions
    template <class TP>
    LinearBlockTmpl<TP>::LinearBlockTmpl()
     : propagators_(),
       cField_(),
       kuhn_(0.0)
    {
        propagator(0).setDirectionId(0);
        propagator(1).setDirectionId(1);
        propagator(0).setPartner(propagator(1));
        propagator(1).setPartner(propagator(0));
    }

    template <class TP>
    void LinearBlockTmpl<TP>::setKuhn(double kuhn)
    {  
        kuhn_ = kuhn; 
    }
}


#endif