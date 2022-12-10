#ifndef DPD_JOINT_TMPL_H
#define DPD_JOINT_TMPL_H

#include <pscf/chem/JointDescriptor.h>

#include <cmath>

namespace Pscf
{
    using namespace Util;

    template <class TP>
    class JointTmpl : public JointDescriptor
    {

    public: 

        typedef TP Propagator;

        JointTmpl();

        virtual ~JointTmpl();

        virtual void setKuhn (double kuhn);

        TP & propagator (int directionId);

        TP const & propagator (int directionId) const;

        double kuhn() const;

    private:

        Pair<Propagator> propagators_;

        double kuhn_;

    };

    template <class TP>
    inline
    TP & JointTmpl<TP>::propagator(int directionId)
    {
        return propagators_[directionId];
    }

    template <class TP>
    inline
    TP const & JointTmpl<TP>::propagator(int directionId) const
    {
        return propagators_[directionId];
    }

    template <class TP>
    inline
    double JointTmpl<TP>::kuhn() const
    {
        return kuhn_;
    }

    template <class TP>
    JointTmpl<TP>::JointTmpl()
     : propagators_(),
       kuhn_(0.0)
    {
        propagator(0).setDirectionId(0);
        propagator(1).setDirectionId(1);
        propagator(0).setPartner(propagator(1));
        propagator(1).setPartner(propagator(0));
    }

    template <class TP>
    JointTmpl<TP>::~JointTmpl() = default;

    template <class TP>
    void JointTmpl<TP>::setKuhn(double kuhn)
    {
        kuhn_ = kuhn;
    }
}
#endif