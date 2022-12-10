#ifndef DPD_ITERATOR_H
#define DPD_ITERATOR_H

#include <util/param/ParamComposite.h>    // base class
#include <util/global.h>  

namespace Pscf {
namespace Pspg {
    namespace DPD
    {
        template <int D>
        class System;

        using namespace Util;

        template <int D>
        class Iterator : public ParamComposite
        {

        public:

            Iterator();

            Iterator(System<D>* system);

            ~Iterator();

            virtual int solve() = 0;

            System<D>* systemPtr_;
        } ;
    }
}
}
#include "Iterator.tpp"
#endif