#ifndef PSPG_BOTTLEBRUSH_TMPL_H
#define PSPG_BOTTLEBRUSH_TMPL_H

#include <pscf/chem/Species.h>           // base class
#include <util/param/ParamComposite.h>   // base class

#include <pscf/chem/Monomer.h>           // member template argument
#include <pscf/chem/Vertex.h>            // member template argument
#include <util/containers/Pair.h>        // member template
#include <util/containers/DArray.h>      // member template
#include <util/containers/DMatrix.h>

#include <cmath>

namespace Pscf{
    
    class Block;
    using namespace Util;
    
    template <class Block>
    class BottlebrushTmpl : public Species, public ParamComposite
    {
    public:
        // // Modified diffusion equation solver for one block.
        // typedef typename Block::Propagator Propagator;
        // // Monomer concentration field.
        // typedef typename Propagator::CField CField;
        // // Chemical potential field.
        // typedef typename Propagator::WField WField;
        BottlebrushTmpl();
        ~BottlebrushTmpl() = default;
        virtual void readParameters(std::istream& in);
        virtual void solve();
    };


    // Non-inline functions

    /*
    * Constructor.
    */
    template <class Block>
    BottlebrushTmpl<Block>::BottlebrushTmpl()
    {  setClassName("BottlebrushTmpl"); }

}
#endif // !PSPG_BOTTLEBRUSH_TMPL_H