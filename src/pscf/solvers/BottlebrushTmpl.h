#ifndef PSCF_BOTTLEBRUSH_TMPL_H
#define PSCF_BOTTLEBRUSH_TMPL_H

#include <pscf/chem/Species.h>           // base class
#include <util/param/ParamComposite.h>   // base class

#include <pscf/chem/Monomer.h>           // member template argument
#include <pscf/chem/Vertex.h>            // member template argument
#include <util/containers/Pair.h>        // member template
#include <util/containers/DArray.h>      // member template
#include <util/containers/DMatrix.h>

#include <cmath>

namespace Pscf
{
    class Block;

    using namespace Util;

    /**
     * Descriptor and MDE solver for a bottlebrush polymer.
     */
    template <class Block>
    class BottlebrushTmpl : public Species, public ParamComposite
    {

    public:

        // Modified diffusion equation solver for one block.
        typedef typename Block::Propagator Propagator;

        // Monomer concentration field.
        typedef typename Propagator::CField CField;

        // Chemical potential field.
        typedef typename Propagator::WField WField;

        /**
         * Constructor
         */
        BottlebrushTmpl();

        /**
         * Destructor
         */
        ~BottlebrushTmpl();

        /**
         * Read and initialize.
         *
         * \param in input parameter stream
         */
        virtual void readParameters(std::istream& in);

        /**
         * Solve modified diffusion equation.
         *
         * Upon return, q functions and block concentration fields
         * are computed for all propagators and blocks.
         */
        virtual void solve();
    };
}

#endif