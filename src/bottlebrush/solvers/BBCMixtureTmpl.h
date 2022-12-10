#ifndef BBC_MIXTURE_TMPL_H
#define BBC_MIXTURE_TMPL_H

#include <pscf/chem/Monomer.h>
#include <util/param/ParamComposite.h>
#include <util/containers/DArray.h>

namespace Pscf
{
    using namespace Util;

    template <class TB, class TS>
    class BBCMixtureTmpl :public ParamComposite
    {

    public:

        /**
         *  Public typedefs
         */
        typedef TB Bottlebrush;

        typedef TS Solvent;

        /**
         * Constructor and destructor
         */
        BBCMixtureTmpl();

        ~BBCMixtureTmpl();

        /**
         * Read parameters from file and initialize.
         *
         * \param in input parameter file
         */
        virtual void readParameters(std::istream& in);


        Monomer & monomer(int id);

        Bottlebrush & bottlebrush(int id);

        Solvent & solvent(int id);

        int nMonomer() const; 

        int nBottlebrush() const;

        int nSolvent() const;

    private:

        DArray<Monomer> monomers_;

        DArray<Bottlebrush> bottlebrushes_;

        DArray<Solvent> solvents_;

        int nMonomer_;

        int nBottlebrush_;

        int nSolvent_;
    };  

    // Inline member functions

    template <class TB, class TS>
    inline
    int BBCMixtureTmpl<TB, TS>::nMonomer() const
    {
        return nMonomer_;
    } 

    template <class TB, class TS>
    inline
    int BBCMixtureTmpl<TB, TS>::nBottlebrush() const
    {
        return nBottlebrush_;
    } 

    template <class TB, class TS>
    inline
    int BBCMixtureTmpl<TB, TS>::nSolvent() const
    {
        return nSolvent_;
    }

    template <class TB, class TS>
    inline 
    Monomer & BBCMixtureTmpl<TB, TS>::monomer(int id)
    {  
        return monomers_[id]; 
    }

    template <class TB, class TS>
    inline 
    TB & BBCMixtureTmpl<TB, TS>::bottlebrush(int id)
    {  
        return bottlebrushes_[id];
    }

    template <class TB, class TS>
    inline 
    TS & BBCMixtureTmpl<TB, TS>::solvent(int id)
    {  
        return solvents_[id]; 
    }

    // Non-inline member functions

    /*
     * Constructor 
     */
    template <class TB, class TS>
    BBCMixtureTmpl<TB, TS>::BBCMixtureTmpl()
        : ParamComposite(),
          monomers_(),
          bottlebrushes_(),
          solvents_(),
          nMonomer_(0),
          nBottlebrush_(0),
          nSolvent_(0)
    {}

    /*
     * Destructor
     */
    template <class TB, class TS>
    BBCMixtureTmpl<TB, TS>::~BBCMixtureTmpl()
    {}

    /*
     * Read all parameters and initialize.
     *
     * This is a template only, the actual implementation
     * will be in class BBCMixture.(?)
     */
    template <class TB, class TS>
    void BBCMixtureTmpl<TB, TS>::readParameters(std::istream & in)
    {
        // Monomers
        read<int>(in, "nMonomer", nMonomer_);
        monomers_.allocate(nMonomer_);
        readDArray< Monomer >(in, "monomers", monomers_, nMonomer_);

        // Bottlebrushes
        read<int>(in, "nBottlebrush", nBottlebrush_);
        bottlebrushes_.allocate(nBottlebrush_);
        for (int i= 0; i < nBottlebrush_; ++i)
        {
            readParamComposite(in, bottlebrushes_[i]);
        }

        // Set statistical segment lengths for all blocks
        double kuhn;
        int monomerId;
        for (int i = 0; i < nBottlebrush_; ++i)
        {
            
        }
    }
}

#endif // !BBC_MIXTURE_TMPL_H