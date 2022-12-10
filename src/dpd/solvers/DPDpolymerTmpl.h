#ifndef DPD_POLYMER_TMPL_H
#define DPD_POLYMER_TMPL_H

#include <pscf/chem/Species.h>           // base class
#include <util/param/ParamComposite.h>   // base class

#include <pscf/chem/Monomer.h>           // member template argument
#include <pscf/chem/Vertex.h>            // member template argument
#include <pscf/chem/JointDescriptor.h>
#include <util/containers/Pair.h>        // member template
#include <util/containers/DArray.h>      // member template
#include <util/containers/DMatrix.h>

#include <cmath>

namespace Pscf
{

    class DPDBlock;
    class Joint;

    using namespace Util;

    template<class DPDBlock, class Joint>
    class DPDpolymerTmpl : public Species, public ParamComposite
    {
    public:

        typedef typename DPDBlock::DPDpropagator DPDpropagator;

        DPDpolymerTmpl();

        ~DPDpolymerTmpl()=default;

        virtual void readParameters(std::istream& in);

        virtual void solve();

        DPDBlock& block(int id);

        DPDBlock const & block(int id) const;

        Joint & joint(int id);

        Joint const & joint(int id) const;

        const Vertex& vertex(int id) const;

        DPDpropagator& propagator(int blockId, int dirId);

        DPDpropagator const & propagator(int blockId, int dirId) const;

        DPDpropagator& propagator_j(int jointId, int dirId);

        DPDpropagator const & propagator_j(int jointId, int dirId) const;

        DPDpropagator& propagator(int id);

        const Pair<int> & propagatorId(int i) const;

        /// Accessors
        int nBlock() const;

        int nJoint() const;

        int nVertex() const;

        int nPropagator() const;

        int N() const;

    protected:

        virtual void makePlan();

        // virtual void readParameters(std::istream& in);
    private:
        /// Array of Block objects in this polymer.
        DArray<DPDBlock> blocks_;

        /// Array of Vertex objects in this polymer.
        DArray<Vertex> vertices_;

        /// Arraay of Joint objects in this polymer.
        DArray<Joint> joints_;

        /// Propagator ids, indexed in order of computation.
        DArray< Pair<int> > propagatorIds_;

        /// Number of blocks in this polymer
        int nBlock_;

        /// Number of vertices (ends or junctions) in this polymer
        int nVertex_;

        /// Number of vertices that are joints
        int nJoint_;

        /// Number of propagators (two per block).
        int nPropagator_;

    };

    template <class DPDBlock, class Joint>
    inline 
    int DPDpolymerTmpl<DPDBlock, Joint>::nBlock() const
    {  return nBlock_; }

    template <class DPDBlock, class Joint>
    inline
    int DPDpolymerTmpl<DPDBlock, Joint>::nJoint() const
    {   
        return nJoint_;
    }

    template <class DPDBlock, class Joint>
    inline 
    DPDBlock& DPDpolymerTmpl<DPDBlock, Joint>::block(int id)
    {  return blocks_[id]; }

    template <class DPDBlock, class Joint>
    inline 
    const DPDBlock& DPDpolymerTmpl<DPDBlock, Joint>::block(int id) const
    {  return blocks_[id]; }

    template <class DPDBlock, class Joint>
    inline
    Joint & DPDpolymerTmpl<DPDBlock, Joint>::joint(int id)
    {
        return joints_[id];
    }

    template <class DPDBlock, class Joint>
    inline
    const Joint & DPDpolymerTmpl<DPDBlock, Joint>::joint(int id) const
    {
        return joints_[id];
    }

    template <class DPDBlock, class Joint>
    inline
    int DPDpolymerTmpl<DPDBlock, Joint>::nVertex() const
    {
        return nVertex_;
    }

    template <class DPDBlock, class Joint>
    inline
    const Vertex& DPDpolymerTmpl<DPDBlock, Joint>::vertex(int id) const
    {
        return vertices_[id];
    }

    template <class DPDBlock, class Joint>
    inline
    int DPDpolymerTmpl<DPDBlock, Joint>::nPropagator() const
    {   
        return nPropagator_;
    }

    template <class DPDBlock, class Joint>
    inline
    int DPDpolymerTmpl<DPDBlock, Joint>::N() const
    {
        int value = 0;
        for(int blockId = 0; blockId < nBlock_; ++blockId)
        {
            value += blocks_[blockId].length();
        }

        for(int jointId = 0; jointId < nJoint_; ++jointId)
        {
            value += 1;
        }

        return value;
    }

    template <class DPDBlock, class Joint>
    inline
    typename DPDBlock::DPDpropagator&
    DPDpolymerTmpl<DPDBlock, Joint>::propagator(int blockId, int dirId)
    {
        return block(blockId).propagator(dirId);
    }

    template <class DPDBlock, class Joint>
    inline
    typename DPDBlock::DPDpropagator const &
    DPDpolymerTmpl<DPDBlock, Joint>::propagator(int blockId, int dirId) const
    {
        return block(blockId).propagator(dirId);
    } 

    template <class DPDBlock, class Joint>
    inline
    typename DPDBlock::DPDpropagator &
    DPDpolymerTmpl<DPDBlock, Joint>::propagator_j(int jointId, int dirId) 
    {
        return joint(jointId).propagator(dirId);
    } 

    template <class DPDBlock, class Joint>
    inline
    typename DPDBlock::DPDpropagator const &
    DPDpolymerTmpl<DPDBlock, Joint>::propagator_j(int jointId, int dirId) const
    {
        return joint(jointId).propagator(dirId);
    } 

    template <class DPDBlock, class Joint>
    inline
    typename DPDBlock::DPDpropagator &
    DPDpolymerTmpl<DPDBlock, Joint>::propagator(int id) 
    {
        Pair<int> propId = propagatorId(id);
        return propagator(propId[0], propId[1]);
    } 

    template <class DPDBlock, class Joint>
    inline
    Pair<int> const &
    DPDpolymerTmpl<DPDBlock, Joint>::propagatorId(int id) const
    {
        UTIL_CHECK(id >= 0)
        UTIL_CHECK(id <= nPropagator_)
        return propagatorIds_[id];
    } 

    template<class DPDBlock, class Joint>
    DPDpolymerTmpl<DPDBlock, Joint>::DPDpolymerTmpl()
     : blocks_(),
       vertices_(),
       propagatorIds_(),
       nBlock_(0),
       nVertex_(0),
       nPropagator_(0)
    {  setClassName("DPDpolymerTmpl"); }

    template <class DPDBlock, class Joint>
    void DPDpolymerTmpl<DPDBlock, Joint>::readParameters(std::istream& in)
    {
        read<int>(in, "nBlock", nBlock_);
        read<int>(in, "nVertex", nVertex_);

        blocks_.allocate(nBlock_);
        vertices_.allocate(nVertex_);
        propagatorIds_.allocate(2*nBlock_);

        readDArray<DPDBlock>(in, "blocks", blocks_, nBlock_);

        for (int blockId = 0; blockId < nBlock_; ++blockId)
        {
            blocks_[blockId].setJoint();
        }

        for (int vertexId = 0; vertexId < nVertex_; ++vertexId)
        {
            vertices_[vertexId].setId(vertexId);
        }

        int vertexId0, vertexId1;

        DPDBlock* blockPtr;

        for (int blockId = 0; blockId < nBlock_; ++blockId)
        {
            blockPtr = &(blocks_[blockId]);
            vertexId0 = blockPtr->vertexId(0);
            vertexId1 = blockPtr->vertexId(1);
            vertices_[vertexId0].addBlock(*blockPtr);
            vertices_[vertexId1].addBlock(*blockPtr);
        }

        makePlan();

        ensemble_ = Species::Closed;
        readOptional<Species::Ensemble>(in, "ensemble", ensemble_);
        if (ensemble_ == Species::Closed) {
            read(in, "phi", phi_);
        } else {
            UTIL_THROW("Cannot set the ensemble not to be closed");
        }

        Vertex const * vertexPtr = nullptr;
        DPDpropagator const * sourcePtr = nullptr;
        DPDpropagator * propagatorPtr = nullptr;
        Pair<int> propagatorId;

        int blockId, directionId, vertexId, i;
        for (blockId = 0; blockId < nBlock(); ++blockId)
        {
            for (directionId = 0; directionId < 2; ++directionId)
            {
                vertexId = block(blockId).vertexId(directionId);
                vertexPtr = &vertex(vertexId);
                propagatorPtr = &block(blockId).propagator(directionId);
                
                for (i = 0; i < vertexPtr->size(); ++i)
                {
                    propagatorId = vertexPtr->inPropagatorId(i);
                    
                    if (propagatorId[0] == blockId)
                    {
                        UTIL_CHECK(propagatorId[1] != directionId)
                    }
                    else {
                        sourcePtr =
                                &block(propagatorId[0]).propagator(propagatorId[1]);
                        propagatorPtr->addSource(*sourcePtr);
                    }
                }
            }
        }
    }

    template <class DPDBlock, class Joint>
    void DPDpolymerTmpl<DPDBlock, Joint>::makePlan()
    {
        if (nPropagator_ != 0) 
        {
            UTIL_THROW("nPropagator !=0 on entry");
        }

        DMatrix<bool> isFinished;
        isFinished.allocate(nBlock_, 2);
        for (int i = 0; i < nBlock_; ++i) 
        {
            for (int iDirection = 0; iDirection < 2; ++iDirection) 
            {
                isFinished(i, iDirection) = false;
            }
        }

        Pair<int> propagatorId;
        Vertex* inVertexPtr = nullptr;
        int inVertexId = -1;
        bool isReady;
        while (nPropagator_ < (nBlock_)*2)
        {
            for (int iBlock = 0; iBlock < nBlock_; ++iBlock)
            {
                for (int iDirection = 0; iDirection < 2; ++ iDirection)
                {
                    if (!isFinished(iBlock, iDirection))
                    {
                        inVertexId = blocks_[iBlock].vertexId(iDirection);
                        inVertexPtr = &vertices_[inVertexId];
                        isReady = true;
                        for (int j = 0; j < inVertexPtr->size(); ++j)
                        {
                            propagatorId = inVertexPtr->inPropagatorId(j);
                            if (propagatorId[0] != iBlock)
                            {
                                if (!isFinished(propagatorId[0],propagatorId[1]))
                                {
                                    isReady = false;
                                    break;
                                }
                            }
                        }
                        if (isReady)
                        {
                            propagatorIds_[nPropagator_][0] = iBlock;
                            propagatorIds_[nPropagator_][1] = iDirection;
                            isFinished(iBlock, iDirection) = true;
                            ++nPropagator_;
                        }
                    }
                } 
            }
        }  
    }

    template <class DPDBlock, class Joint>
    void DPDpolymerTmpl<DPDBlock, Joint>::solve()
    {
        // Clear all propagators
        for (int j = 0; j < nPropagator(); ++j)
        {
            propagator(j).setIsSolved(false);
        }

        for (int j = 0; j < nPropagator(); ++j)
        {   
            UTIL_CHECK(propagator(j).isReady())
            propagator(j).solve();
        }
        
        // for (int j = 0; j < nPropagator(); ++j)
        // {   
        //     // UTIL_CHECK(propagator(j).isReady())
        //     std::cout << "j = " << j << std::endl;
        //     std::cout << "state : " << propagator(j).isReady() <<"   "
        //               << "partner state:" << propagator(j).partner().isSolved() <<  std::endl;

        // }
        // for (int j = 0; j < nBlock(); ++j)
        // {   
        //     // UTIL_CHECK(propagator(j).isReady())
        //     std::cout << "block  " << j << " : ";
        //     std::cout << "p0 :  " << block(j).propagator(0).isSolved() <<"   "
        //               << "p0,partner :  " << block(j).propagator(0).partner().isSolved() <<"   "
        //               << "p1 :  " << block(j).propagator(1).isSolved() <<"   "
        //               << "p0,partner :  " << block(j).propagator(1).partner().isSolved() <<  std::endl;

        // }

        double q = block(0).propagator(0).computeQ();
        // std::cout << q << std::endl;

        double prefactor = 1.0/(N()*q);

        for (int i = 0; i < nBlock(); ++i) 
        {
            block(i).computeConcentration(prefactor);
        }
    }

}



#endif
