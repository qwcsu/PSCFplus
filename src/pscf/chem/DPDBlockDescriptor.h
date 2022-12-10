#ifndef DPD_BLOCK_DESCRIPTOR_H
#define DPD_BLOCK_DESCRIPTOR_H

#include <util/containers/Pair.h>

#include <iostream>

namespace Pscf
{
    using namespace Util;

    class DPDBlockDescriptor
    {
    public:

        DPDBlockDescriptor();

        int id() const;

        int monomerId() const;

        const Pair<int>& vertexIds() const;

        int vertexId(int i) const;

        int length() const;

        void setId(int id);

        void setVertexIds(int vertexAId, int vertexBId);

        void setMonomerId(int monomerId);

        virtual void setLength(int N);

        template <class Archive>
        void serialize(Archive& ar, unsigned int versionId);

    private:

        int id_;

        int monomerId_;

        Pair<int> vertexIds_;

        int N_;

        friend
        std::istream& operator >> (std::istream& in, DPDBlockDescriptor &block);

        friend 
        std::ostream& operator << (std::ostream& out, const DPDBlockDescriptor &block);
    };

    std::istream& operator >> (std::istream& in,  DPDBlockDescriptor &block);

    std::istream& operator << (std::istream& out, const DPDBlockDescriptor &block);

    inline 
    int DPDBlockDescriptor::id() const
    {  return id_; }

    inline 
    int DPDBlockDescriptor::monomerId() const
    {  return monomerId_; }

    inline 
    const 
    Pair<int>& DPDBlockDescriptor::vertexIds() const
    {  return vertexIds_; }

    inline 
    int DPDBlockDescriptor::vertexId(int i) const
    {  return vertexIds_[i]; }

    inline 
    int DPDBlockDescriptor::length() const
    {  return N_; }

    template <class Archive>
    void DPDBlockDescriptor::serialize(Archive& ar, unsigned int)
    {
        ar & id_;
        ar & monomerId_;
        ar & vertexIds_;
        ar & N_;
    }

        
    
}


#endif