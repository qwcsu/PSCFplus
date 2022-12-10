#include "DPDBlockDescriptor.h"

namespace Pscf
{   
    using namespace Util;

    DPDBlockDescriptor::DPDBlockDescriptor()
        : id_(-1),
        monomerId_(-1),
        vertexIds_(),
        N_(0)
    {}

    void DPDBlockDescriptor::setId(int id)
    {  id_ = id; }

    void DPDBlockDescriptor::setVertexIds(int vertexId0, int vertexId1)
    {     
        vertexIds_[0] = vertexId0; 
        vertexIds_[1] = vertexId1; 
    }

    void DPDBlockDescriptor::setMonomerId(int monomerId)
    {  monomerId_ = monomerId; }

    void DPDBlockDescriptor::setLength(int N)
    {  N_ = N; }

    std::istream& operator>>(std::istream& in, DPDBlockDescriptor &block)
    {
        in >> block.id_;
        in >> block.monomerId_;
        in >> block.vertexIds_[0];
        in >> block.vertexIds_[1];
        in >> block.N_;
        return in;
    }

    std::ostream& operator<<(std::ostream& out, const DPDBlockDescriptor &block) 
    {
        out << block.id_;
        out << "  " << block.monomerId_;
        out << "  " << block.vertexIds_[0];
        out << "  " << block.vertexIds_[1];
        out << "  ";
        out << block.N_;
        return out;
    }
    

}