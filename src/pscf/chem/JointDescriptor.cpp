#include "JointDescriptor.h"

namespace Pscf
{
    JointDescriptor::JointDescriptor()
     : id_(-1),
       monomerId_(-1),
       vertexIds_()
    {}

    void JointDescriptor::setId(int id)
    {
        id_ = id;
    }

    void JointDescriptor::setVertexIds(int vertexAId, int vertexBId)
    {
        vertexIds_[0] = vertexAId;
        vertexIds_[1] = vertexBId;
    }

    void JointDescriptor::setMonomerId(int monomerId)
    {
        monomerId_ = monomerId;
    }

    std::istream& operator>> (std::istream& in, JointDescriptor & joint)
    {
        in >> joint.id_;
        in >> joint.monomerId_;
        in >> joint.vertexIds_[0];
        in >> joint.vertexIds_[1];

        return in;
    }

    std::ostream& operator<< (std::ostream& out, const JointDescriptor & joint)
    {
        out << joint.id_;
        out << "  " << joint.monomerId_;
        out << "  " << joint.vertexIds_[0];
        out << "  " << joint.vertexIds_[1];
        return out;
    }
}