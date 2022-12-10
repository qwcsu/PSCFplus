#ifndef DPD_JOINT_DESCRIPTOR_H
#define DPD_JOINT_DESCRIPTOR_H

#include <iostream>

#include <util/containers/Pair.h>

namespace Pscf
{
    using namespace Util;

    class JointDescriptor
    {
    public:

        JointDescriptor ();

        ~JointDescriptor ()=default;

        void setId (int id);

        void setVertexIds(int vertexAId, int vertexBId);

        void setMonomerId(int monomerId);

        int id () const;

        int monomerId() const;

        const Pair<int> & vertexIds() const;

        int vertexId(int i) const;

        template <class Archive>
        void serialize (Archive& ar, unsigned int versionId);


    private:

        int id_;

        int monomerId_;

        Pair<int> vertexIds_;

        friend 
        std::istream& operator >> (std::istream& in, JointDescriptor &joint);
        
        friend 
        std::ostream& operator << (std::ostream& out, const JointDescriptor &joint);
    };

    inline
    int JointDescriptor::id () const
    {
        return id_;
    }

    inline
    int JointDescriptor::monomerId() const
    {
        return monomerId_;
    }

    inline 
    const Pair<int> & JointDescriptor::vertexIds() const
    {
        return vertexIds_;
    }

    inline
    int JointDescriptor::vertexId(int i) const
    {
        return vertexIds_[i];
    }

    std::istream& operator >> (std::istream& in, JointDescriptor &joint);

    std::ostream& operator << (std::ostream& out, const JointDescriptor &joint);

    template <class Archive>
    void JointDescriptor::serialize(Archive& ar, unsigned int)
    {
        ar & id_;
        ar & monomerId_;
        ar & vertexIds_;
    }
}


#endif