#include "Block.h"

namespace Pscf{
namespace Pspg{
    namespace DPDpg
    {
        /*
         * Constructor.
         */
        Block::Block()
        : id_(-1),
          monomerId_(-1),
          length_(0)
        {}

        /*
         * Destructor
         */
        Block::~Block()
        {}

        /*
         * Set the id for this block.
         */
        void Block::setId(int id)
        {
            id_ = id;
        }

        /*
         * Set the monomer id.
         */ 
        void Block::setMonomerId(int monomerId)
        {
            monomerId_ = monomerId;
        }

        /*
         * Set the length of this block.
         */ 
        void Block::setLength(int length)
        {
            length_ = length;
        }

        std::istream& operator>>(std::istream& in, Block &block)
        {
            in >> block.id_;
            in >> block.monomerId_;
            in >> block.length_;
            return in;
        }

        std::ostream& operator<<(std::ostream& out, const Block &block) 
        {
            out << block.id_;
            out << "  " << block.monomerId_;
            out << "  " << block.length_;
            return out;
        }
    }
}
}