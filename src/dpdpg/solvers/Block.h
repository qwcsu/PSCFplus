#ifndef DPD_BLOCK_H
#define DPD_BLOCK_H

#include <iostream>

namespace Pscf{
namespace Pspg{
    namespace DPDpg
    {
        
        class Block
        {

        public:

            /**
             * Constructor
             */
            Block();

            /**
             * Destructor
             */
            ~Block();

            /**
             * Serialize to/from archive.
             *
             * \param ar input or output Archive
             * \param versionId archive format version index
             */ 
            template <class Archive>
            void serialize(Archive& ar, unsigned int versionId);

            /// \name Setters
            //@{

            /**
             * Set the id for this block.
             *
             * \param id integer index for this block
             */ 
            void setId(int id);

            /**
             * Set the monomer id.
             * 
             * \param monomerId integer id of monomer type (>=0)
             */
            void setMonomerId(int monomerId);

            /**
             * Set the length of this block.
             * 
             * The ``length" is Np, p is monomer type.
             * 
             * \param length block length (number of monomers).
             */ 
            virtual void setLength(int length);

            //@}
            /// \name Accessors (getters)
            //@{

            /**
             * Get the id of this block.
             */
            int id() const;

            /**
             * Get the monomer type id.
             */
            int monomerId() const;

            /**
             * Get the length (number of monomers) in this block.
             */
            int length() const;

            //@}

        private:

            /// Identifier for this block, unique within the polymer.
            int id_;

            /// Identifier for the associated monomer type.
            int monomerId_;

            /// Length of this block. 
            int length_;

        //friends

        friend 
        std::istream& operator >> (std::istream& in, Block& block);

        friend 
        std::ostream& operator << (std::ostream& out, const Block& block);
        };

        /**
         * istream extractor for a Monomer.
         *
         * \param in  input stream
         * \param block  Block to be read from stream
         * \return modified input stream
         */
        std::istream& operator >> (std::istream& in, Block& block);

        /**
         * ostream inserter for a Monomer.
         *
         * \param out  output stream
         * \param monomer  Monomer to be written to stream
         * \return modified output stream
         */
        std::ostream& operator << (std::ostream& out, const Block& block);


        // Inline member functions

        inline
        int Block::id() const
        {
            return id_;
        }

        inline
        int Block::monomerId() const
        {
            return monomerId_;
        }

        inline
        int Block::length() const
        {
            return length_;
        }

        /*
         * Serialize to/from an archive.
         */
        template <class Archive>
        void Block::serialize(Archive& ar, unsigned int)
        {
            ar & id_;
            ar & monomerId_;
            ar & length_;
        }

    }
}
}

#endif

