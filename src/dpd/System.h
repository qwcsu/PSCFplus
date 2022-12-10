#ifndef DPD_SYSTEM_H
#define DPD_SYSTEM_H

#include <pspg/field/FieldIo.h>

#include <dpd/iterator/AmIterator.h>

#include <pscf/inter/ChiInteraction.h>
#include <pscf/crystal/Basis.h>

#include <util/param/ParamComposite.h>     // base class
#include <util/misc/FileMaster.h>          // member
#include <util/containers/DArray.h>        // member template
#include <util/containers/Array.h>         // function parameter

#include <dpd/solvers/DPDmixture.h>

namespace Pscf
{
namespace Pspg
{
    namespace DPD
    {
        using namespace Util;

        template <int D>
        class System : public ParamComposite
        {
        public:

            typedef RDField<D> Field;

            typedef typename DPDpropagator<D>::WField WField;

            typedef typename DPDpropagator<D>::CField CField;

            System();

            ~System();

            void setOptions(int argc, char **argv);

            void readParam();

            void readCommands();

            virtual void readParam(std::istream& in);

            virtual void readParameters(std::istream& in);

            void readCommands(std::istream& in);

            /// Accessors (access objects by reference)
            FileMaster & fileMaster();

            DPDmixture<D> & dpdmixture();

            ChiInteraction & interaction();

            AmIterator<D>& iterator();

            Mesh<D> & mesh();

            UnitCell<D> & unitCell();

            WaveList<D> & wavelist();

            Basis<D> & basis();

            FieldIo<D>& fieldIo();

            FFT<D>& fft();

            void computeFreeEnergy();

            void outputThermo(std::ostream& out);

            DArray<Field>& wFields();

            Field& wField(int monomerId);

            DArray<WField>& wFieldsRGrid();

            WField& wFieldRGrid(int monomerId);

            DArray<RDFieldDft<D> >& wFieldsKGrid();

            RDFieldDft<D>& wFieldKGrid(int monomerId);

            DArray<Field>& cFields();

            Field& cField(int monomerId);

            DArray<CField>& cFieldsRGrid();

            CField& cFieldRGrid(int monomerId);

            DArray<RDFieldDft<D> >& cFieldsKGrid();

            RDFieldDft<D>& cFieldKGrid(int monomerId);

            bool hasWFields() const;
            bool hasCFields() const;

        private:

            FileMaster fileMaster_;

            DPDmixture<D> DPDmixture_;

            ChiInteraction * interactionPtr_;

            UnitCell<D> unitCell_;
            
            Mesh<D> mesh_;

            WaveList<D> * wavelistPtr_;

            std::string groupName_;

            Basis<D> * basisPtr_;

            FieldIo<D> fieldIo_;

            FFT<D> fft_;

            AmIterator<D> * iteratorPtr_;

            DArray<Field> wFields_;

            DArray<WField> wFieldsRGrid_;

            DArray<RDFieldDft<D> > wFieldsKGrid_;

            DArray<Field> cFields_;

            DArray<CField> cFieldsRGrid_;

            DArray<RDFieldDft<D> > cFieldsKGrid_;

            bool hasMixture_;

            bool hasUnitCell_;

            bool hasMesh_;

            bool isAllocated_;

            bool hasWFields_;

            bool hasCFields_;

            IntVec<D> kMeshDimensions_;

            void allocate();
        };

        template <int D>
        inline
        FileMaster & System<D>::fileMaster()
        {
            return fileMaster_;
        }

        template <int D>
        inline
        DPDmixture<D> & System<D>::dpdmixture()
        {
            return DPDmixture_;
        }

        template <int D>
        inline
        ChiInteraction & System<D>::interaction()
        {
            UTIL_ASSERT(interactionPtr_)
            return *interactionPtr_;
        }

        template <int D>
        inline
        Mesh<D>& System<D>::mesh()
        {
            return mesh_;
        }

        template <int D>
        inline
        UnitCell<D> & System<D>::unitCell()
        {
            return unitCell_;
        }

        template <int D>
        inline
        WaveList<D> & System<D>::wavelist()
        {
            return *wavelistPtr_;
        }

        template <int D>
        inline
        Basis<D> & System<D>::basis()
        {
            UTIL_ASSERT(basisPtr_)
            return *basisPtr_;
        }

        template <int D>
        inline
        AmIterator<D>& System<D>::iterator()
        {
            UTIL_ASSERT(iteratorPtr_)
            return *iteratorPtr_;
        }

        template <int D>
        inline
        FieldIo<D> & System<D>::fieldIo()
        {
            return fieldIo_;
        }

        template <int D>
        inline
        FFT<D>& System<D>::fft()
        {
            return fft_;
        }


        template <int D>
        inline
        DArray<RDField<D>> & System<D>::wFields()
        {
            return wFields_;
        }

        template <int D>
        inline
        RDField<D> & System<D>::wField(int id)
        {
            return wFields_[id];
        }

        template <int D>
        inline
        DArray<typename System<D>::WField> & System<D>::wFieldsRGrid()
        {
            return wFieldsRGrid_;
        }

        template <int D>
        inline
        typename System<D>::WField & System<D>::wFieldRGrid(int id)
        {
            return wFieldsRGrid_[id];
        }

        template <int D>
        inline
        DArray<RDFieldDft<D> >& System<D>::wFieldsKGrid()
        {
            return wFieldsKGrid_;
        }

        template <int D>
        inline
        RDFieldDft<D>& System<D>::wFieldKGrid(int id)
        {
            return wFieldsKGrid_[id];
        }

        template <int D>
        inline
        DArray<RDField<D>> & System<D>::cFields()
        {
            return cFields_;
        }

        template <int D>
        inline
        RDField<D> & System<D>::cField(int id)
        {
            return cFields_[id];
        }

        template <int D>
        inline
        DArray<typename System<D>::CField> & System<D>::cFieldsRGrid()
        {
            return cFieldsRGrid_;
        }

        template <int D>
        inline
        typename System<D>::CField & System<D>::cFieldRGrid(int id)
        {
            return cFieldsRGrid_[id];
        }

        template <int D>
        inline
        DArray<RDFieldDft<D> >& System<D>::cFieldsKGrid()
        {
            return cFieldsKGrid_;
        }

        template <int D>
        inline
        RDFieldDft<D>& System<D>::cFieldKGrid(int id)
        {
            return cFieldsKGrid_[id];
        }

        template <int D>
        inline
        bool System<D>::hasWFields() const
        {
            return hasWFields_;
        }

        template <int D>
        inline
        bool System<D>::hasCFields() const
        {
            return hasCFields_;
        }


        #ifndef DPD_SYSTEM_TPP
        // Suppress implicit instantiation
        extern template class System<1>;
        extern template class System<2>;
        extern template class System<3>;
        #endif
    }
    
}
}

#endif // ! DPD_SYSTEM_H