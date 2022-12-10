#ifndef DPDPG_SYSTEM_H
#define DPDPG_SYSTEM_H


#include <util/param/ParamComposite.h>     // base class
#include <util/misc/FileMaster.h>          // member
#include <util/containers/DArray.h>        // member template
#include <util/containers/Array.h>         // function parameter

#include <pspg/GpuResources.h>
#include <pspg/field/FieldIo.h>            // member
#include <pspg/solvers/WaveList.h>         // member
#include <pspg/field/RDField.h>            // typedef
#include <pspg/field/RDFieldDft.h>         // typedef

#include <pscf/crystal/Basis.h>            // member
#include <pscf/mesh/Mesh.h>                // member
#include <pscf/crystal/UnitCell.h>         // member
#include <pscf/inter/ChiInteraction.h>     // member

#include <dpdpg/iterator/AmIterator.h>
#include <dpdpg/solvers/DPDdiblock.h>

namespace Pscf
{
namespace Pspg
{
    namespace DPDpg
    {
        using namespace Util;

        template <int D>
        class System : public ParamComposite
        {
        public:

            /// Base class for WField and CField
            typedef RDField<D> Field;
            typedef RDFieldDft<D> FieldDft;

            /**
             * Constructor.
             */
            System();

            /**
            * Destructor.
            */
            ~System();

            /// \name Lifetime (Actions)
            //@{

            /**
             * Process command line options.
             */
            void setOptions(int argc, char **argv);

            /**
             * Read input parameters from default param file.
             */
            void readParam();

            /**
             * Read input parameters (with opening and closing lines).
             *
             * \param in input parameter stream
             */
            virtual void readParam(std::istream& in);

            /**
             * Read body of input parameters block (without opening and closing lines).
             *
             * \param in input parameter stream
             */
            virtual void readParameters(std::istream& in);

            void readCommands();

            void readCommands(std::istream& in);

            void shrinkWField();

            void computeFreeEnergy();
            
            void outputThermo(std::ostream& out);
            //@}

            DArray<Field>& wFields();

            Field& wField(int monomerId);

            DArray<Field>& wFieldsRGrid();

            Field& wFieldRGrid(int monomerId);

            DArray<RDFieldDft<D> >& wFieldsKGrid();

            RDFieldDft<D>& wFieldKGrid(int monomerId);


            DArray<Field>& cFields();

            Field& cField(int monomerId);

            DArray<Field>& cFieldsRGrid();

            DArray<Field>& cFieldsNRGrid();

            Field& cFieldRGrid(int monomerId);

            Field& cFieldNRGrid(int monomerId);

            DArray<RDFieldDft<D> >& cFieldsKGrid();

            RDFieldDft<D>& cFieldKGrid(int monomerId);



            /// \name Accessors (access objects by reference)
            //@{

            /**
             * Get DPD diblock copolymer by reference.
             */
            DPDdiblock<D>& dpddiblock();


            /**
             * Get the Iterator by reference.
             */
            //temporarily changed to allow testing on member functions
            AmIterator<D>& iterator();

            /**
             * Get FileMaster by reference.
             */
            FileMaster& fileMaster();

            UnitCell<D>& unitCell();

            Mesh<D>& mesh();

            WaveList<D>& wavelist();

            Basis<D>& basis();

            FieldIo<D>& fieldIo();

            FFT<D>& fft();

            std::string groupName();

            bool compressibility();

            //@}

        private:

            DArray<Field> wFields_;
            DArray<Field> wFieldsRGrid_;
            DArray<FieldDft> wFieldsKGrid_;

            DArray<Field> cFields_;
            DArray<Field> cFieldsRGrid_;
            DArray<FieldDft> cFieldsKGrid_;
            DArray<Field> cFieldsRGridNorm_;

            /**
             * DPD diblock polymer object (solves MDE for diblock polymer).
             */
            DPDdiblock<D> DPDdiblock_;

            /**
             * Spatial discretization mesh.
             */
            Mesh<D> mesh_;

            /**
             * Container for wavevector data.
             */
            WaveList<D>* wavelistPtr_;


            /**
             * Pointer to a Basis object
             */
            Basis<D>* basisPtr_;

            /**
             * group name.
             */
            std::string groupName_;

            /**
             * Crystallographic unit cell (type and dimensions).
             */
            UnitCell<D> unitCell_;

            /**
             * Filemaster (holds paths to associated I/O files).
             */
            FileMaster fileMaster_;

            /**
             * Pointer to Interaction (excess free energy model).
             */
            ChiInteraction* interactionPtr_;

            /**
             * Pointer to an iterator.
             */
            AmIterator<D>* iteratorPtr_;

            /**
             * FieldIo object for field input/output operations
             */
            FieldIo<D> fieldIo_;

            /**
             * FFT object to be used by iterator
             */
            FFT<D> fft_;

            double fHelmholtz_;
            double U_;
            double SJ_;
            double SA_;
            double SB_;
            double UAB_;
            double UCMP_;

            RDField<D> workArray;


            /**
             * Has the Mesh been initialized?
             */
            bool hasMesh_;

            /**
             * Has the UnitCell been initialized?
             */
            bool hasUnitCell_;

            bool isAllocated_;

            bool hasWFields_;

            bool hasCFields_;

            // 1 : compressible; 0 : incompressible
            bool comp_;    

            cudaReal* d_kernelWorkSpace_{};

            cudaReal* kernelWorkSpace_{};

            void allocate();

            /**
             * Compute inner product of two RDField fields (private, on GPU).
             */
            cudaReal innerProduct(const RDField<D>& a, const RDField<D>& b, int size);

            /**
             * Compute reduction of an RDField (private, on GPU).
             */
            cudaReal reductionH(RDField<D>& a, int size);

            cudaReal RombergIntegration(cudaReal *f, int size);

            double RombergInt(double *f, int size);
        };

        // Inline member functions

        template <int D>
        inline
        DArray<RDField<D>>& System<D>::wFields()
        {
            return wFields_;
        }

        template <int D>
        inline
        RDField<D>& System<D>::wField(int id)
        {
            return wFields_[id];
        }

        template <int D>
        inline
        DArray<RDField<D>>& System<D>::wFieldsRGrid()
        {
            return wFieldsRGrid_;
        }

        template <int D>
        inline
        RDField<D>& System<D>::wFieldRGrid(int id)
        {
            return wFieldsRGrid_[id];
        }

        template <int D>
        inline
        DArray<RDFieldDft<D>>& System<D>::wFieldsKGrid()
        {
            return wFieldsKGrid_;
        }

        // Get all monomer concentration fields, in a basis.
        template <int D>
        inline
        DArray<RDField<D> >& System<D>::cFields()
        {
            return cFields_;
        }

        // Get a single monomer concentration field, in a basis.
        template <int D>
        inline
        RDField<D>& System<D>::cField(int id)
        {
            return cFields_[id];
        }

        // Get all monomer concentration fields, on a grid.
        template <int D>
        inline
        DArray<RDField<D>>& System<D>::cFieldsRGrid()
        {
            return cFieldsRGrid_;
        }

        // Get all monomer normalized concentration fields, on a grid.
        template <int D>
        inline
        DArray<RDField<D>>& System<D>::cFieldsNRGrid()
        {
            return cFieldsRGridNorm_;
        }

        // Get a single monomer concentration field, on a grid.
        template <int D>
        inline
        RDField<D>& System<D>::cFieldRGrid(int id)
        {
            return cFieldsRGrid_[id];
        }

        // Get a single monomer mormalized concentration field, on a grid.
        template <int D>
        inline
        RDField<D>& System<D>::cFieldNRGrid(int id)
        {
            return cFieldsRGridNorm_[id];
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
        RDFieldDft<D>& System<D>::wFieldKGrid(int id)
        {
            return wFieldsKGrid_[id];
        }

        // Get the associated Mixture<D> object.
        template <int D>
        inline
        DPDdiblock<D>& System<D>::dpddiblock()
        {
            return DPDdiblock_;
        }

        // Get the FileMaster.
        template <int D>
        inline
        FileMaster& System<D>::fileMaster()
        {
            return fileMaster_;
        }

        // Get the Iterator.
        template <int D>
        inline
        AmIterator<D>& System<D>::iterator()
        {
            UTIL_ASSERT(iteratorPtr_)
            return *iteratorPtr_;
        }

        template <int D>
        inline
        UnitCell<D>& System<D>::unitCell()
        {
            return unitCell_;
        }

        template <int D>
        Mesh<D>& System<D>::mesh()
        {
            return mesh_;
        }

        template <int D>
        inline
        WaveList<D>& System<D>::wavelist()
        {
            return *wavelistPtr_;
        }

        template <int D>
        inline
        Basis<D>& System<D>::basis()
        {
            UTIL_ASSERT(basisPtr_)
            return *basisPtr_;
        }

        template <int D>
        inline
        FieldIo<D>& System<D>::fieldIo()
        {
            return fieldIo_;
        }

        template <int D>
        inline
        std::string System<D>::groupName()
        {
            return groupName_;
        }

        template <int D>
        inline
        FFT<D>& System<D>::fft()
        {
            return fft_;
        }

        template <int D>
        inline
        bool System<D>::compressibility()
        {
            return comp_;
        }


        #ifndef DPDPG_SYSTEM_TPP
        extern template class System<1>;
        extern template class System<2>;
        extern template class System<3>;
        #endif
    }
}
}

#endif // !DPDPG_SYSYTEM