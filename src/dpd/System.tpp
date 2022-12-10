#ifndef DPD_SYSTEM_TPP
#define DPD_SYSTEM_TPP

#include "System.h"

#include <pspg/GpuResources.h>

#include <pscf/crystal/shiftToMinimum.h>

#include <util/format/Str.h>
#include <util/format/Int.h>
#include <util/format/Dbl.h>

#include <string>
#include <getopt.h>

int THREADS_PER_BLOCK;
int NUMBER_OF_BLOCKS;

namespace Pscf
{
namespace Pspg
{
    namespace DPD
    {
        using namespace Util;

        template <int D>
        System<D>::System()
         : DPDmixture_(),
           mesh_(),
           unitCell_(),
           fileMaster_(),
           interactionPtr_(nullptr),
           iteratorPtr_(0),
           basisPtr_(),
           wavelistPtr_(0),
           fieldIo_(),
           wFields_(),
           cFields_(),
           hasMixture_(false),
           hasUnitCell_(false),
           isAllocated_(false),
           hasWFields_(false),
           hasCFields_(false)
        {
            setClassName("System");

            interactionPtr_ = new ChiInteraction();
            wavelistPtr_ = new WaveList<D>();
            basisPtr_ = new Basis<D>();
            iteratorPtr_ = new AmIterator<D>(this);
        }

        template <int D>
        System<D>::~System()
        {
            delete interactionPtr_;   
            delete wavelistPtr_;
            delete basisPtr_;
            delete iteratorPtr_;
        }

        // This is excatly the same as in pspg
        template <int D>
        void System<D>::setOptions(int argc, char **argv)
        {
            bool eFlag = false;  // echo
            bool pFlag = false;  // param file
            bool cFlag = false;  // command file
            bool iFlag = false;  // input prefix
            bool oFlag = false;  // output prefix
            bool wFlag = false;  // GPU input 1 (# of blocks)
            bool tFlag = false;  // GPU input 2 (threads per block)
            char* pArg = nullptr;
            char* cArg = nullptr;
            char* iArg = nullptr;
            char* oArg = nullptr;

            // Read program arguments
            int c;
            opterr = 0;
            while ((c = getopt(argc, argv, "er:p:c:i:o:f1:2:")) != -1) {
                switch (c) {
                    case 'e':
                        eFlag = true;
                        break;
                    case 'p': // parameter file
                        pFlag = true;
                        pArg  = optarg;
                        break;
                    case 'c': // command file
                        cFlag = true;
                        cArg  = optarg;
                        break;
                    case 'i': // input prefix
                        iFlag = true;
                        iArg  = optarg;
                        break;
                    case 'o': // output prefix
                        oFlag = true;
                        oArg  = optarg;
                        break;
                    case '1': //number of blocks
                        NUMBER_OF_BLOCKS = atoi(optarg);
                        wFlag = true;
                        break;
                    case '2': //threads per block
                        THREADS_PER_BLOCK = atoi(optarg);
                        tFlag = true;
                        //something like this
                        break;
                    case '?':
                        Log::file() << "Unknown option -" << optopt << std::endl;
                        UTIL_THROW("Invalid command line option");
                    default:
                        UTIL_THROW("Default exit (setOptions)");
                }
            }

            // Set flag to echo parameters as they are read.
            if (eFlag) {
                Util::ParamComponent::setEcho(true);
            }

            // If option -p, set parameter file name
            if (pFlag) {
                fileMaster().setParamFileName(std::string(pArg));
            }

            // If option -c, set command file name
            if (cFlag) {
                fileMaster().setCommandFileName(std::string(cArg));
            }

            // If option -i, set path prefix for input files
            if (iFlag) {
                fileMaster().setInputPrefix(std::string(iArg));
            }

            // If option -o, set path prefix for output files
            if (oFlag) {
                fileMaster().setOutputPrefix(std::string(oArg));
            }

            if (!wFlag) {
                std::cout<<"Number of blocks not set " <<std::endl;
                exit(1);
            }

            if (!tFlag) {
                std::cout<<"Threads per block not set " <<std::endl;
                exit(1);
            }

        }

        template <int D>
        void System<D>::readParam()
        {
            readParam(fileMaster().paramFile());
        }

        template <int D>
        void System<D>::readCommands()
        {
            if (fileMaster().commandFileName().empty())
            {
                UTIL_THROW("Empty command file name");
            }
            readCommands(fileMaster().commandFile());
        }

        template <int D>
        void System<D>::readCommands(std::istream &in)
        {
            UTIL_CHECK(isAllocated_)
            std::string command;
            std::string filename;

            bool readNext = true;

            while (readNext)
            {
                in >> command;
                Log::file() << command << std::endl;
                if (command == "FINISH")
                {
                    Log::file() << std::endl;
                    readNext = false;
                }
                else if (command == "READ_W_BASIS")
                {
                    in >> filename;
                    Log::file() << " " << Str(filename, 20) <<std::endl;
                    fieldIo().readFieldsBasis(filename, wFields());
                    fieldIo().convertBasisToRGrid(wFields(), wFieldsRGrid());
                    
                    hasWFields_ = true;
                }
                else if (command == "READ_W_RGRID")
                {
                    in >> filename;
                    Log::file() << " " << Str(filename, 20) <<std::endl;
                    fieldIo().readFieldsRGrid(filename, wFieldsRGrid());
                    hasWFields_ = true;
                }
                else if (command == "ITERATE")
                {
                    Log::file() << std::endl;
                    
                    // Read w fields in grid format iff not already set.
                    if (!hasWFields_)
                    {
                        in >> filename;
                        Log::file() << "Reading w fields from file: "
                                    << Str(filename, 20) <<std::endl;
                        
                        fieldIo().readFieldsRGrid(filename, wFieldsRGrid());
                        hasWFields_ = true;
                    }
                    
                    // Attempt to iteratively solve SCFT equations
                    int fail = iterator().solve();
                    
                    hasCFields_ = true;

                    if (!fail)
                    {
                        computeFreeEnergy();
                        outputThermo(Log::file());
                    }
                    else
                    {
                        Log::file() << "Iterate has failed. Exiting "<<std::endl;
                    }
                }
                else if (command == "WRITE_W_BASIS")
                {
                    UTIL_CHECK(hasWFields_)
                    in >> filename;
                    Log::file() << "  " << Str(filename, 20) << std::endl;
                    fieldIo().convertRGridToBasis(wFieldsRGrid(), wFields());
                    fieldIo().writeFieldsBasis(filename, wFields());
                }
                else if (command == "WRITE_W_RGRID")
                {
                    UTIL_CHECK(hasWFields_)
                    in >> filename;
                    Log::file() << "  " << Str(filename, 20) << std::endl;
                    fieldIo().writeFieldsRGrid(filename, wFieldsRGrid());
                }
                else if (command == "WRITE_C_BASIS")
                {
                    UTIL_CHECK(hasCFields_)
                    in >> filename;
                    Log::file() << "  " << Str(filename, 20) << std::endl;
                    fieldIo().convertRGridToBasis(cFieldsRGrid(), cFields());
                    fieldIo().writeFieldsBasis(filename, cFields());
                }
                else if (command == "WRITE_C_RGRID")
                {
                    UTIL_CHECK(hasCFields_)
                    in >> filename;
                    Log::file() << "  " << Str(filename, 20) << std::endl;
                    fieldIo().writeFieldsRGrid(filename, cFieldsRGrid());
                }
                else if (command == "BASIS_TO_RGRID")
                {
                    hasCFields_ = false;

                    std::string inFileName;
                    in >> inFileName;
                    Log::file() << " " << Str(inFileName, 20) <<std::endl;

                    // fieldIo().readFieldsBasis(inFileName, cFields());
                    // fieldIo().convertBasisToRGrid(cFields(), cFieldsRGrid());

                    std::string outFileName;
                    in >> outFileName;
                    Log::file() << " " << Str(outFileName, 20) <<std::endl;
                    // fieldIo().writeFieldsRGrid(outFileName, cFieldsRGrid());
                }
                else if (command == "RGRID_TO_BASIS")
                {
                    hasCFields_ = false;

                    std::string inFileName;

                    in >> inFileName;
                    Log::file() << " " << Str(inFileName, 20) <<std::endl;
                    // fieldIo().readFieldsRGrid(inFileName, cFieldsRGrid());

                    // fieldIo().convertRGridToBasis(cFieldsRGrid(), cFields());

                    std::string outFileName;
                    in >> outFileName;
                    Log::file() << " " << Str(outFileName, 20) <<std::endl;
                    // fieldIo().writeFieldsBasis(outFileName, cFields());

                }
                else if (command == "KGRID_TO_RGRID") 
                {
                    hasCFields_ = false;

                    // Read from file in k-grid format
                    std::string inFileName;
                    in >> inFileName;
                    Log::file() << " " << Str(inFileName, 20) <<std::endl;
                    // fieldIo().readFieldsKGrid(inFileName, cFieldsKGrid());

                    // Use FFT to convert k-grid r-grid
                    /* for (int i = 0; i < mixture().nMonomer(); ++i)
                     {
                         fft().inverseTransform(cFieldKGrid(i), cFieldRGrid(i));
                     }*/

                    // Write to file in r-grid format
                    std::string outFileName;
                    in >> outFileName;
                    Log::file() << " " << Str(outFileName, 20) <<std::endl;
                    // fieldIo().writeFieldsRGrid(outFileName, cFieldsRGrid());

                }
                else if (command == "RHO_TO_OMEGA")
                {

                    // Read c field file in r-grid format
                    std::string inFileName;
                    in >> inFileName;
                    Log::file() << " " << Str(inFileName, 20) << std::endl;
                    // fieldIo().readFieldsRGrid(inFileName, cFieldsRGrid());

                    // Compute w fields, excluding Lagrange multiplier contribution
                    //code is bad here, `mangled' access of data in array
                    /*for (int i = 0; i < mixture().nMonomer(); ++i)
                    {
                        assignUniformReal <<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                        (wFieldRGrid(i).cDField(), 0, mesh().size());
                    }
                    for (int i = 0; i < mixture().nMonomer(); ++i)
                    {
                        for (int j = 0; j < mixture().nMonomer(); ++j)
                        {
                            pointWiseAddScale <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                            (wFieldRGrid(i).cDField(), cFieldRGrid(j).cDField(),
                             interaction().chi(i,j), mesh().size());
                        }
                    }*/

                    // Write w fields to file in r-grid format
                    std::string outFileName;
                    in >> outFileName;
                    Log::file() << " " << Str(outFileName, 20) << std::endl;
                    // fieldIo().writeFieldsRGrid(outFileName, wFieldsRGrid());

                }
                else
                {
                    Log::file() << "  Error: Unknown command  " << command << std::endl;
                    readNext = false;
                }
            }
        }

        template <int D>
        void System<D>::readParam(std::istream& in)
        {
            readBegin(in, className().c_str());
            readParameters(in);
            readEnd(in);
        }

        template<int D>
        void System<D>::readParameters(std::istream &in)
        {
            
            readParamComposite(in, dpdmixture());
            hasMixture_ = true;

            int nm = dpdmixture().nMonomer();
            int np = dpdmixture().nPolymer();
            
            interaction().setNMonomer(dpdmixture().nMonomer());
            readParamComposite(in, interaction());

            read(in, "unitCell", unitCell_);
            hasUnitCell_ = true;

            read(in, "mesh", mesh_);
            dpdmixture().setMesh(mesh());
            hasMesh_ = true;

            wavelist().allocate(mesh(), unitCell());
            wavelist().computeMinimumImages(mesh(), unitCell());
            dpdmixture().setupUnitCell(unitCell(), wavelist());

            read(in, "groupName", groupName_);
            basis().makeBasis(mesh(), unitCell(), groupName_);
            fieldIo_.associate(unitCell_, mesh_, fft_, 
                               groupName_, basis(), fileMaster_);

            allocate();

            readParamComposite(in, iterator());
            iterator().allocate();
        }

        template <int D>
        void System<D>::allocate()
        {
            UTIL_CHECK(hasMixture_)
            UTIL_CHECK(hasMesh_)

            int nMonomer = dpdmixture().nMonomer();

            wFields_.allocate(nMonomer);
            wFieldsRGrid_.allocate(nMonomer);
            wFieldsKGrid_.allocate(nMonomer);

            cFields_.allocate(nMonomer);
            cFieldsRGrid_.allocate(nMonomer);
            cFieldsKGrid_.allocate(nMonomer);

            for(int i = 0; i < nMonomer; ++i)
            {
                wField(i).allocate(basis().nStar());
                wFieldRGrid(i).allocate(mesh().dimensions());
                wFieldKGrid(i).allocate(mesh().dimensions());
                // std::cout << "KGrid mesh().dimensions() = " << mesh().dimensions() << "\n";
                cField(i).allocate(basis().nStar());
                cFieldRGrid(i).allocate(mesh().dimensions());
                cFieldKGrid(i).allocate(mesh().dimensions());
            }

            // ...

            isAllocated_ = true;
        }
    
        template <int D>
        void System<D>::computeFreeEnergy()
        {

        }

        template <int D>
        void System<D>::outputThermo(std::ostream& out)
        {}
    }

    
}}

#endif // ! DPD_SYSTEM_TPP