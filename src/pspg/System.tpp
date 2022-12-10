#ifndef PSPG_SYSTEM_TPP
#define PSPG_SYSTEM_TPP
// #define double float
/*
* PSCF - Polymer Self-Consistent Field Theory
*
* Copyright 2016 - 2019, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*/

#include "System.h"
#include <pspg/GpuResources.h>

#include <pscf/homogeneous/Clump.h>
#include <pscf/crystal/shiftToMinimum.h>

#include <util/format/Str.h>
#include <util/format/Int.h>
#include <util/format/Dbl.h>

#include <string>
#include <getopt.h>

// Global variable for kernels
int THREADS_PER_BLOCK;
int NUMBER_OF_BLOCKS;

namespace Pscf
{
    namespace Pspg
    {
        using namespace Util;

        /*
         * Constructor
         */
        template <int D>
        System<D>::System()
                : mixture_(),
                  mesh_(),
                  unitCell_(),
                  fileMaster_(),
                  homogeneous_(),
                  interactionPtr_(nullptr),
                  iteratorPtr_(0),
                  basisPtr_(),
                  wavelistPtr_(0),
                  fieldIo_(),
                  wFields_(),
                  cFields_(),
                  f_(),
                  c_(),
                  fHelmholtz_(0.0),
                  pressure_(0.0),
                  hasMixture_(false),
                  hasUnitCell_(false),
                  isAllocated_(false),
                  hasWFields_(false),
                  hasCFields_(false)
        {
            setClassName("System");
            interactionPtr_ = new ChiInteraction();
            iteratorPtr_ = new AmIterator<D>(this);
            wavelistPtr_ = new WaveList<D>();
            basisPtr_ = new Basis<D>();
        }

        template <int D>
        System<D>::~System()
        {
            delete interactionPtr_;
            delete iteratorPtr_;
            delete wavelistPtr_;
            delete basisPtr_;
            cudaFree(d_kernelWorkSpace_);
        }

        /*
         * Process command line options.
         */
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

        /*
        * Read default parameter file.
        */
        template <int D>
        void System<D>::readParam()
        {
            // paramFile() returns the file pointer of parameter file
            readParam(fileMaster().paramFile());
        }

        /*
        * Read parameter file (including open and closing brackets).
        */
        template <int D>
        void System<D>::readParam(std::istream& in)
        {
            readBegin(in, className().c_str());
            readParameters(in);
            readEnd(in);
        }

        /*
        * Read parameters and initialize.
        */
        template<int D>
        void System<D>::readParameters(std::istream &in)
        {
            readParamComposite(in, mixture());
            hasMixture_ = true;

            int nm = mixture().nMonomer();
            int np = mixture().nPolymer();
            int ns = 0;

            // Initialize homogeneous object
            homogeneous_.setNMolecule(np + ns);
            homogeneous_.setNMonomer(nm);
            initHomogeneous();
            // std::cout << c_[1] <<std::endl;
            // exit(1);

            // Read interaction (i.e., chi parameters)
            interaction().setNMonomer(mixture().nMonomer());
            readParamComposite(in, interaction());

            // Read unit cell type and its parameters
            read(in, "unitCell", unitCell_);
            hasUnitCell_ = true;

            /// Read crystallographic unit cell (used only to create basis)
            read(in, "mesh", mesh_);
            mixture().setMesh(mesh());
            hasMesh_ = true;

            // Construct wavelist
            wavelist().allocate(mesh(), unitCell());
            mixture().setupUnitCell(unitCell(), wavelist());

            // Read group name, construct basis
            read(in, "groupName", groupName_);
            basis().makeBasis(mesh(), unitCell(), groupName_);
            fieldIo_.associate(unitCell_, mesh_, fft_, groupName_,
                               basis(), fileMaster_);
            
            wavelist().computedKSq(unitCell());

            // Allocate memory for w and c fields
            allocate();

            // Initialize iterator
            readParamComposite(in, iterator());
            iterator().allocate();
        }

        /*
         * Allocate memory for fields
         */
        template <int D>
        void System<D>::allocate()
        {
            // Preconditions
            UTIL_CHECK(hasMixture_)
            UTIL_CHECK(hasMesh_)

            // Allocate wFields and cFields
            int nMonomer = mixture().nMonomer();

            wFieldsRGrid_ph_.allocate(nMonomer);
            wFieldsKGrid_.allocate(nMonomer);
            cFieldsKGrid_.allocate(nMonomer);
            cFieldsRGrid_.allocate(nMonomer);

            for(int i = 0; i < nMonomer; ++i)
            {
                wFieldRGridPh(i).allocate(mesh().dimensions());
                wFieldKGrid(i).allocate(mesh().dimensions());
                cFieldRGrid(i).allocate(mesh().dimensions());
                cFieldKGrid(i).allocate(mesh().dimensions());
            }

            workArray.allocate(mesh().size());

            cudaMalloc((void**)&d_kernelWorkSpace_, NUMBER_OF_BLOCKS * sizeof(cudaReal));
            kernelWorkSpace_ = new cudaReal[NUMBER_OF_BLOCKS];

            isAllocated_ = true;
        }

        /*
         * Read and execute commands from the default command file.
         */
        template <int D>
        void System<D>::readCommands()
        {
            if (fileMaster().commandFileName().empty())
            {
                UTIL_THROW("Empty command file name");
            }
            readCommands(fileMaster().commandFile());
        }

        /*
         * Read and execute commands from a specified command file.
         */
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
                    fieldIo().readFieldsRGrid(filename, wFieldsRGridPh());
                    hasWFields_ = true;
                }
                else if (command == "ITERATE")
                {
                    Log::file() << std::endl;

                    // Read w fields in grid format iff not already set.
                    if (!hasWFields_)
                    {
                        std::cout << "Read w field before iteration." << std::endl;
                        exit(1);
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
                else if (command == "CHI_PATH")
                {
                    Log::file() << std::endl;

                    if (!hasWFields_)
                    {
                        in >> filename;
                        Log::file() << "Reading w fields from file: "
                                    << Str(filename, 20) <<std::endl;
                        fieldIo().readFieldsRGrid(filename, wFieldsRGrid());
                        hasWFields_ = true;
                    }

                    int i, j;
                    // start point, current point, and end point.
                    // end ponit is read from command file, while 
                    // start point is read from param file.
                    double start_chi, current_chi, end_chi;
                    // current increment, maximum increment, minimum increment,
                    // and adpative factor, which are read from command file
                    double inc, max_inc, min_inc, fac;
                    // Is finished? Set false initially
                    bool isFinished = false;
                    // Is chi increasing? Set true initially
                    bool dir = true;

                    std::ofstream outRunfile;

                    in.ignore(5,'chi(');
                    in >> i ;
                    in >> j;
                    in.ignore(1,')');
                    in >> end_chi
                       >> inc 
                       >> max_inc
                       >> min_inc
                       >> fac
                       >> filename;
                    // in.ignore(5, '(');
                    outRunfile.open(filename, std::ios::app);
                    outRunfile <<std::endl;
                    outRunfile << "        chiN                           fHelmholtz              error          cell params" << std::endl;
                    outRunfile << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

                    start_chi = interaction().chi(i,j);
                    // Log::file() << i << "   " << j << std::endl;
                    // // Log::file() << start_chi << std::endl;
                    // Log::file() << end_chi << std::endl;
                    // Log::file() << inc << std::endl;
                    // Log::file() << fac << std::endl;
                    // Log::file() << filename << std::endl;
                    // exit(1);

                    if(start_chi == end_chi)
                    {
                        Log::file() << "The start point equals to the stop point." << std::endl;
                        exit(1);
                    }

                    if(start_chi > end_chi)
                    {
                        inc *= -1;
                        dir = false;
                    }

                    // Read w fields in grid format iff not already set.
                    if (!hasWFields_)
                    {
                        in >> filename;
                        Log::file() << "Reading w fields from file: "
                                    << Str(filename, 20) <<std::endl;
                        fieldIo().readFieldsRGrid(filename, wFieldsRGrid());
                        hasWFields_ = true;
                    }

                    current_chi = start_chi;

                    while(!isFinished)
                    {
                        current_chi += inc;     // step forward
                        if((dir&&current_chi >= end_chi) || ((!dir)&&current_chi <= end_chi))
                        {
                            current_chi = end_chi;
                            isFinished = true;
                        }
                        interaction().chi_(i,j) = current_chi;
                        interaction().chi_(j,i) = current_chi;
                        unitCell().setParameters(unitCell().parameters());  // Reset cell parameters
                        // iteration

                        // Attempt to iteratively solve SCFT equations
                        Log::file()<<std::endl;
                        Log::file()<< "================================================" << std::endl;
                        Log::file() << "*Current chi = " << current_chi << "*" << std::endl<<std::endl;
                        int fail = iterator().solve();
                        hasCFields_ = true;

                        if (!fail)
                        {
                            computeFreeEnergy();
                            outputThermo(Log::file());
                            outRunfile << Dbl(current_chi, 21, 10) 
                                       << Dbl(fHelmholtz_, 35, 14) 
                                       << Dbl(iterator().final_error, 17, 6);

                            for(int i = 0; i < unitCell().nParameter(); ++i)
                                outRunfile << Dbl(unitCell().parameter(i), 19, 8);
                            outRunfile << std::endl;

                            if (abs(inc*fac) <= max_inc)
                                inc *= fac;
                            else
                            {
                                if(start_chi > end_chi)
                                    inc = max_inc;
                                else
                                    inc = -max_inc;
                            }
                        }
                        else
                        {
                            Log::file() << "Iterate has failed."<<std::endl;
                            if(abs(inc) < min_inc)
                            {
                                Log::file() << "Smallest increment reached."<<std::endl;
                                outRunfile.close();
                            }
                            else
                            {
                                current_chi -= inc;
                                inc /= fac;
                            }
                        }
                    }
                }
                else if (command == "a_PATH")
                {
#if 0
                    Log::file() << std::endl;

                    if (!hasWFields_)
                    {
                        in >> filename;
                        Log::file() << "Reading w fields from file: "
                                    << Str(filename, 20) <<std::endl;
                        fieldIo().readFieldsRGrid(filename, wFieldsRGrid());
                        hasWFields_ = true;
                    }
                    // start point, current point, and end point.
                    // end ponit is read from command file, while 
                    // start point is read from param file.
                    double start_a, current_a, end_a;
                    // current increment, maximum increment, minimum increment,
                    // and adpative factor, which are read from command file
                    double inc, max_inc, min_inc, fac;
                    // Is finished? Set false initially
                    bool isFinished = false;
                    // Is chi increasing? Set true initially
                    bool dir = true;
                    int i;

                    std::ofstream outRunfile;

                    in.ignore(3,'a(');
                    in >> i ;
                    in.ignore(3,')');                    
                    in >> end_a
                       >> inc 
                       >> max_inc
                       >> min_inc
                       >> fac
                       >> filename;
                    // in.ignore(5, '(');
                    outRunfile.open(filename, std::ios::app);
                    outRunfile <<std::endl;
                    outRunfile << "        a                             fHelmholtz              error          cell params" << std::endl;
                    outRunfile << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

                    start_a = mixture().monomer(i).step();
                    Log::file() << i << std::endl;
                    Log::file() << end_a << std::endl;
                    exit(1);
                    // Log::file() << end_chi << std::endl;
                    // Log::file() << inc << std::endl;
                    // Log::file() << fac << std::endl;
                    // Log::file() << filename << std::endl;
                    // exit(1);
#endif
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
                    fieldIo().writeFieldsRGrid(filename, wFieldsRGridPh());
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
                else if (command == "KGRID_TO_RGRID") {
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
                else if (command == "WRITE_THERMO")
                {
                    std::string FileName, ParamFileName;
                    in >> FileName
                       >> ParamFileName;
                    Log::file() << " " << Str(FileName, 20) << Str(ParamFileName, 20)  <<std::endl;
                    std::ofstream outRunfile;
                    outRunfile.open(FileName, std::ios::app);
                    outRunfile <<std::endl;
                    outRunfile << " Parameter file : " << ParamFileName << std::endl;
                    outRunfile <<std::endl;
                    outRunfile << "         fHelmholtz                 pressure                    TS                       U                  error   " << std::endl;
                    outRunfile << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
                
                    outRunfile << Dbl(fHelmholtz_, 25, 14) 
                               << Dbl(pressure_, 25, 14) 
                               << Dbl(S_, 25, 14) 
                               << Dbl(E_, 25, 14) 
                               << Dbl(iterator().final_error, 15, 4);
                    outRunfile.close();

                }
                else if (command == "WRITE_PROPAGATOR")
                {
                    in >> filename;
                    Log::file() << "  " << Str(filename, 20) << std::endl;

                    std::ofstream outRunfile;
                    outRunfile.open(filename, std::ios::out);
                    
                    outRunfile << " Single Chain Partition Function Q = "
                               << Dbl(mixture().polymer(0).block(0).propagator(0).computeQ(), 18, 12) << std::endl;
                    int nPolymer = mixture().nPolymer();
                    for (int i = 0; i < nPolymer; i++)
                    {   
                        outRunfile<< "Polymer " << i << " :" <<std::endl;
                        int nBlock = mixture().polymer(i).nBlock(); 
                        for (int j = 0; j < nBlock; j++)
                        {
                            outRunfile<< "  Block " << j << " :" <<std::endl;
                            int ns = mixture().polymer(i).block(j).ns();

                            Propagator<D> const & p0 = mixture().polymer(i).block(j).propagator(0);
                            Propagator<D> const & p1 = mixture().polymer(i).block(j).propagator(1);
                            
                            cudaReal *pg0, *pg1;
                            
                            pg0 = new cudaReal [ns*mesh().size()];
                            pg1 = new cudaReal [ns*mesh().size()];

                            cudaMemcpy(pg0, p0.head(), ns*mesh().size()*sizeof(cudaReal), cudaMemcpyDeviceToHost);
                            cudaMemcpy(pg1, p1.head(), ns*mesh().size()*sizeof(cudaReal), cudaMemcpyDeviceToHost);
                            
                            outRunfile<< "    Direction 0 :" <<std::endl;    
                            for (int x = 0; x < mesh().size(); x++)
                            {
                                for (int s = 0; s < ns; s++)
                                {
                                    outRunfile<< Dbl(pg0[mesh().size()*s+x], 18, 12) << "    ";
                                }
                               outRunfile<<std::endl;
                            }
                            
                            outRunfile<< "    Direction 1 :" <<std::endl;  
                            for (int x = 0; x < mesh().size(); x++)
                            {
                                for (int s = 0; s < ns; s++)
                                {
                                    outRunfile<< Dbl(pg1[mesh().size()*s+x], 18, 12) << "    ";
                                }
                               outRunfile<<std::endl;
                            } 
                            delete [] pg0;
                            delete [] pg1;
                        }
                    }
                    outRunfile.close();
                }
                else
                {
                    Log::file() << "  Error: Unknown command  " << command << std::endl;
                    readNext = false;
                }
            }
        }

        /*
         * Initialize Pscf::Homogeneous::Mixture homogeneous_ member.
         */
        template <int D>
        void System<D>::initHomogeneous()
        {
            // Set number of molecular species and monomers
            int nm = mixture().nMonomer();
            int np = mixture().nPolymer();
            //int ns = mixture().nSolvent();
            int ns = 0;
            UTIL_CHECK(homogeneous_.nMolecule() == np + ns)
            UTIL_CHECK(homogeneous_.nMonomer() == nm)

            // Allocate c_ work array, if necessary
            if(c_.isAllocated())
            {
                UTIL_CHECK(c_.capacity() == nm)
            }else
            {
                c_.allocate(nm);
            }

            int i;   // molecule index
            int j;   // monomer index
            int k;   // block or clump index
            int nb;  // number of blocks
            int nc;  // number of clumps

            // Loop over polymer molecule species
            for (i = 0; i < np; ++i)
            {
                // Initial array of clump sizes
                for (j = 0; j < nm; ++j)
                {
                    c_[j] = 0.0;
                }

                // Compute clump sizes for all monomer types.
                nb = mixture().polymer(i).nBlock();
                for (k = 0; k < nb; ++k)
                {
                    Block<D>& block = mixture().polymer(i).block(k);
                    j = block.monomerId();
                    c_[j] += block.length();
                }

                // Count the number of clumps of nonzero size
                nc = 0;
                for (j = 0; j < nm; ++j)
                {
                    if (c_[j] > 1.0E-10)
                    {
                        ++nc;
                    }
                }
                homogeneous_.molecule(i).setNClump(nc);

                // Set clump properties for this Homogeneous::Molecule
                k = 0; // Clump index
                for (j = 0; j < nm; ++j)
                {
                    if (c_[j] > 1.0E-10)
                    {
                        homogeneous_.molecule(i).clump(k).setMonomerId(j);
                        homogeneous_.molecule(i).clump(k).setSize(c_[j]);
                        ++k;
                    }
                }
                homogeneous_.molecule(i).computeSize();
            }
        }


        /*
        * Compute Helmholtz free energy and pressure
        */
        template <int D>
        void System<D>::computeFreeEnergy()
        {
            fHelmholtz_ = 0.0;
            E_ = 0.0;
            S_ = 0.0;
            // Compute ideal gas contributions to f_Helmholtz
            Polymer<D>* polymerPtr;
            double phi, mu, length;
            // debug
            double temp = 0.0;
            int np = mixture().nPolymer();
            for (int i = 0; i < np; ++i) {
                polymerPtr = &mixture().polymer(i);
                phi = polymerPtr->phi();
                mu = polymerPtr->mu();
//                 std::cout << "mu  " << mu << "\n";
                // Recall: mu = ln(phi/q)
                length = polymerPtr->length();
                fHelmholtz_ += phi*( mu - 1.0 )/length;
            }

            int nm  = mixture().nMonomer();
            int nx;

            if (D == 3)
            {
                nx = mesh().size();
            }

            cudaMemset(workArray.cDField(), 0, nx);
            for (int i = 0; i < nm; ++i) {
                for (int j = i + 1; j < nm; ++j) {
                    assignUniformReal
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (workArray.cDField(), interaction().chi(i, j), nx);
                    inPlacePointwiseMul
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (workArray.cDField(), cFieldsRGrid_[i].cDField(), nx);
                    inPlacePointwiseMul
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (workArray.cDField(), cFieldsRGrid_[j].cDField(), nx);

                    if(D == 3)
                    {
                        int n[3];   // n[3] stores the number of mesh points in each direction
                        for (int i = 0; i < 3; ++i)
                        {
                            n[i] = mesh().dimensions()[i];
                        }

                        // Here, the triple integral will be transfromed into single integral 
                        double *I_single,
                               *I_double;
                        I_single = new double[n[1]*n[2]];
                        I_double = new double[n[2]];

                        for (int iz = 0; iz < n[2]; ++iz)
                        {
                            for (int iy = 0; iy < n[1]; ++iy)
                            {
// #if REPS == 0
//                                 I_single[iy + iz*n[1]] = RI0_gpu(workArray.cDField() + n[0]*(iy + iz*n[1]), n[0]);
// #endif
// #if REPS == 1
//                                 I_single[iy + iz*n[1]] = RI1_gpu(workArray.cDField() + n[0]*(iy + iz*n[1]), n[0]);
// #endif
// #if REPS == 2
//                                 I_single[iy + iz*n[1]] = RI2_gpu(workArray.cDField() + n[0]*(iy + iz*n[1]), n[0]);
// #endif
// #if REPS == 3
//                                 I_single[iy + iz*n[1]] = RI3_gpu(workArray.cDField() + n[0]*(iy + iz*n[1]), n[0]);
// #endif
// #if REPS == 4
                                I_single[iy + iz*n[1]] = RI4_gpu(workArray.cDField() + n[0]*(iy + iz*n[1]), n[0]);
// #endif
                            }
// #if REPS == 0
//                             I_double[iz] = RI0_cpu(I_single + n[1]*iz, n[1]);
// #endif
// #if REPS == 1
//                             I_double[iz] = RI1_cpu(I_single + n[1]*iz, n[1]);
// #endif
// #if REPS == 2
//                             I_double[iz] = RI2_cpu(I_single + n[1]*iz, n[1]);
// #endif
// #if REPS == 3
//                             I_double[iz] = RI3_cpu(I_single + n[1]*iz, n[1]);
// #endif
// #if REPS == 4
                            I_double[iz] = RI4_cpu(I_single + n[1]*iz, n[1]);
// #endif
                        }
// #if REPS == 0    
//                         fHelmholtz_ += RI0_cpu(I_double, n[2]);
//                         E_ += RI0_cpu(I_double, n[2]);;     
// #endif  
// #if REPS == 1    
//                         fHelmholtz_ += RI1_cpu(I_double, n[2]);
//                         E_ += RI1_cpu(I_double, n[2]);;     
// #endif     
// #if REPS == 2    
//                         fHelmholtz_ += RI2_cpu(I_double, n[2]);
//                         E_ += RI2_cpu(I_double, n[2]);;     
// #endif                   
// #if REPS == 3    
//                         fHelmholtz_ += RI3_cpu(I_double, n[2]);
//                         E_ += RI3_cpu(I_double, n[2]);;     
// #endif
// #if REPS == 4
                        fHelmholtz_ += RI4_cpu(I_double, n[2]);
                        E_ += RI4_cpu(I_double, n[2]);;
// #endif
                        // std::cout << "Q_ = " << Q << std::endl;
                        // std::cout << "Q  = " << RombergInt(I_double, n[2]) << std::endl;
                        delete [] I_single;
                        delete [] I_double;
                    }
                    if(D == 2)
                    {
                        int n[2];   // n[3] stores the number of mesh points in each direction
                        for (int i = 0; i < 2; ++i)
                        {
                            n[i] = mesh().dimensions()[i];
                        }

                        // Here, the triple integral will be transfromed into single integral 
                        double *I_single;
                        I_single = new double[n[1]];

                        for (int ix = 0; ix < n[0]; ++ix)
                        {
// #if REPS == 0
//                             I_single[ix] = RI0_gpu(workArray.cDField() + n[1]*ix, n[1]);
// #endif
// #if REPS == 1
//                             I_single[ix] = RI1_gpu(workArray.cDField() + n[1]*ix, n[1]);
// #endif
// #if REPS == 2
//                             I_single[ix] = RI2_gpu(workArray.cDField() + n[1]*ix, n[1]);
// #endif
// #if REPS == 3
//                             I_single[ix] = RI3_gpu(workArray.cDField() + n[1]*ix, n[1]);
// #endif
// #if REPS == 4
                            I_single[ix] = RI4_gpu(workArray.cDField() + n[1]*ix, n[1]);
// #endif
                        }
// #if REPS == 0
//                         fHelmholtz_ +=  RI0_cpu(I_single, n[0]);
//                         E_ += RI0_cpu(I_single, n[0]);                 
// #endif
// #if REPS == 1
//                         fHelmholtz_ +=  RI1_cpu(I_single, n[0]);
//                         E_ += RI1_cpu(I_single, n[0]);                 
// #endif
// #if REPS == 2
//                         fHelmholtz_ +=  RI2_cpu(I_single, n[0]);
//                         E_ += RI2_cpu(I_single, n[0]);                 
// #endif
// #if REPS == 3
//                         fHelmholtz_ +=  RI3_cpu(I_single, n[0]);
//                         E_ += RI3_cpu(I_single, n[0]);                 
// #endif
// #if REPS == 4
                        fHelmholtz_ +=  RI4_cpu(I_single, n[0]);
                        E_ += RI4_cpu(I_single, n[0]);
// #endif
                        delete [] I_single;
                    }
                    if(D == 1)
                    {
// #if REPS == 0
//                         fHelmholtz_ += RI0_gpu(workArray.cDField(), nx);
//                         E_ += RI0_gpu(workArray.cDField(), nx);
// #endif
// #if REPS == 1
//                         fHelmholtz_ += RI1_gpu(workArray.cDField(), nx);
//                         E_ += RI1_gpu(workArray.cDField(), nx);
// #endif
// #if REPS == 2
//                         fHelmholtz_ += RI2_gpu(workArray.cDField(), nx);
//                         E_ += RI2_gpu(workArray.cDField(), nx);
// #endif
// #if REPS == 3
//                         fHelmholtz_ += RI3_gpu(workArray.cDField(), nx);
//                         E_ += RI3_gpu(workArray.cDField(), nx);
// #endif
// #if REPS == 4
                        fHelmholtz_ += RI4_gpu(workArray.cDField(), nx);
                        E_ += RI4_gpu(workArray.cDField(), nx);
// #endif
                    }
                }
                assignReal <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                        (workArray.cDField(), wFieldsRGrid_ph_[i].cDField(), nx);
                inPlacePointwiseMul <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                        (workArray.cDField(), cFieldsRGrid_[i].cDField(), nx);
                if(D == 3)
                {
                    int n[3];   // n[3] stores the number of mesh points in each direction
                    n[0] = 1;  n[1] = 1;  n[2] = 1;
                    for (int i = 0; i < 3; ++i)
                    {
                        n[i] = mesh().dimensions()[i];
                    }
                    // Here, the triple integral will be transfromed into single integral 
                    double *I_single,
                           *I_double;
                    I_single = new double[n[1]*n[2]];
                    I_double = new double[n[2]];
                    for (int iz = 0; iz < n[2]; ++iz)
                    {
                        for (int iy = 0; iy < n[1]; ++iy)
                        {
// #if REPS == 0 
//                             I_single[iy + iz*n[1]] = RI0_gpu(workArray.cDField() + n[0]*(iy + iz*n[1]), n[0]);
// #endif
// #if REPS == 1 
//                             I_single[iy + iz*n[1]] = RI1_gpu(workArray.cDField() + n[0]*(iy + iz*n[1]), n[0]);
// #endif
// #if REPS == 2 
//                             I_single[iy + iz*n[1]] = RI2_gpu(workArray.cDField() + n[0]*(iy + iz*n[1]), n[0]);
// #endif
// #if REPS == 3 
//                             I_single[iy + iz*n[1]] = RI3_gpu(workArray.cDField() + n[0]*(iy + iz*n[1]), n[0]);
// #endif
// #if REPS == 4 
                            I_single[iy + iz*n[1]] = RI4_gpu(workArray.cDField() + n[0]*(iy + iz*n[1]), n[0]);
// #endif
                        }
                    }
                    for (int iz = 0; iz < n[2]; ++iz)
                    {
// #if REPS == 0       
//                         I_double[iz] = RI0_cpu(I_single + n[1]*iz, n[1]);
// #endif
// #if REPS == 1       
//                         I_double[iz] = RI1_cpu(I_single + n[1]*iz, n[1]);
// #endif
// #if REPS == 2       
//                         I_double[iz] = RI2_cpu(I_single + n[1]*iz, n[1]);
// #endif
// #if REPS == 3       
//                         I_double[iz] = RI3_cpu(I_single + n[1]*iz, n[1]);
// #endif
// #if REPS == 4 
                        I_double[iz] = RI4_cpu(I_single + n[1]*iz, n[1]);
// #endif
                    }
// #if REPS == 0
//                     temp += RI0_cpu(I_double, n[2]);
// #endif
// #if REPS == 1
//                     temp += RI1_cpu(I_double, n[2]);
// #endif
// #if REPS == 2
//                     temp += RI2_cpu(I_double, n[2]);
// #endif
// #if REPS == 3
//                     temp += RI3_cpu(I_double, n[2]);
// #endif
// #if REPS == 4                   
                    temp += RI4_cpu(I_double, n[2]);
// #endif
                    delete [] I_single;
                    delete [] I_double;
                }
                else if(D == 2)  
                {
                    int n[2];   // n[3] stores the number of mesh points in each direction
                    for (int i = 0; i < 2; ++i)
                    {
                        n[i] = mesh().dimensions()[i];
                    }
                    // Here, the triple integral will be transfromed into single integral 
                    double *I_single;
                    I_single = new double[n[1]];

                    for (int ix = 0; ix < n[0]; ++ix)
                    {
// #if REPS == 0
//                         I_single[ix] = RI0_gpu(workArray.cDField() + n[1]*ix, n[1]);
// #endif
// #if REPS == 1
//                         I_single[ix] = RI1_gpu(workArray.cDField() + n[1]*ix, n[1]);
// #endif
// #if REPS == 2
//                         I_single[ix] = RI2_gpu(workArray.cDField() + n[1]*ix, n[1]);
// #endif
// #if REPS == 3
//                         I_single[ix] = RI3_gpu(workArray.cDField() + n[1]*ix, n[1]);
// #endif
// #if REPS == 4
                        I_single[ix] = RI4_gpu(workArray.cDField() + n[1]*ix, n[1]);
// #endif
                    }
// #if REPS == 0
//                     temp += RI0_cpu(I_single, n[0]);
// #endif
// #if REPS == 1
//                     temp += RI1_cpu(I_single, n[0]);
// #endif
// #if REPS == 2
//                     temp += RI2_cpu(I_single, n[0]);
// #endif
// #if REPS == 3
//                     temp += RI3_cpu(I_single, n[0]);
// #endif
// #if REPS == 4
                    temp += RI4_cpu(I_single, n[0]);
// #endif
                    delete [] I_single;
                }   
                else if(D == 1)
                {
// #if REPS == 0
//                     temp += RI0_gpu(workArray.cDField(), nx);
// #endif
// #if REPS == 1
//                     temp += RI1_gpu(workArray.cDField(), nx);
// #endif
// #if REPS == 2
//                     temp += RI2_gpu(workArray.cDField(), nx);
// #endif
// #if REPS == 3
//                     temp += RI3_gpu(workArray.cDField(), nx);
// #endif
// #if REPS == 4
                    temp += RI4_gpu(workArray.cDField(), nx);
// #endif
                }
                // S_ += reductionH(workArray, ns);
            }
//            std::cout << "S        =    " << S_ << "\n";
//            std::cout << "E        =    " << E_ << "\n";

            fHelmholtz_ -= temp;
            S_ = E_ - fHelmholtz_;

            // Compute pressure
            pressure_ = -fHelmholtz_;
            for (int i = 0; i < np; ++i) {
                polymerPtr = &mixture().polymer(i);
                phi = polymerPtr->phi();
                mu = polymerPtr->mu();
                length = polymerPtr->length();
                pressure_ += mu * phi /length;
            }

//            double c1[nx], c2[nx], f=0 ,dx;
//            cudaMemcpy(c1, cFieldRGrid(0).cDField(), nx* sizeof(cudaReal), cudaMemcpyDeviceToHost);
//            cudaMemcpy(c2, cFieldRGrid(1).cDField(), nx* sizeof(cudaReal), cudaMemcpyDeviceToHost);
//
//            dx = 3.71368712341147/nx;
//
//            for(int i = 0; i < nx; ++i)
//            {
//                f += c1[i]*c2[i]*dx;
//            }
//            f *= 15.0;
//            f /= 3.71368712341147;
//            std::cout << "f = " << -f+mu << "\n";
        }

        template <int D>
        void System<D>::outputThermo(std::ostream& out)
        {
            out << std::endl;
            out << "fHelmholtz = " << Dbl(fHelmholtz(), 21, 16) << std::endl;
            out << "pressure   = " << Dbl(pressure(), 21, 16) << std::endl;
            out << "TS         = " << Dbl(S_, 21, 16) << std::endl;
            out << "U          = " << Dbl(E_, 21, 16) << std::endl;
            out << std::endl;

            out << "Polymers:" << std::endl;
            out << "    i"
                << "        phi[i]      "
                << "        mu[i]       "
                << std::endl;
            for (int i = 0; i < mixture().nPolymer(); ++i) {
                out << Int(i, 5)
                    << "  " << Dbl(mixture().polymer(i).phi(),18, 11)
                    << "  " << Dbl(mixture().polymer(i).mu(), 18, 11)
                    << std::endl;
            }
            out << std::endl;
        }

        template <int D>
        cudaReal System<D>::innerProduct(const RDField<D>& a, const RDField<D>& b, int size) {

            switch(THREADS_PER_BLOCK){
                case 512:
                    deviceInnerProduct<512>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                            (d_kernelWorkSpace_, a.cDField(), b.cDField(), size);
                    break;
                case 256:
                    std::cout<<"case " << THREADS_PER_BLOCK << " has been called! " <<"\n";
                    deviceInnerProduct<256>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                            (d_kernelWorkSpace_, a.cDField(), b.cDField(), size);
                    break;
                case 128:
                    deviceInnerProduct<128>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                            (d_kernelWorkSpace_, a.cDField(), b.cDField(), size);
                    break;
                case 64:
                    deviceInnerProduct<64>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                            (d_kernelWorkSpace_, a.cDField(), b.cDField(), size);
                    break;
                case 32:
                    deviceInnerProduct<32>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                            (d_kernelWorkSpace_, a.cDField(), b.cDField(), size);
                    break;
                case 16:
                    deviceInnerProduct<16>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                            (d_kernelWorkSpace_, a.cDField(), b.cDField(), size);
                    break;
                case 8:
                    deviceInnerProduct<8>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                            (d_kernelWorkSpace_, a.cDField(), b.cDField(), size);
                    break;
                case 4:
                    deviceInnerProduct<4>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                            (d_kernelWorkSpace_, a.cDField(), b.cDField(), size);
                    break;
                case 2:
                    deviceInnerProduct<2>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                            (d_kernelWorkSpace_, a.cDField(), b.cDField(), size);
                    break;
                case 1:
                    deviceInnerProduct<1>
                    <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>
                            (d_kernelWorkSpace_, a.cDField(), b.cDField(), size);
                    break;
//                default:
//                    std::cout <<"Error : Unexpected THREADS_PER_BLOCK" << "\n";
//                    exit(1);
            }
            cudaMemcpy(kernelWorkSpace_,
                       d_kernelWorkSpace_,
                       NUMBER_OF_BLOCKS * sizeof(cudaReal),
                       cudaMemcpyDeviceToHost);
            cudaReal final = 0;
            // cudaReal c = 0;
            // // use kahan summation to reduce error
            // for (int i = 0; i < NUMBER_OF_BLOCKS; ++i) {
            //     cudaReal y = kernelWorkSpace_[i] - c;
            //     cudaReal t = final + y;
            //     c = (t - final) - y;
            //     final = t;

            // }
           for(int i = 0; i < NUMBER_OF_BLOCKS; ++i)
           {
               final += kernelWorkSpace_[i];
           }

            return final;
        }

        template<int D>
        cudaReal System<D>::reductionH(RDField<D>& a, int size) {
            reduction <<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal) >>>
                    (d_kernelWorkSpace_, a.cDField(), size);
            cudaMemcpy(kernelWorkSpace_, d_kernelWorkSpace_, NUMBER_OF_BLOCKS * sizeof(cudaReal), cudaMemcpyDeviceToHost);
            cudaReal final = 0;
//            cudaReal c = 0;
//            for (int i = 0; i < NUMBER_OF_BLOCKS; ++i) {
//                cudaReal y = kernelWorkSpace_[i] - c;
//                cudaReal t = final + y;
//                c = (t - final) - y;
//                final = t;
//            }

            for(int i = 0; i < NUMBER_OF_BLOCKS; ++i)
            {
                final += kernelWorkSpace_[i];
            }
            return final;
        }
    
// #if REPS == 4 
        template <int D>
        cudaReal System<D>::RI4_gpu(cudaReal *f, int size)
        {
            cudaReal *I0_dev, *I1_dev,*I2_dev,*I3_dev,*I4_dev,
                     I0, I1, I2, I3, I4, 
                      dm, I;

            cudaMalloc((void**)&I0_dev, 1 * sizeof(cudaReal));
            cudaMalloc((void**)&I1_dev, 1 * sizeof(cudaReal));
            cudaMalloc((void**)&I2_dev, 1 * sizeof(cudaReal));
            cudaMalloc((void**)&I3_dev, 1 * sizeof(cudaReal));
            cudaMalloc((void**)&I4_dev, 1 * sizeof(cudaReal));

            dm = 1.0/double(size+0.5);

            device_RI4 <<<1,32>>>(f, size, I0_dev, I1_dev, I2_dev, I3_dev, I4_dev);

            cudaMemcpy(&I0, I0_dev, sizeof(cudaReal), cudaMemcpyDeviceToHost);
            cudaMemcpy(&I1, I1_dev, sizeof(cudaReal), cudaMemcpyDeviceToHost);
            cudaMemcpy(&I2, I2_dev, sizeof(cudaReal), cudaMemcpyDeviceToHost);
            cudaMemcpy(&I3, I3_dev, sizeof(cudaReal), cudaMemcpyDeviceToHost);
            cudaMemcpy(&I4, I4_dev, sizeof(cudaReal), cudaMemcpyDeviceToHost);

            I0 *= dm;
            I1 *= (2.0*dm);
            I2 *= (4.0*dm);
            I3 *= (8.0*dm);
            I4 *= (16.0*dm);

            I = (1048576.0 * I0 - 348160.0 * I1 + 22848.0 * I2 - 340.0 * I3+ I4) / 722925.0;

            cudaFree(I0_dev);
            cudaFree(I1_dev);
            cudaFree(I2_dev);
            cudaFree(I3_dev);
            cudaFree(I4_dev);
            
            return  I;
        }

        template <int D>
        cudaReal System<D>::RI4_cpu(double *f, int size)
        { 
            double I0, I1, I2, I3, I4, dm;

            // With PBC, we have f[0] = (f[0] + f[size])/2
            I0 = (f[0])/2;  
            I1 = (f[0])/2;  
            I2 = (f[0])/2;   
            I3 = (f[0])/2;  
            I4 = (f[0])/2;   
            dm = 1.0/double(size+0.5);

            for(int i = 1; i < size; i+=1)
            {
                I0 += f[i]; 
                if (i/2*2 == i)
                    I1 += f[i];
                if (i/4*4 == i)
                    I2 += f[i];
                if (i/8*8 == i)
                    I3 += f[i];    
                if (i/16*16 == i)
                    I4 += f[i];
            }

            // I0 *= dm;
            I1 *= 2.0;
            I2 *= 4.0;
            I3 *= 8.0;
            I4 *= 16.0;

            return dm*(1048576.0*I0-348160.0*I1+22848.0*I2-340.0*I3+I4)/722925.0;
        }
// #endif
// #if REPS == 3
//         template <int D>
//         cudaReal System<D>::RI3_gpu(cudaReal *f, int size)
//         {
//             cudaReal *I0_dev, *I1_dev,*I2_dev,*I3_dev,
//                      I0, I1, I2, I3,
//                       dm, I;

//             cudaMalloc((void**)&I0_dev, 1 * sizeof(cudaReal));
//             cudaMalloc((void**)&I1_dev, 1 * sizeof(cudaReal));
//             cudaMalloc((void**)&I2_dev, 1 * sizeof(cudaReal));
//             cudaMalloc((void**)&I3_dev, 1 * sizeof(cudaReal));

//             dm = 1.0/double(size);

//             deviceRI3 <<<1,32>>>(f, size, I0_dev, I1_dev, I2_dev, I3_dev);

//             cudaMemcpy(&I0, I0_dev, sizeof(cudaReal), cudaMemcpyDeviceToHost);
//             cudaMemcpy(&I1, I1_dev, sizeof(cudaReal), cudaMemcpyDeviceToHost);
//             cudaMemcpy(&I2, I2_dev, sizeof(cudaReal), cudaMemcpyDeviceToHost);
//             cudaMemcpy(&I3, I3_dev, sizeof(cudaReal), cudaMemcpyDeviceToHost);

//             I0 *= dm;
//             I1 *= (2.0*dm);
//             I2 *= (4.0*dm);
//             I3 *= (8.0*dm);

//             I = (4096.0 * I0 - 1344.0 * I1 + 84.0 * I2 - I3) / 2835.0;

//             cudaFree(I0_dev);
//             cudaFree(I1_dev);
//             cudaFree(I2_dev);
//             cudaFree(I3_dev);
            
//             return  I;
//         }

//         template <int D>
//         cudaReal System<D>::RI3_cpu(double *f, int size)
//         { 
//             double I0, I1, I2, I3, dm;

//             // With PBC, we have f[0] = (f[0] + f[size])/2
//             I0 = f[0]; 
//             I1 = f[0];
//             I2 = f[0];
//             I3 = f[0];
//             dm = 1.0/double(size);

//             for(int i = 1; i < size; i+=1)
//             {
//                 I0 += f[i]; 
//                 if (i/2*2 == i)
//                     I1 += f[i];
//                 if (i/4*4 == i)
//                     I2 += f[i];
//                 if (i/8*8 == i)
//                     I3 += f[i];    
//             }

//             // I0 *= dm;
//             I1 *= 2.0;
//             I2 *= 4.0;
//             I3 *= 8.0;

//             return dm*(4096.0*I0-1344.0*I1+84.0*I2-I3)/2835.0;
//         }
// #endif
// #if REPS == 2
//         template <int D>
//         cudaReal System<D>::RI2_gpu(cudaReal *f, int size)
//         {
//             cudaReal *I0_dev, *I1_dev,*I2_dev,
//                      I0, I1, I2,
//                       dm, I;

//             cudaMalloc((void**)&I0_dev, 1 * sizeof(cudaReal));
//             cudaMalloc((void**)&I1_dev, 1 * sizeof(cudaReal));
//             cudaMalloc((void**)&I2_dev, 1 * sizeof(cudaReal));

//             dm = 1.0/double(size);

//             deviceRI2 <<<1,32>>>(f, size, I0_dev, I1_dev, I2_dev);

//             cudaMemcpy(&I0, I0_dev, sizeof(cudaReal), cudaMemcpyDeviceToHost);
//             cudaMemcpy(&I1, I1_dev, sizeof(cudaReal), cudaMemcpyDeviceToHost);
//             cudaMemcpy(&I2, I2_dev, sizeof(cudaReal), cudaMemcpyDeviceToHost);

//             I0 *= dm;
//             I1 *= (2.0*dm);
//             I2 *= (4.0*dm);

//             I = (64.0 * I0 - 20.0 * I1 + I2) / 45.0;

//             cudaFree(I0_dev);
//             cudaFree(I1_dev);
//             cudaFree(I2_dev);
            
//             return  I;
//         }

//         template <int D>
//         cudaReal System<D>::RI2_cpu(double *f, int size)
//         { 
//             double I0, I1, I2, dm;

//             // With PBC, we have f[0] = (f[0] + f[size])/2
//             I0 = f[0]; 
//             I1 = f[0];
//             I2 = f[0];
//             dm = 1.0/double(size);

//             for(int i = 1; i < size; i+=1)
//             {
//                 I0 += f[i]; 
//                 if (i/2*2 == i)
//                     I1 += f[i];
//                 if (i/4*4 == i)
//                     I2 += f[i]; 
//             }

//             // I0 *= dm;
//             I1 *= 2.0;
//             I2 *= 4.0;

//             return dm*(64.0*I0 - 20.0*I1 + I2)/45.0;
//         }
// #endif
// #if REPS == 1
//         template <int D>
//         cudaReal System<D>::RI1_gpu(cudaReal *f, int size)
//         {
//             cudaReal *I0_dev, *I1_dev,
//                      I0, I1, 
//                       dm, I;

//             cudaMalloc((void**)&I0_dev, 1 * sizeof(cudaReal));
//             cudaMalloc((void**)&I1_dev, 1 * sizeof(cudaReal));

//             dm = 1.0/double(size);

//             deviceRI1 <<<1,32>>>(f, size, I0_dev, I1_dev);

//             cudaMemcpy(&I0, I0_dev, sizeof(cudaReal), cudaMemcpyDeviceToHost);
//             cudaMemcpy(&I1, I1_dev, sizeof(cudaReal), cudaMemcpyDeviceToHost);

//             I0 *= dm;
//             I1 *= (2.0*dm);

//             I = (4.0 * I0 - I1) / 3.0;

//             cudaFree(I0_dev);
//             cudaFree(I1_dev);
            
//             return  I;
//         }

//         template <int D>
//         cudaReal System<D>::RI1_cpu(double *f, int size)
//         { 
//             double I0, I1, dm;

//             // With PBC, we have f[0] = (f[0] + f[size])/2
//             I0 = f[0]; 
//             I1 = f[0];
//             dm = 1.0/double(size);

//             for(int i = 1; i < size; i+=1)
//             {
//                 I0 += f[i]; 
//                 if (i/2*2 == i)
//                     I1 += f[i]; 
//             }

//             // I0 *= dm;
//             I1 *= 2.0;

//             return dm*(4.0*I0-I1)/3.0;
//         }
// #endif
// #if REPS == 0
//         template <int D>
//         cudaReal System<D>::RI0_gpu(cudaReal *f, int size)
//         {
//             cudaReal *I0_dev,
//                       I0, 
//                       dm, I;

//             cudaMalloc((void**)&I0_dev, 1 * sizeof(cudaReal));

//             dm = 1.0/double(size);

//             deviceRI0 <<<1,32>>>(f, size, I0_dev);

//             cudaMemcpy(&I0, I0_dev, sizeof(cudaReal), cudaMemcpyDeviceToHost);

//             I0 *= dm;

//             I = I0;

//             cudaFree(I0_dev);
            
//             return  I;
//         }

//         template <int D>
//         cudaReal System<D>::RI0_cpu(double *f, int size)
//         { 
//             double I0, dm;

//             // With PBC, we have f[0] = (f[0] + f[size])/2
//             I0 = f[0]; 
//             dm = 1.0/double(size);

//             for(int i = 1; i < size; i+=1)
//             {
//                 I0 += f[i]; 
//             }

//             return dm*I0;
//         }
// #endif
    }
}
#endif
