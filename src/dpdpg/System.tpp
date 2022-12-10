#ifndef DPDPG_SYSTEM_TPP
#define DPDPG_SYSTEM_TPP

#include "System.h"

#include <pspg/GpuResources.h>

#include <pscf/crystal/shiftToMinimum.h>

#include <util/format/Str.h>
#include <util/format/Int.h>
#include <util/format/Dbl.h>

#include <string>
#include <getopt.h>

// Global variable for kernels
int THREADS_PER_BLOCK;
int NUMBER_OF_BLOCKS;

static
__global__
void scaleRealconst(cudaReal* a, cudaReal scale, int size)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = startID; i < size; i += nThreads)
    {
        a[i] *= scale;
    }
}

static
__global__
void scaleComplex(cudaComplex* a, cudaReal* scale, int size)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = startID; i < size; i += nThreads)
    {
        a[i].x *= scale[i];
        a[i].y *= scale[i];
    }
}

static
__global__
void scaleReal(cudaReal* a, cudaReal* scale, int size)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = startID; i < size; i += nThreads)
    {
        a[i] *= scale[i];
    }

}


__global__
void mulComplex(cudaComplex* res, cudaComplex* a, cudaComplex* b, int size)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = startID; i < size; i += nThreads)
    {
        res[i].x = a[i].x*b[i].x - a[i].y*b[i].y;
        res[i].y = a[i].x*b[i].y + a[i].y*b[i].x;
    }
}

namespace Pscf
{
namespace Pspg
{
    namespace DPDpg
    {
        using namespace Util;

        /*
         * Constructor
         */
        template <int D>
        System<D>::System()
        : mesh_(),
          wavelistPtr_(nullptr),
          basisPtr_(nullptr),
          groupName_(""),
          unitCell_(),
          fileMaster_(),
          iteratorPtr_(nullptr),
          interactionPtr_(nullptr),
          hasMesh_(false),
          hasUnitCell_(false),
          comp_(true)
        {
            setClassName("System");
            iteratorPtr_ = new AmIterator<D>(this);
            interactionPtr_ = new ChiInteraction();
            basisPtr_ = new Basis<D>();
            wavelistPtr_ = new WaveList<D>();
        }

        /*
         * Destructor
         */
        template <int D>
        System<D>::~System()
        {
            delete iteratorPtr_;
            delete wavelistPtr_;
            delete basisPtr_;
            delete interactionPtr_;
            delete [] kernelWorkSpace_;
        }

        /*
         * Process command line options.
         */
        template <int D>
        void System<D>::setOptions(int argc, char **argv)
        {
            bool eflag = false;  // echo
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

            int c;
            opterr = 0;
            while ((c = getopt(argc, argv, "er:p:c:i:o:f1:2:")) != -1)
            {
                switch (c) 
                {
                    case 'e':
                        eflag = true;
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
            if (eflag) 
            {
                Util::ParamComponent::setEcho(true);
            }
            // If option -p, set parameter file name
            if (pFlag)
            {
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
            
        }

        /*
        * Read parameters and initialize.
        */
        template<int D>
        void System<D>::readParameters(std::istream &in)
        {
            readParamComposite(in, dpddiblock());

            comp_ = dpddiblock().compressibility();

            // Read unit cell type and its parameters
            read(in, "unitCell", unitCell_);
            hasUnitCell_ = true;

            /// Read crystallographic unit cell (used only to create basis)
            read(in, "mesh", mesh_);
            dpddiblock().setMesh(mesh());
            hasMesh_ = true;

            // Construct wavelist
            wavelist().allocate(mesh(), unitCell());
            wavelist().computeMinimumImages(mesh(), unitCell());
            dpddiblock().setupUnitCell(unitCell(), wavelist());


            // Read group name, construct basis
            read(in, "groupName", groupName_);
            basis().makeBasis(mesh(), unitCell(), groupName_);
            fieldIo().associate(unitCell_, mesh_, fft_, groupName_,
                                basis(), fileMaster_);
            dpddiblock().allocate(basis());
            dpddiblock().setBasis(basis(), unitCell_);
            
            allocate();

            // Initialize iterator
            readParamComposite(in, iterator());
            iterator().allocate();
        }

        template <int D>
        void System<D>::allocate()
        {
            UTIL_CHECK(hasMesh_)

            int nMonomer = dpddiblock().nMonomer();
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

            workArray.allocate(mesh().size());

            cudaMalloc((void**)&d_kernelWorkSpace_, NUMBER_OF_BLOCKS * sizeof(cudaReal));
            kernelWorkSpace_ = new cudaReal[NUMBER_OF_BLOCKS];

            isAllocated_ = true;
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
                    std::ofstream outRunfile;
                    outRunfile.open("run", std::ios::app);
                    if (compressibility() == true)
                    {
                        outRunfile <<std::endl;
                        outRunfile <<std::endl;
                        outRunfile << "N     = " << dpddiblock().getN() << std::endl;
                        outRunfile << "NA    = " << dpddiblock().getNA() << std::endl;
                        outRunfile << "chiN  = " << dpddiblock().chiN() << std::endl;
                        outRunfile << "eps   = " << dpddiblock().eps() << std::endl;
                        outRunfile << "sigma = " << dpddiblock().sigma() << std::endl;
                        outRunfile << "kpN   = " << dpddiblock().kpN() << std::endl;
                        outRunfile << "Space group " << groupName() << std::endl;
                        outRunfile <<std::endl;
                        outRunfile << "       fHelmholtz                 U                     UAB                    UCMP                   SA                     SB              error      cell param1     cell param2     cell param3" << std::endl;
                        outRunfile << "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" <<std::endl;
                        outRunfile <<std::endl;
                    }
                    else
                    {
                        outRunfile <<std::endl;
                        outRunfile <<std::endl;
                        outRunfile << "N     = " << dpddiblock().getN() << std::endl;
                        outRunfile << "NA    = " << dpddiblock().getNA() << std::endl;
                        outRunfile << "chiN  = " << dpddiblock().chiN() << std::endl;
                        outRunfile << "eps   = " << dpddiblock().eps() << std::endl;
                        outRunfile << "sigma = " << dpddiblock().sigma() << std::endl;
                        outRunfile << "Space group " << groupName() << std::endl;
                        outRunfile <<std::endl;
                        outRunfile << "       fHelmholtz                 U                      SA                     SB              error      cell param1     cell param2     cell param3" << std::endl;
                        outRunfile << "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" <<std::endl;
                        outRunfile <<std::endl;
                    }
                    

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
                        if (compressibility() == true)
                        {
                            outRunfile << Dbl(fHelmholtz_, 23, 14)
                                       << Dbl(U_, 23, 14)
                                       << Dbl(UAB_, 23, 14)
                                       << Dbl(UCMP_, 23, 14)
                                       << Dbl(SA_, 23, 14) 
                                       << Dbl(SB_, 23, 14) 
                                       << Dbl(iterator().final_error, 11, 2);
                        }
                        else
                        {
                            outRunfile << Dbl(fHelmholtz_, 23, 14)
                                       << Dbl(U_, 23, 14)
                                       << Dbl(SA_, 23, 14) 
                                       << Dbl(SB_, 23, 14) 
                                       << Dbl(iterator().final_error, 11, 2);
                        }
                        for(int i = 0; i < dpddiblock().nParam(); ++i)
                            outRunfile << Dbl(unitCell().parameter(i), 17, 8);
                        outRunfile << std::endl;
                        outRunfile.close();
                    }
                    else
                    {
                        Log::file() << "Iterate has failed. Exiting "<<std::endl;
                        outRunfile.close();
                    }
                    
                }
                else if (command == "ITERATE(3m)")
                {
                    std::cout << "Accelerate the cuFFT using crystallographic FFT (3m)" 
                              << std::endl;
                    shrinkWField();

                    std::ofstream outRunfile;
                    outRunfile.open("run", std::ios::app);
                    outRunfile <<std::endl;
                    outRunfile <<std::endl;
                    outRunfile << "N     = " << dpddiblock().getN() << std::endl;
                    outRunfile << "NA    = " << dpddiblock().getNA() << std::endl;
                    outRunfile << "chiN  = " << dpddiblock().chiN() << std::endl;
                    outRunfile << "eps   = " << dpddiblock().eps() << std::endl;
                    outRunfile << "sigma = " << dpddiblock().sigma() << std::endl;
                    outRunfile << "Space group " << groupName() << std::endl;
                    outRunfile <<std::endl;
                    outRunfile << "       fHelmholtz                 U                     SA                     SB              error      cell param1     cell param2     cell param3" << std::endl;
                    outRunfile << "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" <<std::endl;
                    outRunfile <<std::endl;

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
                    int fail = iterator().solve_3m();
                    hasCFields_ = true;

                    outRunfile.close();
                    exit(1);
                }
                else if (command == "CHI_PATH")
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

                    in >> end_chi
                       >> inc 
                       >> max_inc
                       >> min_inc
                       >> fac
                       >> filename;

                    outRunfile.open(filename, std::ios::app);

                    outRunfile <<std::endl;
                    outRunfile << "        chiN               fHelmholtz                 U                    UAB                    UCMP                   S                     SA                     SB              error      cell param1     cell param2     cell param3" << std::endl;
                    outRunfile << "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" <<std::endl;
                    outRunfile <<std::endl;

                    if (compressibility() == true)
                    {
                        outRunfile <<std::endl;
                        outRunfile << "        chiN               fHelmholtz                 U                    UAB                    UCMP                   S                     SA                     SB              error      cell param1     cell param2     cell param3" << std::endl;
                        outRunfile << "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" <<std::endl;
                        outRunfile <<std::endl;
                    }
                    else
                    {
                        outRunfile <<std::endl;
                        outRunfile << "        chiN               fHelmholtz                 U                     S                     SA                     SB              error      cell param1     cell param2     cell param3" << std::endl;
                        outRunfile << "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" <<std::endl;
                        outRunfile <<std::endl;
                    }
                    
                    start_chi = dpddiblock().chiN();
                    current_chi = start_chi;

                    std::cout << "Run file name is " << filename << std::endl;

                    Log::file() << "Calculation along the path with respect to chiN:" 
                                << std::endl;
                    Log::file() << "Starting point of chiN: " 
                                << start_chi << std::endl;
                    Log::file() << " Current point of chiN: "  
                                << current_chi << std::endl;
                    Log::file() << "    Stop point of chiN: "
                                << end_chi << std::endl;
                    Log::file() << "     Initial increment: "
                                << inc << std::endl;

                    if(start_chi == end_chi)
                    {
                        Log::file() << "The start point equals to the stop point." 
                                    << std::endl;
                        exit(1);
                    }

                    if(start_chi > end_chi)
                    {
                        inc *= -1;
                        dir = false;
                    }

                    while(!isFinished)
                    {
                        // Step "forward" (of courese inc can be negative)
                        current_chi += inc;
                        if((dir&&current_chi >= end_chi) || ((!dir)&&current_chi <= end_chi))
                        {
                            current_chi = end_chi;
                            isFinished = true;
                        }
                        dpddiblock().setChi(current_chi);
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
                            
                            if (compressibility() == true)
                            {
                                outRunfile << Dbl(current_chi, 19, 10) 
                                           << Dbl(fHelmholtz_, 23, 14)
                                           << Dbl(U_, 23, 14)
                                           << Dbl(UAB_, 23, 14)
                                           << Dbl(UCMP_, 23, 14)
                                           << Dbl(SA_+SB_, 23, 14) 
                                           << Dbl(SA_, 23, 14) 
                                           << Dbl(SB_, 23, 14) 
                                           << Dbl(iterator().final_error, 11, 2);
                            }
                            else
                            {
                                outRunfile << Dbl(current_chi, 19, 10) 
                                           << Dbl(fHelmholtz_, 23, 14)
                                           << Dbl(U_, 23, 14)
                                           << Dbl(SA_+SB_, 23, 14) 
                                           << Dbl(SA_, 23, 14) 
                                           << Dbl(SB_, 23, 14) 
                                           << Dbl(iterator().final_error, 11, 2);
                            }

                            for(int i = 0; i < dpddiblock().nParam(); ++i)
                                outRunfile << Dbl(unitCell().parameter(i), 17, 8);
                            outRunfile << std::endl;

                            if (abs(inc*fac) <= max_inc)
                                inc *= fac;
                            else
                            {
                                if(dir)
                                    inc = max_inc;
                                else
                                    inc = -max_inc;
                            }
                        }
                        else
                        {
                            Log::file() << "Iterate has failed."<<std::endl;
                            if(inc > min_inc)
                            {
                                current_chi -= inc;
                                inc /= fac;
                            }
                            else
                            {
                                Log::file() << "Smallest increment reached."<<std::endl;
                                exit(1);
                            }
                        }
                    }

                }
                else if (command == "EPS_PATH")
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

                    double start_eps, current_eps, end_eps;
                    // current increment, maximum increment, minimum increment,
                    // and adpative factor, which are read from command file
                    double inc, max_inc, min_inc, fac;
                    // Is finished? Set false initially
                    bool isFinished = false;
                    // Is chi increasing? Set true initially
                    bool dir = true;

                    std::ofstream outRunfile;;

                    in >> end_eps
                       >> inc
                       >> max_inc
                       >> min_inc
                       >> fac
                       >> filename;

                    outRunfile.open(filename, std::ios::app);

                    if (compressibility() == true)
                    {
                        outRunfile <<std::endl;
                        outRunfile << "        eps                fHelmholtz                 U                    UAB                    UCMP                   S                     SA                     SB              error      cell param1     cell param2     cell param3" << std::endl;
                        outRunfile << "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" <<std::endl;
                        outRunfile <<std::endl;
                    }
                    else
                    {
                        outRunfile <<std::endl;
                        outRunfile << "        eps               fHelmholtz                 U                     S                     SA                     SB              error      cell param1     cell param2     cell param3" << std::endl;
                        outRunfile << "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" <<std::endl;
                        outRunfile <<std::endl;
                    }

                    start_eps = dpddiblock().eps();
                    current_eps = start_eps;

                    std::cout << "Run file name is " << filename << std::endl;
                     Log::file() << "Calculation along the path with respect to eps:" 
                                << std::endl;
                    Log::file() << "Starting point of eps: " 
                                << start_eps << std::endl;
                    Log::file() << " Current point of eps: "  
                                << current_eps << std::endl;
                    Log::file() << "    Stop point of eps: "
                                << end_eps << std::endl;
                    Log::file() << "    Initial increment: "
                                << inc << std::endl;

                    if(start_eps == end_eps)
                    {
                        Log::file() << "The start point equals to the stop point." 
                                    << std::endl;
                        exit(1);
                    }

                    if(start_eps > end_eps)
                    {
                        inc *= -1;
                        dir = false;
                    }

                    while(!isFinished)
                    {
                        // Step "forward" (of courese inc can be negative)
                        current_eps += inc;
                        if((dir&&current_eps >= end_eps) || ((!dir)&&current_eps <= end_eps))
                        {
                            current_eps = end_eps;
                            isFinished = true;
                        }
                        dpddiblock().setEps(current_eps);

                        // Attempt to iteratively solve SCFT equations
                        Log::file()<<std::endl;
                        Log::file()<< "================================================" << std::endl;
                        Log::file() << "*Current eps = " << current_eps << "*" << std::endl<<std::endl;
                        int fail = iterator().solve();
                        hasCFields_ = true;

                        if (!fail)
                        {
                            computeFreeEnergy();
                            outputThermo(Log::file());
                            if (compressibility() == true)
                            {
                                outRunfile << Dbl(current_eps, 19, 10) 
                                           << Dbl(fHelmholtz_, 23, 14)
                                           << Dbl(U_, 23, 14)
                                           << Dbl(UAB_, 23, 14)
                                           << Dbl(UCMP_, 23, 14)
                                           << Dbl(SA_+SB_, 23, 14) 
                                           << Dbl(SA_, 23, 14) 
                                           << Dbl(SB_, 23, 14) 
                                           << Dbl(iterator().final_error, 11, 2);
                            }
                            else
                            {
                                outRunfile << Dbl(current_eps, 19, 10) 
                                           << Dbl(fHelmholtz_, 23, 14)
                                           << Dbl(U_, 23, 14)
                                           << Dbl(SA_+SB_, 23, 14) 
                                           << Dbl(SA_, 23, 14) 
                                           << Dbl(SB_, 23, 14) 
                                           << Dbl(iterator().final_error, 11, 2);
                            }

                            for(int i = 0; i < dpddiblock().nParam(); ++i)
                                outRunfile << Dbl(unitCell().parameter(i), 17, 8);
                            outRunfile << std::endl;

                            if (abs(inc*fac) <= max_inc)
                                inc *= fac;
                            else
                            {
                                if(dir)
                                    inc = max_inc;
                                else
                                    inc = -max_inc;
                            }
                        }
                        else
                        {
                            Log::file() << "Iterate has failed."<<std::endl;
                            if(inc > min_inc)
                            {
                                current_eps -= inc;
                                inc /= fac;
                            }
                            else
                            {
                                Log::file() << "Smallest increment reached."<<std::endl;
                                exit(1);
                            }
                        }
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
                else if (command == "WRITE_C_KGRID")
                {
                    UTIL_CHECK(hasCFields_)
                    fft_.forwardTransform(cFieldsRGrid_[0], cFieldsKGrid_[0]);
                    fft_.forwardTransform(cFieldsRGrid_[1], cFieldsKGrid_[1]);
                    in >> filename;
                    Log::file() << "  " << Str(filename, 20) << std::endl;
                    fieldIo().writeFieldsKGrid(filename, cFieldsKGrid());
                    
                }
                else if (command == "BASIS_TO_RGRID")
                {
                    hasCFields_ = false;

                    std::string inFileName;
                    in >> inFileName;
                    Log::file() << " " << Str(inFileName, 20) <<std::endl;

                    fieldIo().readFieldsBasis(inFileName, cFields());
                    fieldIo().convertBasisToRGrid(cFields(), cFieldsRGrid());

                    std::string outFileName;
                    in >> outFileName;
                    Log::file() << " " << Str(outFileName, 20) <<std::endl;
                    fieldIo().writeFieldsRGrid(outFileName, cFieldsRGrid());
                }
                else if (command == "RGRID_TO_BASIS")
                {
                    hasCFields_ = false;

                    std::string inFileName;

                    in >> inFileName;
                    Log::file() << " " << Str(inFileName, 20) <<std::endl;
                    fieldIo().readFieldsRGrid(inFileName, cFieldsRGrid());

                    fieldIo().convertRGridToBasis(cFieldsRGrid(), cFields());

                    std::string outFileName;
                    in >> outFileName;
                    Log::file() << " " << Str(outFileName, 20) <<std::endl;
                    fieldIo().writeFieldsBasis(outFileName, cFields());

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
                else if (command == "WRITE_CN_RGRID")
                {
                    UTIL_CHECK(hasCFields_)
                    in >> filename;
                    Log::file() << "  " << Str(filename, 20) << std::endl;

                    int nx = mesh().size();
                    int nMonomer = dpddiblock().nMonomer();
                    cFieldsRGridNorm_.allocate(nMonomer);
                    for(int i = 0; i < nMonomer; ++i)
                    {
                        cFieldNRGrid(i).allocate(nx);
                    }

                    normalization<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (cFieldNRGrid(0).cDField(), 
                     cFieldsRGrid_[0].cDField(), 
                     cFieldsRGrid_[1].cDField(), 
                     nx);
                    normalization<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (cFieldNRGrid(1).cDField(), 
                     cFieldsRGrid_[1].cDField(), 
                     cFieldsRGrid_[0].cDField(), 
                     nx);

                    fieldIo().writeFieldsRGrid(filename, cFieldsNRGrid());

                    for(int i = 0; i < nMonomer; ++i)
                    {
                        cFieldNRGrid(i).deallocate();
                    }
                    cFieldsRGridNorm_.deallocate();
                }
                else
                {
                    Log::file() << "  Error: Unknown command  " << command << std::endl;
                    readNext = false;
                }

            }
        }

        template <int D>
        void System<D>::shrinkWField()
        {   
            int meshVec[D], meshVecHalf[D];
            int nx = mesh().size();
            double *wA_ini, *wB_ini, *wA, *wB;

            wA_ini = new double [nx];
            wB_ini = new double [nx];
            wA     = new double [nx/8];
            wB     = new double [nx/8];

            for (int i = 0; i < D; ++i)
            {
                meshVec[i] = this->mesh().dimensions()[i];
                meshVecHalf[i] = meshVec[i]/2;
            }

            cudaMemcpy(wA_ini, 
                       wFieldsRGrid_[0].cDField(), 
                       nx*sizeof(double),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(wB_ini, 
                       wFieldsRGrid_[1].cDField(), 
                       nx*sizeof(double),
                       cudaMemcpyDeviceToHost);
            
            for (int i = 0; i < meshVecHalf[0]; ++i)
            {
                for (int j = 0; j < meshVecHalf[1]; ++j)
                {
                    for (int k = 0; k < meshVecHalf[2]; ++k)
                    {
                        int idx_src = (2 * i * meshVecHalf[1] * 2 + 2 * j) * meshVecHalf[2] * 2 + 2 * k;
                        int idx_dst = (i * meshVecHalf[1] + j) * meshVecHalf[2] + k;
                        wA[idx_dst] = wA_ini[idx_src];
					    wB[idx_dst] = wB_ini[idx_src];
                    }
                }
            }

            delete [] wA_ini;
            delete [] wB_ini;
            delete [] wA;
            delete [] wB;
            
        }

        template <int D>
        void System<D>::computeFreeEnergy()
        {
            fHelmholtz_ = 0.0;

            int nx = mesh().size();
            int ns = basis().nStar();
            assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (workArray.cDField(), 0.5, nx);
            inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (workArray.cDField(), cFieldsRGrid_[0].cDField(), nx);
            inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (workArray.cDField(), wFieldsRGrid_[0].cDField(), nx);

            int N = dpddiblock().getN(), NA = dpddiblock().getNA();

            double *wA, *wB, *cA, *cB,
                   *SAint, *SBint, *Uint;
            double lnQ = std::log(dpddiblock().Q());
            wA = new double [nx];
            wB = new double [nx];
            cA = new double [nx];
            cB = new double [nx];
            SAint = new double [nx];
            SBint = new double [nx];
            Uint = new double [nx];
            cudaMemcpy(wA, wFieldsRGrid_[0].cDField(), nx*sizeof(double),cudaMemcpyDeviceToHost);
            cudaMemcpy(wB, wFieldsRGrid_[1].cDField(), nx*sizeof(double),cudaMemcpyDeviceToHost);
            cudaMemcpy(cA, cFieldsRGrid_[0].cDField(), nx*sizeof(double),cudaMemcpyDeviceToHost);
            cudaMemcpy(cB, cFieldsRGrid_[1].cDField(), nx*sizeof(double),cudaMemcpyDeviceToHost);

            for (int i = 0; i < nx; ++i)
            {;
                SAint[i] = wA[i]*cA[i];
                SBint[i] = wB[i]*cB[i];
                Uint[i]  = wA[i]*cA[i] + wB[i]*cB[i];
            }

            double SA = 0.0, SB= 0.0, U =0.0;
            if (D == 3)
            {
                int n[3];  
                n[0] = 1;  n[1] = 1;  n[2] = 1;
                for (int i = 0; i < 3; ++i)
                {
                    n[i] = mesh().dimensions()[i];
                }
                double *I_single_A, *I_double_A,
                       *I_single_B, *I_double_B,
                       *I_single_U, *I_double_U;
                I_single_A = new double[n[1]*n[2]];
                I_double_A = new double[n[2]];
                I_single_B = new double[n[1]*n[2]];
                I_double_B = new double[n[2]];
                I_single_U = new double[n[1]*n[2]];
                I_double_U = new double[n[2]];
                for (int iz = 0; iz < n[2]; ++iz)
                {
                    for (int iy = 0; iy < n[1]; ++iy)
                    {
                        I_single_A[iy + iz*n[1]] = RombergInt(SAint + n[0]*(iy + iz*n[1]), n[0]);
                        I_single_B[iy + iz*n[1]] = RombergInt(SBint + n[0]*(iy + iz*n[1]), n[0]);
                        I_single_U[iy + iz*n[1]] = RombergInt(Uint + n[0]*(iy + iz*n[1]), n[0]);
                    }
                }
                for (int iz = 0; iz < n[2]; ++iz)
                {
                    I_double_A[iz] = RombergInt(I_single_A + n[1]*iz, n[1]);
                    I_double_B[iz] = RombergInt(I_single_B + n[1]*iz, n[1]);
                    I_double_U[iz] = RombergInt(I_single_U + n[1]*iz, n[1]);
                }
                
                SA = -RombergInt(I_double_A, n[2]) - NA*lnQ/N;
                SB = -RombergInt(I_double_B, n[2])- (N-NA)*lnQ/N;
                U  = 0.5*RombergInt(I_double_U, n[2]);
                
                delete [] I_single_A;
                delete [] I_double_A;
                delete [] I_single_B;
                delete [] I_double_B;
                delete [] I_single_U;
                delete [] I_double_U;

                std::cout << "-sc,A = " << SA << std::endl;
                std::cout << "-sc,B = " << SB << std::endl;
                std::cout << "-sc   = " << SA+SB << std::endl;
                std::cout << " uc   = " << U << std::endl;
                SA_ = SA;
                SB_ = SB;
                U_ = U;
            }
            else if (D == 2)
            {
               int n[2];   // n[3] stores the number of mesh points in each direction
                for (int i = 0; i < 2; ++i)
                {
                    n[i] = mesh().dimensions()[i];
                }
                // Here, the triple integral will be transfromed into single integral 
                double *I_single_A,
                       *I_single_B,
                       *I_single_U;
                I_single_A = new double[n[1]];
                I_single_B = new double[n[1]];
                I_single_U = new double[n[1]];
                for (int ix = 0; ix < n[0]; ++ix)
                {
                    I_single_A[ix] = RombergInt(SAint + n[1]*ix, n[1]);
                    I_single_B[ix] = RombergInt(SBint + n[1]*ix, n[1]);
                    I_single_U[ix] = RombergInt(Uint + n[1]*ix, n[1]);
                }
                SA = -RombergInt(I_single_A, n[0]) - NA*lnQ/N;
                SB = -RombergInt(I_single_B, n[0]) - (N-NA)*lnQ/N;
                U  = 0.5*RombergInt(I_single_U, n[0]);
                delete [] I_single_A;
                delete [] I_single_B;
                delete [] I_single_U;
                std::cout << "-sc,A = " << SA << std::endl;
                std::cout << "-sc,B = " << SB << std::endl;
                std::cout << "-sc   = " << SA+SB << std::endl;
                std::cout << " uc   = " << U << std::endl;
                SA_ = SA;
                SB_ = SB;
                U_ = U;
            }
            else 
            {
                SA = -RombergInt(SAint, nx) - NA*lnQ/N;
                SB = -RombergInt(SBint, nx) - (N-NA)*lnQ/N;
                U =  0.5*RombergInt(Uint, nx);
                SA_ = SA;
                SB_ = SB;
                U_ = U;
            }
            
            delete [] wA;
            delete [] wB;
            delete [] cA;
            delete [] cB;
            delete [] SAint;
            delete [] SBint;
            delete [] Uint;
            // cudaMemcpy(qt, dpddiblock().qt() + N - NA, nx * sizeof(cudaReal),cudaMemcpyDeviceToHost);
            if (D == 3)
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
                        I_single[iy + iz*n[1]] = RombergIntegration(workArray.cDField() + n[0]*(iy + iz*n[1]), n[0]);
                    }
                }
                for (int iz = 0; iz < n[2]; ++iz)
                {
                    I_double[iz] = RombergInt(I_single + n[1]*iz, n[1]);
                    // std::cout << iz << ": " << I_double[iz] << std::endl;
                }
                
                fHelmholtz_ -= RombergInt(I_double, n[2]);
                // std::cout << "Q_ = " << Q << std::endl;
                // std::cout << "Q  = " << RombergInt(I_double, n[2]) << std::endl;
                delete [] I_single;
                delete [] I_double;
            }
            else if (D == 2)
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
                    I_single[ix] = RombergIntegration(workArray.cDField() + n[1]*ix, n[1]);
                }
                fHelmholtz_ -= RombergInt(I_single, n[0]);
                delete [] I_single;
            }
            else 
            {
                fHelmholtz_ -= RombergIntegration(workArray.cDField(), nx);
            }

            assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (workArray.cDField(), 0.5, nx);
            inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (workArray.cDField(), cFieldsRGrid_[1].cDField(), nx);
            inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (workArray.cDField(), wFieldsRGrid_[1].cDField(), nx);
            if (D == 3)
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
                        I_single[iy + iz*n[1]] = RombergIntegration(workArray.cDField() + n[0]*(iy + iz*n[1]), n[0]);
                    }
                }
                for (int iz = 0; iz < n[2]; ++iz)
                {
                    I_double[iz] = RombergInt(I_single + n[1]*iz, n[1]);
                    // std::cout << iz << ": " << I_double[iz] << std::endl;
                }
                
                fHelmholtz_ -= RombergInt(I_double, n[2]);
                // std::cout << "Q_ = " << Q << std::endl;
                // std::cout << "Q  = " << RombergInt(I_double, n[2]) << std::endl;
                delete [] I_single;
                delete [] I_double;
            }
            else if (D == 2)
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
                    I_single[ix] = RombergIntegration(workArray.cDField() + n[1]*ix, n[1]);
                }
                fHelmholtz_ -= RombergInt(I_single, n[0]);
                delete [] I_single;
            }
            else 
            {
                fHelmholtz_ -= RombergIntegration(workArray.cDField(), nx);
            }
            fHelmholtz_ -= std::log(dpddiblock().Q());

            if (compressibility() == true)
            {
                RDField<D> gr;
                gr.allocate(mesh().dimensions());
    
                assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (gr.cDField(), dpddiblock().bu0().cDField(), basis().nStar());
                scaleReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (gr.cDField(), cFields_[0].cDField(), basis().nStar());
                scaleReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (gr.cDField(), cFields_[1].cDField(), basis().nStar());
    
                cudaReal *tmp_cpu;
                UAB_ = 0.0;
                tmp_cpu = new cudaReal[basis().nStar()];
                cudaMemcpy(tmp_cpu, gr.cDField(), sizeof(cudaReal)*basis().nStar(), cudaMemcpyDeviceToHost);
                for (int i = 0; i < basis().nStar(); ++i)
                    UAB_ += tmp_cpu[i];
                    // std::cout << tmp_cpu[i] << std::endl;
                delete [] tmp_cpu;
                UAB_ *= dpddiblock().chiN();
    
                UCMP_ = U_ - UAB_;
                          
                gr.deallocate();
            }
        }

        template <int D>
        void System<D>::outputThermo(std::ostream& out)
        {
            out << std::endl;
            out << "fc       = " << Dbl(fHelmholtz_, 21, 16) << std::endl;
            out << "uc       = " << Dbl(U_, 21, 16) << std::endl;
            
            if (compressibility() == true)
            {
                out << "uc,AB    = " << Dbl(UAB_, 21, 16) << std::endl;
                out << "uc,CMP   = " << Dbl(UCMP_, 21, 16) << std::endl;
            }

            out << "sc       = " << Dbl(-SA_-SB_ , 21, 16) << std::endl;
            out << "sc,A     = " << Dbl(-SA_, 21, 16) << std::endl;
            out << "sc,B     = " << Dbl(-SB_, 21, 16) << std::endl;
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
            }
            cudaMemcpy(kernelWorkSpace_,
                       d_kernelWorkSpace_,
                       NUMBER_OF_BLOCKS * sizeof(cudaReal),
                       cudaMemcpyDeviceToHost);
            cudaReal final = 0.0;
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
        cudaReal System<D>::reductionH(RDField<D>& a, int size) 
        {
            reduction <<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal) >>>
                    (d_kernelWorkSpace_, a.cDField(), size);
            cudaMemcpy(kernelWorkSpace_, d_kernelWorkSpace_, NUMBER_OF_BLOCKS * sizeof(cudaReal), cudaMemcpyDeviceToHost);
            cudaReal final = 0.0;

            for(int i = 0; i < NUMBER_OF_BLOCKS; ++i)
            {
                final += kernelWorkSpace_[i];
            }
            return final;
        }
    
            
        template <int D>
        cudaReal System<D>::RombergIntegration(cudaReal *f, int size)
        {
            cudaReal *I0_dev, *I1_dev,*I2_dev,*I3_dev,*I4_dev,
                     I0, I1, I2, I3, I4, 
                      dm, I;

            cudaMalloc((void**)&I0_dev, 1 * sizeof(cudaReal));
            cudaMalloc((void**)&I1_dev, 1 * sizeof(cudaReal));
            cudaMalloc((void**)&I2_dev, 1 * sizeof(cudaReal));
            cudaMalloc((void**)&I3_dev, 1 * sizeof(cudaReal));
            cudaMalloc((void**)&I4_dev, 1 * sizeof(cudaReal));

            dm = 1.0/double(size);

            deviceRI4 <<<1,32>>>(f, size, I0_dev, I1_dev, I2_dev, I3_dev, I4_dev);

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
        cudaReal System<D>::RombergInt(double *f, int size)
        { 
            double I0, I1, I2, I3, I4, dm;

            // With PBC, we have f[0] = (f[0] + f[size])/2
            I0 = f[0]; 
            I1 = f[0];
            I2 = f[0];
            I3 = f[0];
            I4 = f[0];
            dm = 1.0/double(size);

            for(int i = 1; i < size; ++i)
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

            I1 *= 2.0;
            I2 *= 4.0;
            I3 *= 8.0;
            I4 *= 16.0;

            return dm*(1048576.0*I0 - 348160.0*I1+ 22848.0*I2 - 340.0*I3 + I4)/722925.0;
        }

    }  
}
}

#endif
