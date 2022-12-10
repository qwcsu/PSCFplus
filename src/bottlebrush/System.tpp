#ifndef BBCPG_SYSTEM_TPP
#define BBCPG_SYSTEM_TPP

#include "System.h"

#include <pspg/GpuResources.h>

#include <pscf/homogeneous/Clump.h>
#include <pscf/crystal/shiftToMinimum.h>

#include <util/format/Str.h>
#include <util/format/Int.h>
#include <util/format/Dbl.h>

#include <string>
#include <getopt.h>

int THREADS_PER_BLOCK;
int NUMBER_OF_BLOCKS;

namespace Pscf{
namespace Pspg{
    namespace BBCpg{

        using namespace Util;

        template <int D>
        System<D>::System()
        {
            setClassName("System");
        }

        template <int D>
        System<D>::~System()
        {
            
        }

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

            // Read program arguments
            int c;
            opterr = 0;
            while ((c = getopt(argc, argv, "er:p:c:i:o:f1:2:")) != -1) {
                switch (c) {
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
            if (eflag) {
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
        void System<D>::readParam(std::istream& in)
        {
            std::cout << "Reading parameters file" << std::endl;
            readBegin(in, className().c_str());
            readParameters(in);
        }

        template<int D>
        void System<D>::readParameters(std::istream &in)
        {
            readParamComposite(in, mixture());
            hasMixture_ = true;
        }
    }
}
}

#endif