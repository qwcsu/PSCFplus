#ifndef BBC_SYSTEM_TPP
#define BBC_SYSTEM_TPP

#include <bbc/System.h>
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
    namespace BBC
    {
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
            bool eFlag = false,
                 pFlag = false,
                 cFlag = false,
                 iFlag = false,
                 oFlag = false,
                 wFlag = false,
                 tFlag = false;
            char *pArg = nullptr,
                 *cArg = nullptr,
                 *iArg = nullptr,
                 *oArg = nullptr;
            int c;

            opterr = 0;
            while ((c = getopt(argc, argv, "er:p:c:i:o:f1:2:")) != -1)
            {
                switch (c)
                {
                    case 'e':
                        eFlag = true;
                        break;
                    case 'p':
                        pFlag = true;
                        pArg  = optarg;
                        break; 
                    case 'c':
                        cFlag = true;
                        cArg  = optarg;
                        break;
                    case 'i':
                        iFlag = true;
                        iArg  = optarg;
                        break;
                    case 'o':
                        oFlag = true;
                        oArg  = optarg;
                        break;
                    case '1':
                        NUMBER_OF_BLOCKS = atoi(optarg);
                        wFlag = true;
                        break;
                    case '2':
                        THREADS_PER_BLOCK = atoi(optarg);
                        tFlag = true;
                        break;
                    case '?':
                        Log::file() << "Unknown option - " << optopt << std::endl;
                        UTIL_THROW("Invalid command line option");
                    default:
                        UTIL_THROW("Default exit (setOptions)");
                }
            }

            if (eFlag)
            {
                Util::ParamComponent::setEcho (true);
            }

            if (pFlag)
            {
                fileMaster().setParamFileName(std::string(pArg));
            }

            if (cFlag) 
            {
                fileMaster().setCommandFileName(std::string(cArg));
            }

            if (iFlag) 
            {
                fileMaster().setInputPrefix(std::string(iArg));
            }

            if (oFlag) 
            {
                fileMaster().setOutputPrefix(std::string(oArg));
            }

            if (!wFlag) 
            {
                std::cout<<"Number of blocks not set " <<std::endl;
                exit(1);
            }

            if (!tFlag) 
            {
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
            // std::cout << className().c_str() << std::endl;
            readBegin(in, className().c_str());
            readParameters(in);
            // readEnd(in);
            
            
        }

        template<int D>
        void System<D>::readParameters(std::istream &in)
        {
            readParamComposite(in, mixture());
        }
    }

}

#endif