#ifndef DPD_PROPAGATOR_TPP
#define DPD_PROPAGATOR_TPP

#include "DPDpropagator.h"
#include "dpd/solvers/DPDBlock.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <thrust/count.h>
#include <pspg/GpuResources.h>
#include <pscf/mesh/Mesh.h>

namespace Pscf
{
namespace Pspg
{
    namespace DPD{

        using namespace Util;

        template <int D>
        DPDpropagator<D>::DPDpropagator()
         : blockPtr_(0),
           meshPtr_(0),
           N_(0),
           isAllocated_(false)
        {
        }

        template <int D>
        DPDpropagator<D>::~DPDpropagator()
        {
            cudaFree(qFields_d);
        }

        template <int D>
        void DPDpropagator<D>::allocate(int N, const Mesh<D>& mesh)
        {
            N_ = N;
            meshPtr_ = &mesh;

            cudaMalloc((void**)&qFields_d, sizeof(cudaReal) * mesh.size() * N);
            
            isAllocated_ = true;
        }

        template <int D>
        void DPDpropagator<D>::computeHead()
        {
            int nx = meshPtr_->size();
            cudaReal uniform = 1.0;
            
            cudaReal qr_host[nx];
            setUniformReal<<<32, 512>>>
            (qFields_d, uniform, nx);
            cudaMemcpy(qr_host, qFields_d, sizeof(cudaReal)*nx, cudaMemcpyDeviceToHost);            

            const cudaReal* qt;

            for (int is = 0; is < nSource(); ++is)
            {
                if (!source(is).isSolved()) 
                {
                    UTIL_THROW("Source not solved in computeHead");
                }

                qt = source(is).tail();

                inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (qFields_d, qt, nx);
            }

            inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
            (qFields_d, block().expW().cDField(), nx);
        }

        template <int D>
        void DPDpropagator<D>::solve()
        {
            UTIL_CHECK(isAllocated())

            computeHead();

            block().setupFFT();

            int currentIdx;
            for (int iStep = 0; iStep < N_-1; ++iStep)
            {
                currentIdx = iStep * meshPtr_->size();
                block().step(qFields_d + currentIdx,
                             qFields_d + currentIdx + meshPtr_->size());
            }

            setIsSolved(true);
        }

        template <int D>
        double DPDpropagator<D>::computeQ()
        {

            if (!isSolved()) 
            {
                UTIL_THROW("Propagator is not solved.");
            }

            const cudaReal * qh = head();
            const cudaReal * qt = partner().tail();
            int nx = meshPtr_->size();

            double Q;          
            cudaReal *tmp, *W;
            cudaMalloc((void**)&tmp, nx * sizeof(cudaReal));
            cudaMalloc((void**)&W, nx * sizeof(cudaReal));

            assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tmp, qh, nx);

            inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tmp, qt, nx);

            inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tmp, W, nx);
            
            if (D == 3)
            {
                int n[3];   // n[3] stores the number of mesh points in each direction
                n[0] = 1;  n[1] = 1;  n[2] = 1;
                for (int i = 0; i < 3; ++i)
                {
                    n[i] = meshPtr_->dimensions()[i];
                }

                double *I_single,
                       *I_double;
                I_single = new double[n[0]*n[1]];
                I_double = new double[n[0]];

                for (int ix = 0; ix < n[0]; ++ix)
                {
                    for (int iy = 0; iy < n[1]; ++iy)
                    {
                        I_single[iy + ix*n[1]] = RI4_gpu(tmp + n[2]*(iy + ix*n[1]), n[2]);
                    }

                    I_double[ix] = RI4_cpu(I_single + n[1]*ix, n[1]);
                }

                Q = RI4_cpu(I_double, n[0]);
                
                delete [] I_single;
                delete [] I_double;
            }
            else if (D ==2)
            {
                int n[2];   
                n[0] = 1;  n[1] = 1;
                for (int i = 0; i < 2; ++i)
                {
                    n[i] = meshPtr_->dimensions()[i];
                }

                double *I_single;
                I_single = new double[n[1]];

                for (int ix = 0; ix < n[0]; ++ix)
                {
                    I_single[ix] = RI4_gpu(tmp + n[1]*ix, n[1]);
                }

                Q = RI4_cpu(I_single, n[0]);

                delete [] I_single;
            }
            else
            {
                Q = RI4_gpu(tmp, nx);
            }
            cudaFree(tmp);
            cudaFree(W);
            // exit(1);
            return Q;
        }

        template <int D>
        cudaReal DPDpropagator<D>::RI4_gpu(cudaReal *f, int size)
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
        cudaReal DPDpropagator<D>::RI4_cpu(double *f, int size)
        { 
            double I0, I1, I2, I3, I4, dm;

            // With PBC, we have f[0] = (f[0] + f[size])/2
            I0 = f[0]; 
            I1 = f[0];
            I2 = f[0];
            I3 = f[0];
            I4 = f[0];
            dm = 1.0/double(size);

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
    }
}
}

#endif