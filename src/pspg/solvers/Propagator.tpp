#ifndef PSPG_PROPAGATOR_TPP
#define PSPG_PROPAGATOR_TPP

/*
* PSCF - Polymer Self-Consistent Field Theory
*
* Copyright 2016 - 2019, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*/

#include "Propagator.h"
#include "Block.h"
// #include <thrust/reduce.h>
#include "device_launch_parameters.h"
#include <cuda.h>
//#include <device_functions.h>
#include <thrust/count.h>
#include <pspg/GpuResources.h>
#include <pscf/mesh/Mesh.h>
//#include <Windows.h>


__global__
void assignUniformReal(cudaReal* result, cudaReal uniform, int size) {
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("uniform = %f\n", uniform);
    for(int i = startID; i < size; i += nThreads) {
        result[i] = uniform;
    }

}

__global__
void assignReal(cudaReal* result, const cudaReal* rhs, int size) {
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = startID; i < size; i += nThreads) {
        result[i] = rhs[i];
    }
}

__global__
void inPlacePointwiseMul(cudaReal* a, const cudaReal* b, int size) {
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = startID; i < size; i += nThreads) {
        a[i] *= b[i];
    }
}

#if 0
template<unsigned int blockSize>
__global__ void deviceInnerProduct(cudaReal* c, const cudaReal* a,
	const cudaReal* b, int size) {
        //int nThreads = blockDim.x * gridDim.x;
	int startID = blockIdx.x * blockDim.x + threadIdx.x;

	//do all pointwise multiplication
	volatile extern __shared__ cudaReal cache[];
	cudaReal temp = 0;
	//no need for loop here will be wrong.
	//for (int i = startID; i < size; i += nThreads) {
	temp += a[startID] * b[startID];
	//}
	cache[threadIdx.x] = temp;

	__syncthreads();

	if(blockSize >= 512) {
	  if (threadIdx.x < 256){
	    cache[threadIdx.x] += cache[threadIdx.x + 256];
	  }
	  __syncthreads();
	}
	if(blockSize >= 256) {
	  if (threadIdx.x < 128){
	    cache[threadIdx.x] += cache[threadIdx.x + 128];
	  }
	  __syncthreads();
	}
	if(blockSize >= 128) {
	  if (threadIdx.x < 64){
	    cache[threadIdx.x] += cache[threadIdx.x + 64];
	  }
	  __syncthreads();
	}
	//reduce operation
	//256/2 -- needs to be power of two
	//for (int j = blockDim.x / 2; j > 32; j /= 2) {
	//	if (threadIdx.x < j) {
	//		cache[threadIdx.x] += cache[threadIdx.x + j];
	//	}
	//	__syncthreads();
	//}


	if (threadIdx.x < 32) {
	  if(blockSize >= 64) cache[threadIdx.x] += cache[threadIdx.x + 32];
	  if(blockSize >= 32) cache[threadIdx.x] += cache[threadIdx.x + 16];
	  if(blockSize >= 16) cache[threadIdx.x] += cache[threadIdx.x + 8];
	  if(blockSize >= 8) cache[threadIdx.x] += cache[threadIdx.x + 4];
	  if(blockSize >= 4) cache[threadIdx.x] += cache[threadIdx.x + 2];
	  if(blockSize >= 2) cache[threadIdx.x] += cache[threadIdx.x + 1];

	}

	if (threadIdx.x == 0) {
		c[blockIdx.x] = cache[0];
	}
}
#endif

namespace Pscf {
    namespace Pspg {

        using namespace Util;

        /*
        * Constructor.
        */
        template <int D>
        Propagator<D>::Propagator()
                : blockPtr_(0),
                  meshPtr_(0),
                  ns_(0),
                  isAllocated_(false)
        {
        }

        /*
        * Destructor.
        */
        template <int D>
        Propagator<D>::~Propagator()
        {
            delete[] temp_;
            cudaFree(d_temp_);
            cudaFree(qFields_d);
        }

        template <int D>
        void Propagator<D>::allocate(int ns, const Mesh<D>& mesh)
        {
            ns_ = ns;
            meshPtr_ = &mesh;
            int nx = 1;
            for (int i = 0; i < 3; ++i)
            {
                nx *= meshPtr_->dimensions()[i];
            }
            // std::cout<< nx <<std::endl;
            cudaMalloc((void**)&qFields_d, sizeof(cudaReal)* nx * ns);
            cudaMalloc((void**)&d_temp_, NUMBER_OF_BLOCKS * sizeof(cudaReal));
            temp_ = new cudaReal[NUMBER_OF_BLOCKS];
            isAllocated_ = true;
        }

        /*
        * Compute initial head QField from final tail QFields of sources.
        */
        template <int D>
        void Propagator<D>::computeHead()
        {

            // Reference to head of this propagator
            //QField& qh = qFields_[0];

            // Initialize qh field to 1.0 at all grid points
            int nx = 1;
            for (int i = 0; i < 3; ++i)
            {
                nx *= meshPtr_->dimensions()[i];
            }
            //qh[ix] = 1.0;
            //qFields_d points to the first float in gpu memory
            assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qFields_d, 1.0, nx);

            // Pointwise multiply tail QFields of all sources
            // this could be slow with many sources. Should launch 1 kernel for the whole
            // function of computeHead
            const cudaReal* qt;
            for (int is = 0; is < nSource(); ++is) {
                if (!source(is).isSolved()) {
                    UTIL_THROW("Source not solved in computeHead");
                }
                //need to modify tail to give the total_size - mesh_size pointer
                qt = source(is).tail();
                //qh[ix] *= qt[ix];
                inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (qFields_d, qt, nx);
            }
            
            // exit(1);
        }

        /*
        * Solve the modified diffusion equation for this block.
        */
        template <int D>
        void Propagator<D>::solve()
        {   
            UTIL_CHECK(isAllocated())
            int nx = 1;
            for (int i = 0; i < 3; ++i)
            {
                nx *= meshPtr_->dimensions()[i];
            }
            
            computeHead();
            // cudaReal qtc[32];
            // cudaMemcpy(qtc, qFields_d+  (ns_-1)* meshPtr_->size(), sizeof(cudaReal)*32, cudaMemcpyDeviceToHost);
            // for (int i = 0; i < 32; ++i)
            //     std::cout << qtc[i] << std::endl;
            // exit(1);
            // Setup solver and solve
            block().setupFCT();

            int currentIdx;
            for (int iStep = 0; iStep < ns_ - 1; ++iStep)
            {
                currentIdx = iStep * nx;
                block().step(qFields_d + currentIdx,
                             qFields_d + currentIdx + nx);
            }
            // exit(1);
            setIsSolved(true);
        }

        /*
        * Solve the modified diffusion equation with specified initial field.
        */
        template <int D>
        void Propagator<D>::solve(const cudaReal * head)
        {
            int nx = 1;
            for (int i = 0; i < 3; ++i)
            {
                nx *= meshPtr_->dimensions()[i];
            }

            // Initialize initial (head) field
            cudaReal* qh = qFields_d;
            // qh[i] = head[i];
            assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(qh, head, nx);

            // Setup solver and solve
            int currentIdx;
            for (int iStep = 0; iStep < ns_ - 1; ++iStep) {
                currentIdx = iStep * nx;
                block().step(qFields_d + currentIdx, qFields_d + currentIdx + nx);
            }
            
            setIsSolved(true);
        }

        /*
        * Integrate to calculate monomer concentration for this block
        */
        template <int D>
        double Propagator<D>::computeQ()
        {
            // Preconditions
            if (!isSolved()) {
                UTIL_THROW("Propagator is not solved.");
            }
            if (!hasPartner()) {
                UTIL_THROW("Propagator has no partner set.");
            }
            if (!partner().isSolved()) {
                UTIL_THROW("Partner propagator is not solved");
            }
            const cudaReal * qh = head();
            const cudaReal * qt = partner().tail();
            int nx = 1;
            for (int i = 0; i < 3; ++i)
            {
                nx *= meshPtr_->dimensions()[i];
            }
            
            // std::cout << "mesh size is " << nx << std::endl;
            // std::cout << "mesh is " << meshPtr_->dimensions()[0] << std::endl;
            // exit(1);

            // Take inner product of head and partner tail fields
            // cannot reduce assuming one propagator, qh == 1
            // polymers are divided into blocks midway through
            double Q;          
            cudaReal *tmp;
            cudaMalloc((void**)&tmp, nx * sizeof(cudaReal));
            assignReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tmp, qh, nx);
            inPlacePointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(tmp, qt, nx);

            // 3-D
            if(D == 3)
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
// #if REPS == 0
//                         I_single[iy + ix*n[1]] = RI0_gpu(tmp + n[2]*(iy + ix*n[1]), n[2]);
// #endif
// #if REPS == 1
//                         I_single[iy + ix*n[1]] = RI1_gpu(tmp + n[2]*(iy + ix*n[1]), n[2]);
// #endif
// #if REPS == 2
//                         I_single[iy + ix*n[1]] = RI2_gpu(tmp + n[2]*(iy + ix*n[1]), n[2]);
// #endif
// #if REPS == 3
//                         I_single[iy + ix*n[1]] = RI3_gpu(tmp + n[2]*(iy + ix*n[1]), n[2]);
// #endif
// #if REPS == 4
                        I_single[iy + ix*n[1]] = RI4_gpu(tmp + n[2]*(iy + ix*n[1]), n[2]);
// #endif
                    }
// #if REPS == 0
//                     I_double[ix] = RI0_cpu(I_single + n[1]*ix, n[1]);
// #endif
// #if REPS == 1
//                     I_double[ix] = RI1_cpu(I_single + n[1]*ix, n[1]);
// #endif
// #if REPS == 2
//                     I_double[ix] = RI2_cpu(I_single + n[1]*ix, n[1]);
// #endif
// #if REPS == 3
//                     I_double[ix] = RI3_cpu(I_single + n[1]*ix, n[1]);
// #endif
// #if REPS == 4
                    I_double[ix] = RI4_cpu(I_single + n[1]*ix, n[1]);
// #endif
                }
// #if REPS == 0
//                 Q = RI0_cpu(I_double, n[0]);
// #endif               
// #if REPS == 1
//                 Q = RI1_cpu(I_double, n[0]);
// #endif
// #if REPS == 2
//                 Q = RI2_cpu(I_double, n[0]);
// #endif
// #if REPS == 3
//                 Q = RI3_cpu(I_double, n[0]);
// #endif
// #if REPS == 4
                Q = RI4_cpu(I_double, n[0]);
// #endif
                
                delete [] I_single;
                delete [] I_double;

                // std::cout << "Q = " << Q << std::endl;
                // exit(1);
            }
            else if (D ==2)
            {
                int n[2];   // n[3] stores the number of mesh points in each direction
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
// #if REPS == 0
//                 Q = RI0_gpu(tmp, nx);
// #endif
// #if REPS == 1
//                 Q = RI1_gpu(tmp, nx);
// #endif
// #if REPS == 2
//                 Q = RI2_gpu(tmp, nx);
// #endif
// #if REPS == 3
//                 Q = RI3_gpu(tmp, nx);
// #endif
// #if REPS == 4
                Q = RI4_gpu(tmp, nx);
// #endif
            }
            cudaFree(tmp);
            
            return Q;
        }

        template <int D>
        cudaReal Propagator<D>::innerProduct(const cudaReal* a, const cudaReal* b, int size) {

            switch(THREADS_PER_BLOCK){
                case 512:
                    deviceInnerProduct<512><<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>(d_temp_, a, b, size);
                    break;
                case 256:
                    deviceInnerProduct<256><<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>(d_temp_, a, b, size);
                    break;
                case 128:
                    deviceInnerProduct<128><<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>(d_temp_, a, b, size);
                    break;
                case 64:
                    deviceInnerProduct<64><<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>(d_temp_, a, b, size);
                    break;
                case 32:
                    deviceInnerProduct<32><<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>(d_temp_, a, b, size);
                    break;
                case 16:
                    deviceInnerProduct<16><<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>(d_temp_, a, b, size);
                    break;
                case 8:
                    deviceInnerProduct<8><<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>(d_temp_, a, b, size);
                    break;
                case 4:
                    deviceInnerProduct<4><<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>(d_temp_, a, b, size);
                    break;
                case 2:
                    deviceInnerProduct<2><<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>(d_temp_, a, b, size);
                    break;
                case 1:
                    deviceInnerProduct<1><<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(cudaReal)>>>(d_temp_, a, b, size);
                    break;
            }

            cudaMemcpy(temp_, d_temp_, NUMBER_OF_BLOCKS * sizeof(cudaReal), cudaMemcpyDeviceToHost);
            cudaReal final = 0;
            // cudaReal c = 0;
            // //use kahan summation
            // for(int i = 0; i < NUMBER_OF_BLOCKS; ++i)
            // {
            //     cudaReal y = temp_[i] - c;
            //     cudaReal t = final + y;
            //     c = (t - final) - y;
            //     final = t;
            // }
           for(int i = 0; i < NUMBER_OF_BLOCKS; ++i)
           {
               final += temp_[i];
               // std::cout << NUMBER_OF_BLOCKS << "\n";
           }

            return final;
        }
// #if REPS == 4 
        template <int D>
        cudaReal Propagator<D>::RI4_gpu(cudaReal *f, int size)
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
        cudaReal Propagator<D>::RI4_cpu(double *f, int size)
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
//         cudaReal Propagator<D>::RI3_gpu(cudaReal *f, int size)
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
//         cudaReal Propagator<D>::RI3_cpu(double *f, int size)
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
//         cudaReal Propagator<D>::RI2_gpu(cudaReal *f, int size)
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
//         cudaReal Propagator<D>::RI2_cpu(double *f, int size)
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

//             return dm * (64.0 * I0 - 20.0*  I1 + I2) / 45.0;
//         }
// #endif
// #if REPS == 1
//         template <int D>
//         cudaReal Propagator<D>::RI1_gpu(cudaReal *f, int size)
//         {
//             cudaReal *I0_dev, *I1_dev,
//                      I0, I1, 
//                      dm, I;

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
//         cudaReal Propagator<D>::RI1_cpu(double *f, int size)
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
//         cudaReal Propagator<D>::RI0_gpu(cudaReal *f, int size)
//         {
//             cudaReal *I0_dev,
//                      I0, 
//                      dm, I;

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
//         cudaReal Propagator<D>::RI0_cpu(double *f, int size)
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
