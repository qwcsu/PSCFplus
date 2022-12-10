#ifndef PSPG_BLOCK_TPP
#define PSPG_BLOCK_TPP
// #define double float

/*
* PSCF - Polymer Self-Consistent Field Theory
*
* Copyright 2016 - 2019, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*/

#include "Block.h"
#include <pspg/GpuResources.h>
#include <pscf/mesh/Mesh.h>
#include <pscf/mesh/MeshIterator.h>
#include <pscf/crystal/shiftToMinimum.h>
#include <util/containers/FMatrix.h>      // member template
#include <util/containers/DArray.h>      // member template
#include <util/containers/FArray.h>      // member template
#include <ctime>


//not a bad idea to rewrite these as functors
static __global__ 
void pointwiseMul(const cudaReal* a, const cudaReal* b, cudaReal* result, int size) {
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads) {
        result[i] = a[i] * b[i];
    }
}

// double precision
static __global__
void pointwiseFloatMul(const cudaReal* a, const double* b, cudaReal* result, int size)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads) {
        result[i] = a[i] * b[i];
        // printf("result[%d], =  %d\n", i , result[i]);
    }
}

static __global__ void mulDelKsq(cudaReal* result, cudaReal* q1,
                                 cudaReal* q2, cudaReal* delKsq,
                                 int paramN, int size) {
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads) {

        result[i] =  q1[i] * q2[i] * delKsq[paramN * size + i];

    }
}

static __global__
void equalize ( const cudaReal* a, double* result, int size){
    //try to add elements of array here itself

    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads) {
        result [i] = a [i];
    }

}

static __global__ void pointwiseMulUnroll2(const cudaReal* a, const cudaReal* b, cudaReal* result, int size) {
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
    cudaReal localResult[2];
    for (int i = startID; i < size; i += nThreads * 2) {
        localResult[0] = a[i] * b[i];
        localResult[1] = a[i + 1] * b[i + 1];
        result[i] = localResult[0];
        result[i + 1] = localResult[1];
        //result[i] = a[i] * b[i];
        //result[i + 1] = a[i + 1] * b[i + 1];

    }
}

static __global__ void pointwiseMulCombi(cudaReal* a,const cudaReal* b, cudaReal* c,const cudaReal* d,const cudaReal* e, int size) {
    //c = a * b
    //a = d * e
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    cudaReal tempA;
    for (int i = startID; i < size; i += nThreads) {
        tempA = a[i];
        c[i] = tempA * b[i];
        a[i] = d[i] * e[i];

    }
}


static __global__
void pointwiseMulSameStart(const cudaReal* a,
                           const cudaReal* expW,
                           const cudaReal* expW2,
                           cudaReal* q1, cudaReal* q2,
                           int size)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    cudaReal input;
    for (int i = startID; i < size; i += nThreads) {
        input = a[i];
        q1[i] = expW[i] * input;
        q2[i] = expW2[i] * input;
    }
}

static __global__
void pointwiseMulTwinned(const cudaReal* qr1,
                         const cudaReal* qr2,
                         const cudaReal* expW,
                         cudaReal* q1, cudaReal* q2,
                         int size)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    cudaReal scale;
    for (int i = startID; i < size; i += nThreads)
    {
        scale = expW[i];
        q1[i] = qr1[i] * scale;
        q2[i] = qr2[i] * scale;
    }
}

static __global__
void scaleComplexTwinned(cudaComplex* qk1,
                         cudaComplex* qk2,
                         const cudaReal* expksq1,
                         const cudaReal* expksq2, int size)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads) {
        qk1[i].x *= expksq1[i];
        qk1[i].y *= expksq1[i];
        qk2[i].x *= expksq2[i];
        qk2[i].y *= expksq2[i];
    }
}

static __global__ void scaleComplex(cudaComplex* a, cudaReal* scale, int size) {
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = startID; i < size; i += nThreads) {
        a[i].x *= scale[i];
        a[i].y *= scale[i];
    }
}

static __global__ void scaleRealPointwise(cudaReal* a, cudaReal* scale, int size) {
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = startID; i < size; i += nThreads) {
        a[i] *= scale[i];
    }
}

static __global__ void assignExp(cudaReal* expW, const cudaReal* w, int size, double cDs) {
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = startID; i < size; i += nThreads) {
        expW[i] = exp(-w[i]*cDs);
    }
}

#if REPS == 0
static __global__
void richardsonExp_0(cudaReal* qNew,
                   const cudaReal* q1,
                   int size)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads) {
        qNew[i] = q1[i];
    }
}
#endif
#if REPS == 1
static __global__
void richardsonExp_1(cudaReal* qNew,
                   const cudaReal* q1,
                   const cudaReal* q2,
                   int size)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads) {
        qNew[i] = (4.0 * q2[i] - q1[i]) / 3.0;
    }
}
#endif
#if REPS == 2
static __global__
void richardsonExp_2(cudaReal* qNew,
                   const cudaReal* q1,
                   const cudaReal* q2,
                   const cudaReal* q3,
                   int size)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads) {
        qNew[i] = (  64*q3[i]
                   - 20*q2[i] + q1[i]) / 45.0;
    }
}
#endif
#if REPS == 3
static __global__
void richardsonExp_3(cudaReal* qNew,
                   const cudaReal* q1,
                   const cudaReal* q2,
                   const cudaReal* q3,
                   const cudaReal* q4,
                   int size)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads) {
        qNew[i] = (  4096*q4[i]
                   - 1344*q3[i]
                   + 84*q2[i] -q1[i]) / 2835.0;
    }
}
#endif
#if REPS == 4
static __global__
void richardsonExp_4(cudaReal* qNew,
                   const cudaReal* q1,
                   const cudaReal* q2,
                   const cudaReal* q3,
                   const cudaReal* q4,
                   const cudaReal* q5,
                   int size)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads) {
        qNew[i] = (  1048576 * q5[i]
                   - 348160  * q4[i] 
                   + 22848   * q3[i]
                   - 340     * q2[i] + q1[i]) / 722925.0;
    }
}
#endif

// Not using this function
// static __global__ void richardsonExpTwinned(cudaReal* qNew, const cudaReal* q1,
//                                             const cudaReal* qr, const cudaReal* expW2, int size) {
//     int nThreads = blockDim.x * gridDim.x;
//     int startID = blockIdx.x * blockDim.x + threadIdx.x;
//     cudaReal q2;
//     for (int i = startID; i < size; i += nThreads) {
//         q2 = qr[i] * expW2[i];
//         qNew[i] = (4.0 * q2 - q1[i]) / 3.0;
//     }
// }

namespace Pscf {
    namespace Pspg {

        using namespace Util;

        static __global__ void multiplyScaleQQ(cudaReal* result,
                                               const cudaReal* p1,
                                               const cudaReal* p2,
                                               int size, double scale) {

            int nThreads = blockDim.x * gridDim.x;
            int startID = blockIdx.x * blockDim.x + threadIdx.x;

            for(int i = startID; i < size; i += nThreads) {
                result[i] += scale * p1[i] * p2[i];
            }

        }

        static __global__ void scaleReal(cudaReal* result, int size, double scale) {
            int nThreads = blockDim.x * gridDim.x;
            int startID = blockIdx.x * blockDim.x + threadIdx.x;

            for (int i = startID; i < size; i += nThreads) {
                result[i] *= scale;
            }
        }
        /*
        * Constructor.
        */
        template <int D>
        Block<D>::Block()
                : meshPtr_(0),
                  ds_(0.0),
                  ns_(0)
        {
            propagator(0).setBlock(*this);
            propagator(1).setBlock(*this);
        }

        /*
        * Destructor.
        */
        template <int D>
        Block<D>::~Block()
        {
            delete[] temp_;
            cudaFree(d_temp_);

#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1 || REPS ==0
            delete[] expKsq_host;
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1
            delete[] expKsq2_host;
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 
            delete[] expKsq3_host;
#endif
#if REPS == 4 || REPS == 3
            delete[] expKsq4_host;
#endif
#if REPS == 4
            delete[] expKsq5_host;
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1 || REPS ==0
            expKsq_.deallocate();
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1
            expKsq2_.deallocate();
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 
            expKsq3_.deallocate();
#endif
#if REPS == 4 || REPS == 3
            expKsq4_.deallocate();
#endif
#if REPS == 4
            expKsq5_.deallocate();
#endif

#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1 || REPS ==0
            expW_.deallocate();
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1
            expW2_.deallocate();
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 
            expW3_.deallocate();
#endif
#if REPS == 4 || REPS == 3
            expW4_.deallocate();
#endif
#if REPS == 4
            expW5_.deallocate();
#endif
        }

        template <int D>
        void Block<D>::setDiscretization(double ds, const Mesh<D>& mesh)
        {
            UTIL_CHECK(mesh.size() > 1)
            UTIL_CHECK(ds > 0.0)

            // Set association to mesh
            meshPtr_ = &mesh;


            // Set contour length discretization
            ns_ = floor(length()/ds + 0.5) + 1;
            if (ns_%2 == 0) {
                ns_ += 1;
            }

            ds_ = length()/double(ns_ - 1);

            // Allocate work arrays
            size_ph_ = meshPtr_->size();
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1 || REPS ==0
            expKsq_.allocate(size_ph_);

#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1
            expKsq2_.allocate(size_ph_);
#endif 
#if REPS == 4 || REPS == 3 || REPS == 2 
            expKsq3_.allocate(size_ph_);
#endif
#if REPS == 4 || REPS == 3
            expKsq4_.allocate(size_ph_);
#endif
#if REPS == 4 
            expKsq5_.allocate(size_ph_);
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1 || REPS ==0
            expW_.allocate(meshPtr_->dimensions());
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1
            expW2_.allocate(meshPtr_->dimensions());
#endif 
#if REPS == 4 || REPS == 3 || REPS == 2 
            expW3_.allocate(meshPtr_->dimensions());
#endif
#if REPS == 4 || REPS == 3
            expW4_.allocate(meshPtr_->dimensions());
#endif
#if REPS == 4 
            expW5_.allocate(meshPtr_->dimensions());
#endif
            
            qr_.allocate(meshPtr_->dimensions());
            
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1 || REPS ==0
            q1_.allocate(meshPtr_->dimensions());
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1
            q2_.allocate(meshPtr_->dimensions());
#endif 
#if REPS == 4 || REPS == 3 || REPS == 2            
            q3_.allocate(meshPtr_->dimensions());
#endif
#if REPS == 4 || REPS == 3
            q4_.allocate(meshPtr_->dimensions());
#endif
#if REPS == 4 
            q5_.allocate(meshPtr_->dimensions());
#endif
            
            propagator(0).allocate(ns_, mesh);
            propagator(1).allocate(ns_, mesh);

            cField().allocate(meshPtr_->dimensions());

            cudaMalloc((void**)&d_temp_,
                       NUMBER_OF_BLOCKS * sizeof(cudaReal));
            temp_ = new cudaReal[NUMBER_OF_BLOCKS];

#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1 || REPS ==0
            expKsq_host = new cudaReal[size_ph_];
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1
            expKsq2_host = new cudaReal[size_ph_];
#endif 
#if REPS == 4 || REPS == 3 || REPS == 2   
            expKsq3_host = new cudaReal[size_ph_];
#endif
#if REPS == 4 || REPS == 3
            expKsq4_host = new cudaReal[size_ph_];
#endif
#if REPS == 4 
            expKsq5_host = new cudaReal[size_ph_];
#endif
        }

        /*
         * Setup data that depend on the unit cell parameters.
         */
        template <int D>
        void
        Block<D>::setupUnitCell(const UnitCell<D>& unitCell, 
                                const WaveList<D>& wavelist)
        {
            
            int v[3];
            
            v[0] = mesh().dimensions()[0];
            v[1] = mesh().dimensions()[1];
            v[2] = mesh().dimensions()[2];
            
            meshPtr_ph_->dimensions()[0] = v[0];
            meshPtr_ph_->dimensions()[1] = v[1];
            meshPtr_ph_->dimensions()[2] = v[2];
            
            IntVec<D> mesh_v(v);

            for (int k0 = 0; k0 < v[0]; ++k0)
            {
                for (int k1 = 0; k1 < v[1]; ++k1)
                {
                    for (int k2 = 0; k2 < v[2]; ++k2)
                    {
                        int tmp_v[3];
                        tmp_v[0] = k0;  tmp_v[1] = k1;  tmp_v[2] = k2;
                        IntVec<D> kv(tmp_v), vm;
                        // vm = shiftToMinimum(kv, mesh_v, unitCell);
                        // std::cout << "(" << v[0] << ", " << v[1] << ", " << v[2] << ") -> ";
                        // std::cout << "(" << vm[0] << ", " << vm[1] << ", " << vm[2] << ") : ";
                        // double Gsq = unitCell.ksq(vm);
                        double Gsq = unitCell.ksq(kv);
                        int idx = k2 + k1 * v[2] + k0 * v[1] *v[2];
                        double factor = -1.0*kuhn()*kuhn()*ds_/6.0;
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1 || REPS ==0
                        expKsq_host[idx] = exp(Gsq*factor);
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1
                        expKsq2_host[idx] = exp(0.5*Gsq*factor);
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 
                        expKsq3_host[idx] = exp(0.25*Gsq*factor);
#endif
#if REPS == 4 || REPS == 3 
                        expKsq4_host[idx] = exp(0.125*Gsq*factor);
#endif
#if REPS == 4 
                        expKsq5_host[idx] = exp(0.0625*Gsq*factor);
#endif
                    }
                }
            }
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1 || REPS ==0
            cudaMemcpy(expKsq_.cDField(), expKsq_host, 
            size_ph_ * sizeof(cudaReal), cudaMemcpyHostToDevice);
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1
            cudaMemcpy(expKsq2_.cDField(), expKsq2_host, 
            size_ph_ * sizeof(cudaReal), cudaMemcpyHostToDevice);
#endif
#if REPS == 4 || REPS == 3 || REPS == 2 
            cudaMemcpy(expKsq3_.cDField(), expKsq3_host, 
            size_ph_ * sizeof(cudaReal), cudaMemcpyHostToDevice);
#endif
#if REPS == 4 || REPS == 3 
            cudaMemcpy(expKsq4_.cDField(), expKsq4_host, 
            size_ph_ * sizeof(cudaReal), cudaMemcpyHostToDevice);
#endif
#if REPS == 4
            cudaMemcpy(expKsq5_.cDField(), expKsq5_host, 
            size_ph_ * sizeof(cudaReal), cudaMemcpyHostToDevice);
#endif
            nParams_ = unitCell.nParameter();
        }

        /*
        * Setup the contour length step algorithm.
        */
        template <int D>
        void
        Block<D>::setupSolver(Block<D>::WField const& w)
        {
            UTIL_CHECK(size_ph_ > 0)
            int Discretization_of_each_block = ns()-1;
            // Discretization of each block should satisfies the following 
#if REPS == 4
            UTIL_CHECK(Discretization_of_each_block%16 == 0)
#endif  
#if REPS == 3
            UTIL_CHECK(Discretization_of_each_block%8 == 0)
#endif  
#if REPS == 2
            UTIL_CHECK(Discretization_of_each_block%4 == 0)
#endif 
#if REPS == 1
            UTIL_CHECK(Discretization_of_each_block%2 == 0)
#endif  

#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1 || REPS ==0
            
            assignExp<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
            (expW_.cDField(), w.cDField(), size_ph_, (double)0.5* ds_);
            
#endif  
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1
            assignExp <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (expW2_.cDField(), w.cDField(), size_ph_, (double)0.25 * ds_);
#endif  
#if REPS == 4 || REPS == 3 || REPS == 2 
            assignExp <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (expW3_.cDField(), w.cDField(), size_ph_, (double)0.125 * ds_);
#endif
#if REPS == 4 || REPS == 3
            assignExp <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (expW4_.cDField(), w.cDField(), size_ph_, (double)0.0625 * ds_);
#endif
#if REPS == 4
            assignExp <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (expW5_.cDField(), w.cDField(), size_ph_, (double)0.03125 * ds_);
#endif
        }

        /*
        * Integrate to calculate monomer concentration for this block
        */
        template <int D>
        void Block<D>::computeConcentration(double prefactor)
        {
            // Preconditions
            UTIL_CHECK(size_ph_ > 0)
            UTIL_CHECK(ns_ > 0)
            UTIL_CHECK(ds_ > 0)
            UTIL_CHECK(propagator(0).isAllocated())
            UTIL_CHECK(propagator(1).isAllocated())
            
            // Initialize cField to zero at all points
            //cField()[i] = 0.0;
            assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(cField().cDField(), 0.0, size_ph_);

            Pscf::Pspg::Propagator<D> const & p0 = propagator(0);
            Pscf::Pspg::Propagator<D> const & p1 = propagator(1);
            // std::cout << "ds_ = " << ds_ << "\n";
            // std::cout << "ns_ = " << ns_ << "\n";
            // exit(1);
#if REPS == 0
            // I0 : ds
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(0), p1.q(ns_ - 1), size_ph_, 0.5);
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(ns_-1), p1.q(0), size_ph_, 0.5);
            for (int j = 1; j < ns_ - 1; j += 1) {
                //odd indices
                multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (cField().cDField(), p0.q(j), p1.q(ns_ - 1 - j), size_ph_, 1.0);
            }

            scaleReal <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>> 
            (cField().cDField(), size_ph_, (double)(prefactor *ds_));

            // double q_c[4];
            // cudaMemcpy(q_c, cField().cDField(), sizeof(double)*4, cudaMemcpyDeviceToHost);
            // for(int i = 0; i < 4; ++i)
            //     std::cout << q_c[i] << std::endl;

#endif
#if REPS == 1
            // I0 : ds
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(0), p1.q(ns_ - 1), size_ph_, 0.5*4.0);
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(ns_-1), p1.q(0), size_ph_, 0.5*4.0);
            for (int j = 1; j < ns_ - 1; j += 1) {
                //odd indices
                multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (cField().cDField(), p0.q(j), p1.q(ns_ - 1 - j), size_ph_, 1.0*4.0);
            }

            // I1 : 2ds
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(0), p1.q(ns_ - 1), size_ph_, -2.0*0.5);
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(ns_-1), p1.q(0), size_ph_, -2.0*0.5);
            for (int j = 2; j < ns_ - 2; j += 2) {
                //even indices
                multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (cField().cDField(), p0.q(j), p1.q(ns_ - 1 - j), size_ph_, -2.0*1.0);
            }

            scaleReal <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>> 
            (cField().cDField(), size_ph_, (double)(prefactor *ds_ / 3.0));
#endif
#if REPS == 2
            // I0 : ds
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(0), p1.q(ns_ - 1), size_ph_, 0.5*64.0);
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(ns_-1), p1.q(0), size_ph_, 0.5*64.0);
            for (int j = 1; j < ns_ - 1; j += 1) {
                //odd indices
                multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (cField().cDField(), p0.q(j), p1.q(ns_ - 1 - j), size_ph_, 1.0*64.0);
            }

            // I1 : 2ds
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(0), p1.q(ns_ - 1), size_ph_, -2.0*0.5*20.0);
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(ns_-1), p1.q(0), size_ph_, -2.0*0.5*20.0);
            for (int j = 2; j < ns_ - 2; j += 2) {
                //even indices
                multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (cField().cDField(), p0.q(j), p1.q(ns_ - 1 - j), size_ph_, -2.0*1.0*20.0);
            }

            // I2 : 4ds
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(0), p1.q(ns_ - 1), size_ph_, 4.0*0.5);
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(ns_-1), p1.q(0), size_ph_, 4.0*0.5);
            for (int j = 4; j < ns_ - 4; j += 4) {
                //even indices
                multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (cField().cDField(), p0.q(j), p1.q(ns_ - 1 - j), size_ph_, 4.0*1.0);
            }
            
            scaleReal <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), size_ph_, (double)(prefactor *ds_ / 45.0));
#endif
#if REPS == 3
            // I0 : ds
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(0), p1.q(ns_ - 1), size_ph_, 0.5*4096.0);
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(ns_-1), p1.q(0), size_ph_, 0.5*4096.0);
            for (int j = 1; j < ns_ - 1; j += 1) {
                //odd indices
                multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (cField().cDField(), p0.q(j), p1.q(ns_ - 1 - j), size_ph_, 1.0*4096.0);
            }

            // I1 : 2ds
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(0), p1.q(ns_ - 1), size_ph_, -2.0*0.5*1344.0);
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(ns_-1), p1.q(0), size_ph_, -2.0*0.5*1344.0);
            for (int j = 2; j < ns_ - 2; j += 2) {
                //even indices
                multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (cField().cDField(), p0.q(j), p1.q(ns_ - 1 - j), size_ph_, -2.0*1.0*1344.0);
            }

            // I2 : 4ds
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(0), p1.q(ns_ - 1), size_ph_, 4.0*0.5*84.0);
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(ns_-1), p1.q(0), size_ph_, 4.0*0.5*84.0);
            for (int j = 4; j < ns_ - 4; j += 4) {
                //even indices
                multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (cField().cDField(), p0.q(j), p1.q(ns_ - 1 - j), size_ph_, 4.0*1.0*84.0);
            }

            // I3 : 8ds
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(0), p1.q(ns_ - 1), size_ph_, -8.0*0.5);
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(ns_-1), p1.q(0), size_ph_, -8.0*0.5);
            for (int j = 8; j < ns_ - 8; j += 8) {
                //even indices
                multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (cField().cDField(), p0.q(j), p1.q(ns_ - 1 - j), size_ph_, -8.0*1.0);
            }
            
            scaleReal <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), size_ph_, (double)(prefactor *ds_ / 2835.0));
#endif
#if REPS == 4
            // I0 : ds
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(0), p1.q(ns_ - 1), size_ph_, 0.5*1048576);
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(ns_-1), p1.q(0), size_ph_, 0.5*1048576);
            for (int j = 1; j < ns_ - 1; j += 1) {
                //odd indices
                multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (cField().cDField(), p0.q(j), p1.q(ns_ - 1 - j), size_ph_, 1.0*1048576.0);
            }

            // I1 : 2ds
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(0), p1.q(ns_ - 1), size_ph_, -2.0*0.5*348160);
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(ns_-1), p1.q(0), size_ph_, -2.0*0.5*348160);
            for (int j = 2; j < ns_ - 2; j += 2) {
                //even indices
                multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (cField().cDField(), p0.q(j), p1.q(ns_ - 1 - j), size_ph_, -2.0*1.0*348160);
            }

            // I2 : 4ds
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(0), p1.q(ns_ - 1), size_ph_, 4.0*0.5*22848);
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(ns_-1), p1.q(0), size_ph_, 4.0*0.5*22848);
            for (int j = 4; j < ns_ - 4; j += 4) {
                //even indices
                multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (cField().cDField(), p0.q(j), p1.q(ns_ - 1 - j), size_ph_, 4.0*1.0*22848);
            }

            // I3 : 8ds
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(0), p1.q(ns_ - 1), size_ph_, -8.0*0.5*340);
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(ns_-1), p1.q(0), size_ph_, -8.0*0.5*340);
            for (int j = 8; j < ns_ - 8; j += 8) {
                //even indices
                multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (cField().cDField(), p0.q(j), p1.q(ns_ - 1 - j), size_ph_, -8.0*1.0*340);
            }

            // I4 : 16ds
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(0), p1.q(ns_ - 1), size_ph_, 16.0*0.5);
            multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), p0.q(ns_-1), p1.q(0), size_ph_, 16.0*0.5);
            if(ns_ >= 33)
            {
                for (int j = 16; j < ns_ - 16; j += 16) 
                {
                //even indices
                multiplyScaleQQ <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (cField().cDField(), p0.q(j), p1.q(ns_ - 1 - j), size_ph_, 16.0*1.0);
                }
            }
            
            scaleReal <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (cField().cDField(), size_ph_, (double)(prefactor *ds_ / 722925.0));
#endif
        }

        template <int D>
        void Block<D>::setupFCT() {
            if(!fct_.isSetup()) {
                int mesh[3];
                mesh[0] = meshPtr_->dimension(0);
                mesh[1] = meshPtr_->dimension(1);
                mesh[2] = meshPtr_->dimension(2);

                fct_.setup(mesh);
            }
        }

        /*
        * Propagate solution by one step.
        */
        //step have to be done in gpu
        template <int D>
        void Block<D>::step(const cudaReal* q, cudaReal* qNew)
        {
            UTIL_CHECK(size_ph_ > 0)
            UTIL_CHECK(qr_.capacity() == size_ph_)
            UTIL_CHECK(expW_.capacity() == size_ph_)

            // double q_c[4];
            // cudaMemcpy(q_c, q, sizeof(double)*4, cudaMemcpyDeviceToHost);
            // for(int i = 0; i < 4; ++i)
            //     std::cout << q_c[i] << std::endl;
            
            /// Apply pseudo-spectral algorithm
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1 || REPS ==0
            // 1-step
            pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (q, expW_.cDField(), qr_.cDField(), size_ph_);
            fct_.forwardTransform(qr_.cDField());
            scaleRealPointwise <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (qr_.cDField(), expKsq_.cDField(), size_ph_);
            fct_.inverseTransform(qr_.cDField());
            pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (qr_.cDField(), expW_.cDField(), q1_.cDField(), size_ph_);

#endif  
#if REPS == 4 || REPS == 3 || REPS == 2 || REPS ==1
            // 2-step
            pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (q, expW2_.cDField(), qr_.cDField(), size_ph_);
            fct_.forwardTransform(qr_.cDField());
            scaleRealPointwise <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (qr_.cDField(), expKsq2_.cDField(), size_ph_);
            fct_.inverseTransform(qr_.cDField());
            scaleRealPointwise<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (qr_.cDField(), expW_.cDField(), size_ph_);

            fct_.forwardTransform(qr_.cDField());
            scaleRealPointwise<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (qr_.cDField(), expKsq2_.cDField(), size_ph_);
            fct_.inverseTransform(qr_.cDField());
            pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (qr_.cDField(), expW2_.cDField(), q2_.cDField(), size_ph_);
#endif 
#if REPS == 4 || REPS == 3 || REPS == 2 
            // 4-step
            pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (q, expW3_.cDField(), qr_.cDField(), size_ph_);
            fct_.forwardTransform(qr_.cDField());
            scaleRealPointwise <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (qr_.cDField(), expKsq3_.cDField(), size_ph_);
            fct_.inverseTransform(qr_.cDField());
            pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (qr_.cDField(), expW2_.cDField(), qr_.cDField(), size_ph_);

            for(int i = 0; i < 2; ++i)
            {
                fct_.forwardTransform(qr_.cDField());
                scaleRealPointwise <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (qr_.cDField(), expKsq3_.cDField(), size_ph_);
                fct_.inverseTransform(qr_.cDField());
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (qr_.cDField(), expW2_.cDField(), qr_.cDField(), size_ph_);
            }

            fct_.forwardTransform(qr_.cDField());
            scaleRealPointwise <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (qr_.cDField(), expKsq3_.cDField(), size_ph_);
            fct_.inverseTransform(qr_.cDField());
            pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (qr_.cDField(), expW3_.cDField(), q3_.cDField(), size_ph_);
#endif
#if REPS == 4 || REPS == 3
            // 8-step
            pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (q, expW4_.cDField(), qr_.cDField(), size_ph_);
            fct_.forwardTransform(qr_.cDField());
            scaleRealPointwise <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (qr_.cDField(), expKsq4_.cDField(), size_ph_);
            fct_.inverseTransform(qr_.cDField());
            pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (qr_.cDField(), expW3_.cDField(), qr_.cDField(), size_ph_);

            for(int i = 0; i < 6; ++i)
            {
                fct_.forwardTransform(qr_.cDField());
                scaleRealPointwise <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (qr_.cDField(), expKsq4_.cDField(), size_ph_);
                fct_.inverseTransform(qr_.cDField());
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (qr_.cDField(), expW3_.cDField(), qr_.cDField(), size_ph_);
            }

            fct_.forwardTransform(qr_.cDField());
            scaleRealPointwise <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (qr_.cDField(), expKsq4_.cDField(), size_ph_);
            fct_.inverseTransform(qr_.cDField());
            pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (qr_.cDField(), expW4_.cDField(), q4_.cDField(), size_ph_);
#endif
#if REPS == 4
            // 16-step
            pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (q, expW5_.cDField(), qr_.cDField(), size_ph_);
            fct_.forwardTransform(qr_.cDField());
            scaleRealPointwise <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (qr_.cDField(), expKsq5_.cDField(), size_ph_);
            fct_.inverseTransform(qr_.cDField());
            pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (qr_.cDField(), expW4_.cDField(), qr_.cDField(), size_ph_);

            for(int i = 0; i < 14; ++i)
            {
                fct_.forwardTransform(qr_.cDField());
                scaleRealPointwise <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (qr_.cDField(), expKsq5_.cDField(), size_ph_);
                fct_.inverseTransform(qr_.cDField());
                pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (qr_.cDField(), expW4_.cDField(), qr_.cDField(), size_ph_);
            }

            fct_.forwardTransform(qr_.cDField());
            scaleRealPointwise <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (qr_.cDField(), expKsq5_.cDField(), size_ph_);
            fct_.inverseTransform(qr_.cDField());
            pointwiseMul<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (qr_.cDField(), expW5_.cDField(), q5_.cDField(), size_ph_);
#endif
#if REPS == 0
            richardsonExp_0<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (qNew, 
             q1_.cDField(), 
             size_ph_);

#endif
#if REPS == 1
            richardsonExp_1<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (qNew, 
             q1_.cDField(), 
             q2_.cDField(), 
             size_ph_);
#endif
#if REPS == 2
            richardsonExp_2<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (qNew, 
             q1_.cDField(), 
             q2_.cDField(), 
             q3_.cDField(), 
             size_ph_);
#endif
#if REPS == 3
            richardsonExp_3<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (qNew, 
             q1_.cDField(), 
             q2_.cDField(), 
             q3_.cDField(), 
             q4_.cDField(), 
             size_ph_);
#endif

#if REPS == 4
            richardsonExp_4<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (qNew, 
             q1_.cDField(), 
             q2_.cDField(), 
             q3_.cDField(), 
             q4_.cDField(), 
             q5_.cDField(),
             size_ph_);
#endif
        }

        /*
        * Integrate to Stress exerted by the chain for this block
        * For many reasons, the code is written in away that tries
        * to function with basis functions but does nothing
        * to actually ensure its correctness.
        * To optimize it, I am rewriting it to only allow
        * the symmetry I since none of the code is correct in the first place
        */
        template <int D>
        void Block<D>::computeStress(WaveList<D>& wavelist, double prefactor)
        {
            // Preconditions

            int nx = mesh().size();
             
            UTIL_CHECK(nx > 0);
            UTIL_CHECK(ns_ > 0);
            UTIL_CHECK(ds_ > 0);
            UTIL_CHECK(propagator(0).isAllocated());
            UTIL_CHECK(propagator(1).isAllocated());

            double normal, increment, inc0, inc1, inc2, inc3, inc4,rmg0;

            //dont use a compile time array.....
            FArray<double, 6> dQ;

            int i;
            for (i = 0; i < 6; ++i) 
            {
                dQ [i] = 0.0;
                stress_[i] = 0.0;
            }

            Pscf::Pspg::Propagator<D> const & p0 = propagator(0);
            Pscf::Pspg::Propagator<D> const & p1 = propagator(1);

            cudaReal *p0_tmp, *p1_tmp;
            cudaMalloc((void**)&p0_tmp, ns_*nx*sizeof(cudaReal));
            cudaMalloc((void**)&p1_tmp, ns_*nx*sizeof(cudaReal));
            equalize<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (p0.head(), p0_tmp, ns_*nx);
            equalize<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
            (p1.head(), p1_tmp, ns_*nx);

            for (int s = 0; s < ns_; s++)
            {
                fct_.forwardTransform(p0_tmp + s*nx);
                fct_.forwardTransform(p1_tmp + s*nx);
            }
            // fct_.inverseTransform(p0_tmp);
            // cudaReal *q_c;
            // q_c = new cudaReal [64];
            // cudaMemcpy(q_c, p0_tmp, 64*sizeof(cudaReal), cudaMemcpyDeviceToHost);
            // for(int i = 0; i < 64; ++i)
            //     std::cout << q_c[i] << "\n";
            // exit(1);
            
#if REPS == 0
            normal = 6.0;
            for (int n = 0; n < nParams_ ; ++n) 
            {
                rmg0 = 0;
                inc0 = 0;
                
                mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (q1_.cDField(), 
                 p0_tmp, 
                 p1_tmp + (nx * (ns_ -1)),
                 wavelist.dkSq(), n, nx);
                rmg0 += reductionH(q1_, nx);  
                
                mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (q1_.cDField(), 
                 p0_tmp + (nx* (ns_ -1)), 
                 p1_tmp,
                 wavelist.dkSq(), n , nx);
                 rmg0 += reductionH(q1_, nx);  
                
                rmg0 *= 0.5;
                rmg0 *= ds_;

                for (int j = 1; j < ns_ - 1; ++j) 
                {
                    mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (q1_.cDField(), 
                     p0_tmp + (j * nx), 
                     p1_tmp + (nx* (ns_ -1 -j)),
                     wavelist.dkSq(), n, nx);

                    inc0 += reductionH(q1_, nx);                
                }
                inc0 *=  ds_;  

                increment = rmg0 + inc0;     
                increment = (increment * kuhn() * kuhn())/normal;   
                
                dQ [n] = dQ[n]-increment;
            }
#endif
#if REPS == 1
            normal = 6.0;
            for (int n = 0; n < nParams_ ; ++n) 
            {
                rmg0 = 0;
                inc0 = 0;  inc1 = 0;  
                
                mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (q1_.cDField(), 
                 p0_tmp, 
                 p1_tmp + (nx * (ns_ -1)),
                 wavelist.dkSq(), n , nx);
                rmg0 += reductionH(q1_, nx); 
                
                // cudaReal *dkSq_c;
                // dkSq_c = new cudaReal [64];
                // cudaMemcpy(dkSq_c, wavelist.dkSq(), 64*sizeof(cudaReal), cudaMemcpyDeviceToHost);
                // for(int i = 0; i < 6; ++i)
                //     std::cout << dkSq_c[i] << "\n";
                // std::cout << rmg0 << "\n";
                // exit(1);

                mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (q1_.cDField(), 
                 p0_tmp + (nx * (ns_ -1)), 
                 p1_tmp,
                 wavelist.dkSq(), n , nx);
                rmg0 += reductionH(q1_, nx);  

                rmg0 *= ds_;
                
                for (int j = 1; j < ns_ - 1; ++j) 
                {
                    mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (q1_.cDField(), 
                     p0_tmp + (j * nx), 
                     p1_tmp + (nx * (ns_ -1 -j)),
                     wavelist.dkSq(), n , nx);

                    inc0 += reductionH(q1_, nx);                 
                }
                inc0 *=  4.0 * ds_; 

                for (int j = 2; j < ns_ - 2; j += 2) 
                {
                    mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (q1_.cDField(), 
                     p0_tmp + (j * nx), 
                     p1_tmp + (nx * (ns_ -1 -j)),
                     wavelist.dkSq(), n , nx);

                    inc1 += reductionH(q1_, nx);               
                }
                inc1 *= -2.0 * ds_; 

                increment = (rmg0 + inc0 + inc1) / 3.0;     
                increment = (increment * kuhn() * kuhn())/normal;   
                dQ [n] = dQ[n]-increment;
            }
#endif
#if REPS == 2
            normal = 6.0;
            for (int n = 0; n < nParams_ ; ++n) 
            {
                rmg0 = 0;
                inc0 = 0;  inc1 = 0;  inc2 = 0;  inc3 = 0;
                
                mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (q1_.cDField(), 
                 p0_tmp, 
                 p1_tmp + (nx * (ns_ -1)),
                 wavelist.dkSq(), n , nx);
                rmg0 += reductionH(q1_, nx);  
                mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (q1_.cDField(), 
                 p0_tmp + (nx * (ns_ -1)), 
                 p1_tmp,
                 wavelist.dkSq(), n , nx);
                rmg0 += reductionH(q1_, nx);  
                
                rmg0 *= 12.0;
                rmg0 *= ds_;

                for (int j = 1; j < ns_ - 1; ++j) 
                {
                    mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (q1_.cDField(), 
                     p0_tmp + (j * nx), 
                     p1_tmp + (nx * (ns_ -1 -j)),
                     wavelist.dkSq(), n , nx);

                    inc0 += reductionH(q1_, nx);                
                }
                inc0 *=  64 * ds_; 

                for (int j = 2; j < ns_ - 2; j += 2) 
                {
                    mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (q1_.cDField(), 
                     p0_tmp + (j * nx), 
                     p1_tmp + (nx * (ns_ -1 -j)),
                     wavelist.dkSq(), n , nx);

                    inc1 += reductionH(q1_, nx);                  
                }
                inc1 *= -2 * 20.0 * ds_; 

                for (int j = 4; j < ns_ - 4; j += 4) 
                {
                    mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (q1_.cDField(), 
                     p0_tmp + (j * nx), 
                     p1_tmp + (nx * (ns_ -1 -j)),
                     wavelist.dkSq(), n , nx);

                    inc2 += reductionH(q1_, nx);               
                }
                inc2 *= 4.0 * ds_;   

                increment = (rmg0 + inc0 + inc1 + inc2 + inc3) / 2835.0;     
                increment = (increment * kuhn() * kuhn())/normal;   
                dQ [n] = dQ[n]-increment;
            }
#endif
#if REPS == 3
            normal = 6.0;
            for (int n = 0; n < nParams_ ; ++n) 
            {
                rmg0 = 0;
                inc0 = 0;  inc1 = 0;  inc2 = 0;  inc3 = 0;
                
                mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (q1_.cDField(), 
                 p0_tmp, 
                 p1_tmp + (nx * (ns_ -1)),
                 wavelist.dkSq(), n , nx);

                rmg0 += reductionH(q1_, nx);  

                mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (q1_.cDField(), 
                 p0_tmp + (nx * (ns_ -1)), 
                 p1_tmp,
                 wavelist.dkSq(), n , nx);
                 
                rmg0 += reductionH(q1_, nx);  

                rmg0 *= 868.0;
                rmg0 *= ds_;

                for (int j = 1; j < ns_ - 1; ++j) 
                {
                    mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (q1_.cDField(), 
                     p0_tmp + (j * nx), 
                     p1_tmp + (nx * (ns_ -1 -j)),
                     wavelist.dkSq(), n , nx);

                    inc0 += reductionH(q1_, nx);                
                }
                inc0 *=  4096.0 * ds_; 

                for (int j = 2; j < ns_ - 2; j += 2) 
                {
                    mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (q1_.cDField(), 
                     p0_tmp + (j * nx), 
                     p1_tmp + (nx * (ns_ -1 -j)),
                     wavelist.dkSq(), n , nx);

                    inc1 += reductionH(q1_, nx);                  
                }
                inc1 *= -2 * 1344.0 * ds_; 

                for (int j = 4; j < ns_ - 4; j += 4) 
                {
                    mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (q1_.cDField(), 
                     p0_tmp + (j * nx), 
                     p1_tmp + (nx * (ns_ -1 -j)),
                     wavelist.dkSq(), n , nx);

                    inc2 += reductionH(q1_, nx);                
                }
                inc2 *= 4.0 * 84.0 * ds_;  

                for (int j = 8; j < ns_ - 8; j += 8) 
                {
                    mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (q1_.cDField(), 
                     p0_tmp + (j * nx), 
                     p1_tmp + (nx * (ns_ -1 -j)),
                     wavelist.dkSq(), n , nx);

                    inc3 += reductionH(q1_, nx);                 
                }
                inc3 *=  -8.0 * ds_;   

                increment = (rmg0 + inc0 + inc1 + inc2 + inc3) / 2835.0;     
                increment = (increment * kuhn() * kuhn())/normal;   
                dQ [n] = dQ[n]-increment;
            }
#endif
#if REPS == 4
            normal = 6.0;
            for (int n = 0; n < nParams_ ; ++n) 
            {
                rmg0 = 0;
                inc0 = 0;  inc1 = 0;  inc2 = 0;  inc3 = 0;  inc4 = 0;
                
                mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (q1_.cDField(), 
                 p0_tmp, 
                 p1_tmp + (nx * (ns_ -1)),
                 wavelist.dkSq(), n , nx);

                rmg0 += reductionH(q1_, nx);  

                mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (q1_.cDField(), 
                 p0_tmp + (nx * (ns_ -1)), 
                 p1_tmp,
                 wavelist.dkSq(), n , nx);
                 
                rmg0 += reductionH(q1_, nx);  
                
                rmg0 *= 220472.0;
                rmg0 *= ds_;

                for (int j = 1; j < ns_ - 1; ++j) 
                {
                    mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (q1_.cDField(), 
                     p0_tmp + (j * nx), 
                     p1_tmp + (nx * (ns_ -1 -j)),
                     wavelist.dkSq(), n , nx);

                    inc0 += reductionH(q1_, nx);                  
                }
                inc0 *=  1048576.0 * ds_; 

                for (int j = 2; j < ns_ - 2; j += 2) 
                {
                    mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (q1_.cDField(), 
                     p0_tmp + (j * nx), 
                     p1_tmp + (nx * (ns_ -1 -j)),
                     wavelist.dkSq(), n , nx);

                    inc1 += reductionH(q1_, nx);                  
                }
                inc1 *= -2 * 348160 * ds_; 

                for (int j = 4; j < ns_ - 4; j += 4) 
                {
                    mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (q1_.cDField(), 
                     p0_tmp + (j * nx), 
                     p1_tmp + (nx * (ns_ -1 -j)),
                     wavelist.dkSq(), n , nx);

                    inc2 += reductionH(q1_, nx);              
                }
                inc2 *= 4.0 * 22848 * ds_;  

                for (int j = 8; j < ns_ - 8; j += 8) 
                {
                    mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (q1_.cDField(), 
                     p0_tmp + (j * nx), 
                     p1_tmp + (nx * (ns_ -1 -j)),
                     wavelist.dkSq(), n , nx);

                    inc3 += reductionH(q1_, nx);               
                }
                inc3 *=  -8.0 * 340 *ds_;   

                if(ns_ >= 33)
                {
                    for (int j = 16; j < ns_ - 16; j += 16) 
                    {
                        mulDelKsq<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                        (q1_.cDField(), 
                         p0_tmp + (j * nx), 
                         p1_tmp + (nx * (ns_ -1 -j)),
                         wavelist.dkSq(), n , nx);

                        inc4 += reductionH(q1_, nx);            
                    }
                    inc4 *=  16.0 *ds_;
                }

                increment = (rmg0 + inc0 + inc1 + inc2 + inc3 + inc4) / 722925.0;     
                increment = (increment * kuhn() * kuhn())/normal;   
                // std::cout << "REPS5 = " << increment <<std::endl;
                dQ [n] = dQ[n]-increment;
            }
#endif
            // Normalize
            for (i = 0; i < nParams_; ++i) 
            {
                stress_[i] = stress_[i] - (dQ[i] * prefactor);
                // std::cout << stress_[i] << "\n";
            }

            cudaFree(p0_tmp);
            cudaFree(p1_tmp);  
        }

        template<int D>
        cudaReal Block<D>::reductionH(RDField<D>& a, int size)
        {
            reduction <<< NUMBER_OF_BLOCKS , THREADS_PER_BLOCK, THREADS_PER_BLOCK*sizeof(cudaReal) >>>
                    (d_temp_, a.cDField(), size);
            cudaMemcpy(temp_, d_temp_, NUMBER_OF_BLOCKS  * sizeof(cudaReal), cudaMemcpyDeviceToHost);
            cudaReal final = 0.0;
            cudaReal c = 0;
             for (int i = 0; i < NUMBER_OF_BLOCKS/2 ; ++i) {
                cudaReal y = temp_[i] - c;
                cudaReal t = final + y;
                c = (t - final) - y;
                final = t;
             }
        //    for (int i = 0; i < NUMBER_OF_BLOCKS ; ++i)
        //    {
        //        final += temp_[i];
        //    }
            return final;
        }

    }
}
#endif
