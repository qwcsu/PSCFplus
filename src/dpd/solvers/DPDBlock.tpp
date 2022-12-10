#ifndef DPD_BLOCK_TPP
#define DPD_BLOCK_TPP

#include "DPDBlock.h"

static
__global__
void ExpW(cudaReal* expW, const cudaReal* w, 
               int size, double cDs)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads)
    {
        expW[i] = exp(-w[i]*cDs);
        // printf("w = %f, cDs = %f\n", w[i], cDs);
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
void pointwiseEqual(const cudaReal* a, cudaReal* b, int size)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads)
    {
        b[i] = a[i];
    }
}

static
__global__
void pointwiseTriple(cudaReal* result,
                     const cudaReal* w,
                     const cudaReal* q1,
                     const cudaReal* q2,
                     int size)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads)
    {
        result[i] += w[i]*q1[i]*q2[i];
    }
}

namespace Pscf {
namespace Pspg {
    namespace DPD{

        using namespace Util;

        template <int D>
        DPDBlock<D>::DPDBlock()
         : meshPtr_(0),
           kMeshDimensions_(0)
        {
            propagator(0).setBlock(*this);
            propagator(1).setBlock(*this);
        }

        template <int D>
        DPDBlock<D>::~DPDBlock()
        {
            delete[] expKsq_host;

            cField().deallocate();
            expW_.deallocate();
            expWp_.deallocate();
        }

        template <int D>
        void DPDBlock<D>::setDiscretization(const Mesh<D>& mesh)
        {
            UTIL_CHECK(mesh.size() > 1)

            meshPtr_ = &mesh;

            for (int i = 0; i < D; ++i)
            {
                if (i < D - 1) {
                    kMeshDimensions_[i] = mesh.dimensions()[i];
                } else {
                    kMeshDimensions_[i] = mesh.dimensions()[i]/2 + 1;
                }
            }

            kSize_ = 1;

            for(int i = 0; i < D; ++i) {
                kSize_ *= kMeshDimensions_[i];
            }

            expKsq_host = new cudaReal[kSize_];
            expKsq_.allocate(kMeshDimensions_);

            qr_.allocate(mesh.dimensions());
            qk_.allocate(mesh.dimensions());

            propagator(0).allocate(length(), mesh);
            propagator(1).allocate(length(), mesh);

            cField().allocate(mesh.dimensions());
            expW_.allocate(mesh.dimensions());
            expWp_.allocate(mesh.dimensions());
        }

        template <int D>
        void DPDBlock<D>::setupUnitCell (const UnitCell<D>& unitCell,
                                         const WaveList<D>& waveList)
        {
            nParams_ = unitCell.nParameter();
            MeshIterator<D> iter;
            iter.setDimensions(kMeshDimensions_);
            IntVec<D> G;
            double Gsq;
            double factor = -1.0*kuhn()*kuhn();

            int kSize = 1;
            for(int i = 0; i < D; ++i) 
            {
                kSize *= kMeshDimensions_[i];
            }

            int i;
            for (iter.begin(); !iter.atEnd(); ++iter) 
            {
                i = iter.rank();
                Gsq = unitCell.ksq(waveList.minImage(iter.rank()));
                expKsq_host[i] = exp(factor*Gsq);
            }
            cudaMemcpy(expKsq_.cDField(),  
                       expKsq_host, 
                       kSize * sizeof(cudaReal), 
                       cudaMemcpyHostToDevice);
            // std::cout << "Error: " << cudaGetErrorString(cudaGetLastError()) << '\n';
        }

        template <int D>
        void DPDBlock<D>::setupSolver(DPDBlock<D>::WField const & w, int N)
        {
            int nx = mesh().size();
            UTIL_CHECK(nx > 0)
            
            double ds = 1.0/double(N);
            // std::cout << "nx = " << nx << "\n";
            if (!isJoint())
            {
                ExpW<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (expW_.cDField(), w.cDField(), nx, ds);

                ExpW<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (expW_.cDField(), w.cDField(), nx, -ds);
            }
            else{
                assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (expW_.cDField(), 1.0, nx);
                assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (expWp_.cDField(), 1.0, nx);
            }
        }

        template <int D>
        void DPDBlock<D>::computeConcentration(double prefactor)
        {
            if (!isJoint())
            {
                int nx = mesh().size();
                UTIL_CHECK(nx > 0)
                
                int N = length();

                UTIL_CHECK(N > 0)

                assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (cField().cDField(), 0.0, nx);

                Pscf::Pspg::DPD::DPDpropagator<D> const & p0 = propagator(0);
                Pscf::Pspg::DPD::DPDpropagator<D> const & p1 = propagator(1);


                for(int s = 0; s < N; ++s)
                {
                    pointwiseTriple<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                    (cField().cDField(), expWp_.cDField(),
                    p0.q(s), p1.q(N- 1 - s), nx);
                }
                
                scaleRealconst <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>> 
                (cField().cDField(), (double)(prefactor), nx);
            }
        }

        template <int D>
        void DPDBlock<D>::setupFFT()
        {
            if (!fft_.isSetup())
            {
                fft_.setup(qr_, qk_);
            }
        }

        template <int D>
        void DPDBlock<D>::step(const cudaReal *q, cudaReal *qNew)
        {
            int nx = mesh().size();
            UTIL_CHECK(nx > 0)
            UTIL_CHECK(qr_.capacity() == nx)
            UTIL_CHECK(expW_.capacity() == nx)

            int nk = qk_.capacity();
            UTIL_CHECK(expKsq_.capacity() == nk)

            pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
            (q, qr_.cDField(), nx);

            fft_.forwardTransform(qr_, qk_);
            
            scaleComplex<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
            (qk_.cDField(), expKsq_.cDField(), kSize_);

            fft_.inverseTransform(qk_, qr_);

            scaleReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
            (qr_.cDField(), expW_.cDField(), nx);

            pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
            (qr_.cDField(), qNew, nx);
        }
   
    }
}
}

#endif // !DPD_BLOCK_TPP