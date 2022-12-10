#ifndef DPD_PROPAGATOR_H
#define DPD_PROPAGATOR_H

#include <pscf/solvers/PropagatorTmpl.h> // base class template
#include <pspg/field/RDField.h>          // member template
#include <util/containers/DArray.h>      // member template

#include <pspg/solvers/Propagator.h>

namespace Pscf { template <int D> class Mesh; }

namespace Pscf
{
namespace Pspg
{
    namespace DPD
    {
        template <int D> class DPDBlock;
        template <int D> class Joint;
        using namespace Util;

        template <int D>
        class DPDpropagator : public PropagatorTmpl <DPDpropagator <D>>
        {

        public:

            typedef RDField<D> Field;
            typedef RDField<D> WField;
            typedef RDField<D> CField;
            typedef RDField<D> QField;

            DPDpropagator();

            ~DPDpropagator();

            void setBlock(DPDBlock<D>& block);

            void setJoint(Joint<D>& joint);

            void allocate(int N, const Mesh<D>& mesh);

            void computeHead();

            void solve();

            double computeQ();

            const cudaReal* q(int i) const;

            cudaReal* head() const;

            const cudaReal* tail() const;

            DPDBlock<D>& block();

            bool isAllocated() const;

            using PropagatorTmpl< DPDpropagator<D> >::nSource;
            using PropagatorTmpl< DPDpropagator<D> >::source;
            using PropagatorTmpl< DPDpropagator<D> >::partner;
            using PropagatorTmpl< DPDpropagator<D> >::setIsSolved;
            using PropagatorTmpl< DPDpropagator<D> >::isSolved;
            using PropagatorTmpl< DPDpropagator<D> >::hasPartner;


        private:

            cudaReal RI4_gpu(cudaReal *f, int size);

            double RI4_cpu(double *f, int size);

            DPDBlock<D> * blockPtr_;

            Joint<D> * jointPtr_;

            Mesh<D> const * meshPtr_;

            int N_;

            bool isAllocated_;

            cudaReal* qFields_d;
        };

        template <int D>
        inline
        void DPDpropagator<D>::setBlock(DPDBlock<D>& block)
        {
            blockPtr_ = &block;
        }

        template <int D>
        inline
        void DPDpropagator<D>::setJoint(Joint<D>& joint)
        {
            jointPtr_ = &joint;
        }

        template <int D>
        inline
        cudaReal* DPDpropagator<D>::head() const
        {
            return qFields_d;
        }

        template <int D>
        inline
        const cudaReal* DPDpropagator<D>::tail() const
        {
            return qFields_d + ((N_ - 1) * meshPtr_->size());
        }

        template <int D>
        inline 
        const cudaReal* DPDpropagator<D>::q(int i) const
        {
            return qFields_d + (i * meshPtr_->size());
        }

        template <int D>
        inline
        DPDBlock<D>& DPDpropagator<D>::block()
        {
            assert(blockPtr_);
            return *blockPtr_;
        }

        template <int D>
        inline
        bool DPDpropagator<D>::isAllocated() const
        {
            return isAllocated_;
        }

#ifndef DPD_PROPAGATOR_TPP
extern template class DPDpropagator<1>;
extern template class DPDpropagator<2>;
extern template class DPDpropagator<3>;
#endif

    }
}
}

__global__
void assignUniformReal(cudaReal* result, cudaReal uniform, int size);

__global__
void assignReal(cudaReal* result, const cudaReal* rhs, int size);

__global__
void inPlacePointwiseMul(cudaReal* a, const cudaReal* b, int size);

// template<unsigned int blockSize>
// __global__
// void deviceInnerProduct(cudaReal* c, const cudaReal* a,
//                                    const cudaReal* b, int size) {
//     //int nThreads = blockDim.x * gridDim.x;
//     int startID = blockIdx.x * blockDim.x + threadIdx.x;
//     int nThreads = blockDim.x * gridDim.x;
//     //do all pointwise multiplication
//     volatile extern __shared__ cudaReal cache[];
//     cudaReal temp = 0;
//     for(int i = startID; i < size; i += nThreads)
//     {
// //        printf("a[%5d] = %12.8f  b[%5d] = %12.8f  \n", i, a[i], i, b[i]);
//         temp += a[i] * b[i];
//     }

//     cache[threadIdx.x] = temp;
// //    printf("cache[%5d] =  %12.8f \n", threadIdx.x, cache[threadIdx.x]);

//     __syncthreads();

//     for (int i = blockDim.x/2; i != 0; i/=2)
//     {
//         if(threadIdx.x < i)
//             cache[threadIdx.x] += cache[threadIdx.x + i];
//         __syncthreads();
//     }

//     if (threadIdx.x == 0)
//     {
//         c[blockIdx.x] = cache[0];
//     }
// }

#endif // !DPD_PROPAGATOR_H