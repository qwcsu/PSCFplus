#ifndef DPDDIBLOCK_TPP
#define DPDDIBLOCK_TPP

#include "DPDdiblock.h"
#include "pspg/solvers/Propagator.h"


__global__ 
void assignExp(cudaReal* expW, const cudaReal* w, int size, double cDs) 
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = startID; i < size; i += nThreads) 
    {
        expW[i] = exp(-w[i]*cDs);
    }
}

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
    namespace DPDpg
    {
        using namespace Util;


        static __global__ 
        void dpdStress_c(cudaReal* ca, cudaReal* cb,
                         cudaReal* dk, cudaReal* dbu0,
                         double chiN, double kpN, double sigma,
                         cudaReal* result,
                         int size)
        {

            int nThreads = blockDim.x * gridDim.x;
            int startID = blockIdx.x * blockDim.x + threadIdx.x;

            for(int i = startID; i < size; i += nThreads) 
            {
                if (i != 0)
                {
                    result[i] += (0.5*kpN*(ca[i] + cb[i])*(ca[i] + cb[i]) + chiN*ca[i]*cb[i])*sigma*dk[i]*dbu0[i];
                }
            }
        }

        static __global__ 
        void dpdStress_inc(cudaReal* ca, cudaReal* cb,
                         cudaReal* dk, cudaReal* dbu0,
                         double chiN, double sigma,
                         cudaReal* result,
                         int size)
        {

            int nThreads = blockDim.x * gridDim.x;
            int startID = blockIdx.x * blockDim.x + threadIdx.x;

            for(int i = startID; i < size; i += nThreads) 
            {
                result[i] += (chiN*ca[i]*cb[i])*sigma*dk[i]*dbu0[i];
            }
        }

        static __global__ 
        void dpdStress_q(const cudaReal* qt, const cudaReal* qts,
                         cudaReal* dk, cudaReal* dphi,
                         double Q,
                         cudaReal* result,
                         int size)
        {

            int nThreads = blockDim.x * gridDim.x;
            int startID = blockIdx.x * blockDim.x + threadIdx.x;

            for(int i = startID; i < size; i += nThreads) 
            {
                if (i != 0)
                    result[i] -= dk[i]*dphi[i]*qt[i]*qts[i]/Q;
            }
        }

        /*
         * Constructor
         */ 
        template <int D>
        DPDdiblock<D>::DPDdiblock()
        : nMonomer_(0)
        {
            setClassName("DPDdiblock");
        }

        /*
         * Destructor
         */
        template <int D>
        DPDdiblock<D>::~DPDdiblock()
        {
            monomers_.deallocate();
            blocks_.deallocate();

            delete [] expA_host;
            delete [] expAB_host;
            delete [] expB_host;

            expA_.deallocate();
            expAB_.deallocate();
            expB_.deallocate();
            bu0_.deallocate();
            expWA_.deallocate();
            expWB_.deallocate();
            expWAp_.deallocate();
            expWBp_.deallocate();
            qr_.deallocate();
            qk_.deallocate();
            cudaFree(qFields_d);
            cudaFree(qtFields_d);

            delete [] q_;
            delete [] qt_;
        }

        template <int D>
        void DPDdiblock<D>::readParameters(std::istream& in)
        {
            read<int>(in, "nMonomer", nMonomer_);
            monomers_.allocate(nMonomer_);
            readDArray< Monomer >(in, "monomers", monomers_, nMonomer_);
            eps_ = monomers_[0].step()/monomers_[1].step();
            read<int>(in, "nBlock", nBlock_);
            blocks_.allocate(nBlock_);
            readDArray< Block >(in, "blocks", blocks_, nBlock_);
            NA_ = blocks_[0].length();
            read<int>(in, "N", N_);
            read<double>(in, "sigma", sigma_);
            read<double>(in, "kpN", kpN_);

            if(kpN_ >= 0)
            {
                comp_ = true;
            }
            else
            {
                comp_ = false;
            }

            read<double>(in, "chiN", chiN_);

            chiInverse_.allocate(nMonomer(), nMonomer());
            idemp_.allocate(nMonomer(), nMonomer());

            double det = chiN_*chiN_;
            double norm = 2.0*chiN_*chiN_;
            if (fabs(det/norm) < 1.0E-10) 
            {
                UTIL_THROW("Singular chi matrix");
            }
            chiInverse_(0,1) = -chiN_/det;
            chiInverse_(1,0) = -chiN_/det;
            chiInverse_(1,1) = 0;
            chiInverse_(0,0) = 0;

            double sum = 0;
            int i, j, k;

            for (i = 0; i < nMonomer(); ++i) 
            {
                idemp_(0,i) = 0;
                for (j = 0; j < nMonomer(); ++j) 
                {
                    idemp_(0,i) -= chiInverse_(j,i);
                }
                sum -= idemp_(0,i);
                for (k = 0; k < nMonomer(); ++k) 
                { //row
                    idemp_(k,i) = idemp_(0,i);
                }
            }

            for (i = 0; i < nMonomer(); ++i) 
            { //row
                for (j = 0; j < nMonomer(); ++j) 
                { //coloumn
                    idemp_(i,j) /= sum;
                }
                idemp_(i,i) +=1 ;
            }
        }

        template <int D>
        void DPDdiblock<D>::setMesh(Mesh<D> const& mesh)
        {
            meshPtr_ = &mesh;

            // Compute Fourier space kMeshDimensions_
            for (int i = 0; i < D; ++i) 
            {
                if (i < D - 1) 
                {
                    kMeshDimensions_[i] = mesh.dimensions()[i];
                } else 
                {
                    kMeshDimensions_[i] = mesh.dimensions()[i]/2 + 1;
                }
            }

            kSize_ = 1;
            for(int i = 0; i < D; ++i) 
            {
                kSize_ *= kMeshDimensions_[i];
            }

            expA_host = new cudaReal[kSize_];
            expAB_host = new cudaReal[kSize_];
            expB_host = new cudaReal[kSize_];
            
            expA_.allocate(kMeshDimensions_);
            expAB_.allocate(kMeshDimensions_);
            expB_.allocate(kMeshDimensions_);
            expWA_.allocate(mesh.dimensions());
            expWB_.allocate(mesh.dimensions());
            expWAp_.allocate(mesh.dimensions());
            expWBp_.allocate(mesh.dimensions());
            qr_.allocate(mesh.dimensions());
            qk_.allocate(mesh.dimensions());

            cudaMalloc((void**)&qFields_d, sizeof(cudaReal)* mesh.size() * N_);
            cudaMalloc((void**)&qtFields_d, sizeof(cudaReal)* mesh.size() * N_);
            fft_.setup(qr_, qk_);

            q_ = new double[mesh.size()];
            qt_ = new double[mesh.size()];
        }

        template <int D>
        void DPDdiblock<D>::setupUnitCell(const UnitCell<D>& unitCell, 
                                          const WaveList<D>& wavelist)
        { 
            nParams_ = unitCell.nParameter();
            MeshIterator<D> iter;
            iter.setDimensions(kMeshDimensions_);
            IntVec<D> G;
            double Gsq;
            double  factor = -1.0/double(N_);

            //setup expKsq values on Host then transfer to device
            int kSize = 1;
            for(int i = 0; i < D; ++i) 
            {
                kSize *= kMeshDimensions_[i];
            }

            int i;
            for (iter.begin(); !iter.atEnd(); ++iter) 
            {
                i = iter.rank();
                Gsq = unitCell.ksq(wavelist.minImage(iter.rank()));
                expB_host[i] = exp(Gsq*factor);
                expAB_host[i] = exp(Gsq*factor*eps());
                expA_host[i] = exp(Gsq*factor*eps()*eps());
            }
            cudaMemcpy(expB_.cDField(),  expB_host,  kSize * sizeof(cudaReal), cudaMemcpyHostToDevice);
            cudaMemcpy(expAB_.cDField(), expAB_host, kSize * sizeof(cudaReal), cudaMemcpyHostToDevice);
            cudaMemcpy(expA_.cDField(),  expA_host,  kSize * sizeof(cudaReal), cudaMemcpyHostToDevice);
        }

        template <int D>
        void DPDdiblock<D>::allocate(Basis<D> const& basis)
        {
            basisPtr_ = &basis;
            int ns =  basis.nStar();
            bu0_.allocate(ns);
        }

        template <int D>
        void DPDdiblock<D>::setBasis(Basis<D> const& basis, UnitCell<D>& unitCell)
        {
            basisPtr_ = &basis;
            int ns =  basis.nStar();
            int hkltmp[D];
            double q;
            
            bu0_host = new cudaReal[ns];

            for (int i = 0; i < ns; ++i)
            {
                q =0.0;
                if (!basis.star(i).cancel)
                {
                    for(int j= 0; j < D; ++j)
                    {
                        hkltmp[j] = basis.star(i).waveBz[j];
                        // std::cout<< basis.star(i).waveBz[j] << "  ";
                    }
                    // std::cout << std::endl;

                    IntVec<D> hkl(hkltmp);

                    q = unitCell.ksq(hkl);
                    q = sqrt(q);
                    // std::cout<<q<<std::endl;
                    q *= sigma_;
                    // std::cout<< q << std::endl;

                    if(q != 0)
                    {
                        bu0_host[i] = 60.0 * (2*q + q*cos(q) - 3*sin(q)) / pow(q, 5.0);
                        // bu0_host[i] = 1.0;
                    }
                    else
                    {
                        bu0_host[i] = 1.0;
                    }
                    // std::cout<<bu0_host[i]<<std::endl;
                }
                else
                {
                    bu0_host[i] = 0.0;
                }
            }

            cudaMemcpy(bu0_.cDField(), bu0_host, ns*sizeof(cudaReal),cudaMemcpyHostToDevice);

            delete [] bu0_host;
        }

        template <int D>
        void DPDdiblock<D>::compute(DArray<RDField<D>> const & wFields,
                                    DArray<RDField<D>>& cFields)
        {
            int nx = mesh().size();
            int nm = nMonomer();
            double ds = 1.0/N_;

            int NA = blocks_[0].length();
            int NB = blocks_[1].length();

            for(int i = 0; i < nm; ++i)
            {
                UTIL_CHECK(cFields[i].capacity() == nx)
                UTIL_CHECK(wFields[i].capacity() == nx)
                assignUniformReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>(cFields[i].cDField(), 0.0, nx);
            }
            assignExp<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
            (expWA_.cDField(), wFields[0].cDField(), nx, ds);
            assignExp<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
            (expWB_.cDField(), wFields[1].cDField(), nx, ds);
            assignExp<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
            (expWAp_.cDField(), wFields[0].cDField(), nx, -ds);
            assignExp<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
            (expWBp_.cDField(), wFields[1].cDField(), nx, -ds);
            

            pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
            (expWA_.cDField(), qFields_d, nx);

            int currentIdx;
            for(int s = 0; s < NA-1; ++s)
            {
                currentIdx = s * nx;
                pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (qFields_d + currentIdx, qr_.cDField(), nx);

                fft_.forwardTransform(qr_, qk_);

                scaleComplex<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (qk_.cDField(), expA_.cDField(), kSize_);;

                fft_.inverseTransform(qk_, qr_);

                scaleReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (qr_.cDField(), expWA_.cDField(), nx);

                pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (qr_.cDField(), qFields_d + currentIdx + nx, nx);

                // double qold[1], qnew[1];
                // cudaMemcpy(qold, qFields_d +currentIdx,sizeof(cudaReal),cudaMemcpyDeviceToHost);
                // cudaMemcpy(qnew, qFields_d +currentIdx+nx,sizeof(cudaReal),cudaMemcpyDeviceToHost);
                // std::cout<< qold[0] << " -> " << qnew[0] << "\n";
            }
// std::cout << "===========================" <<"\n";
            pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
            (qFields_d + (NA-1)*nx, qr_.cDField(), nx);

            fft_.forwardTransform(qr_, qk_);
            
            scaleComplex<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
            (qk_.cDField(), expAB_.cDField(), kSize_);

            fft_.inverseTransform(qk_, qr_);
        
            scaleReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
            (qr_.cDField(), expWB_.cDField(), nx);    

            pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
            (qr_.cDField(), qFields_d +  NA*nx, nx);

            // double qold[nx], qnew[nx], qolder[nx];
            // cudaMemcpy(qolder, qFields_d +(NA-2)*nx,sizeof(cudaReal)*nx,cudaMemcpyDeviceToHost);
            // cudaMemcpy(qold, qFields_d +(NA-1)*nx,sizeof(cudaReal)*nx,cudaMemcpyDeviceToHost);
            // cudaMemcpy(qnew, qFields_d +NA*nx,sizeof(cudaReal)*nx,cudaMemcpyDeviceToHost);
            // // std::cout<< qold[0] << " -> " << qnew[0] << "\n";
            // for(int i = 0; i < nx; i++)
            //     std::cout << qolder[i] << std::endl;
            // std::cout << "===========================" <<"\n";
            // for(int i = 0; i < nx; i++)
            //     std::cout << qold[i] << std::endl;
            // std::cout << "===========================" <<"\n";
            // for(int i = 0; i < nx; i++)
            //     std::cout << qnew[i] << std::endl;
            // std::cout << "===========================" <<"\n";

            for(int s = NA; s < N_-1; ++s)
            {
                currentIdx = s * nx;
                pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (qFields_d + currentIdx, qr_.cDField(), nx);

                fft_.forwardTransform(qr_, qk_);

                scaleComplex<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (qk_.cDField(), expB_.cDField(), kSize_);

                fft_.inverseTransform(qk_, qr_);

                scaleReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (qr_.cDField(), expWB_.cDField(), nx);

                pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (qr_.cDField(), qFields_d + currentIdx + nx, nx);

                // cudaMemcpy(qold, qFields_d +currentIdx,sizeof(cudaReal),cudaMemcpyDeviceToHost);
                // cudaMemcpy(qnew, qFields_d +currentIdx+nx,sizeof(cudaReal),cudaMemcpyDeviceToHost);
                // std::cout<< qold[0] << " -> " << qnew[0] << "\n";
            }
            // std::cout << "===========================" <<"\n";

            fft_.forwardTransform(qr_, qk_);
            cudaComplex Q[1];
            // cudaReal Qr[nx] ,sum = 0.0;
            cudaMemcpy(Q, qk_.cDField(), sizeof(cudaComplex), cudaMemcpyDeviceToHost);
            // cudaMemcpy(Qr, qr_.cDField(), nx*sizeof(cudaReal), cudaMemcpyDeviceToHost);

            // for(int i= 0; i < nx; ++i)
            //     std::cout << "Qr = " << Qr[i] <<"\n";
            // for(int i= 0; i < nx; ++i)
            //     sum += Qr[i]/nx;
            // std::cout << "sum = " << sum <<"\n";
            Q_ = Q[0].x;
            // std::cout << "Q = " << Q[0].x << "  "<< Q[0].y <<"\n";


            pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
            (expWB_.cDField(), qtFields_d, nx);

             for(int s = 0; s < NB-1; ++s)
            {
                currentIdx = s * nx;
                pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (qtFields_d + currentIdx, qr_.cDField(), nx);

                fft_.forwardTransform(qr_, qk_);

                scaleComplex<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (qk_.cDField(), expB_.cDField(), kSize_);

                fft_.inverseTransform(qk_, qr_);

                scaleReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (qr_.cDField(), expWB_.cDField(), nx);

                pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (qr_.cDField(), qtFields_d + currentIdx + nx, nx);

                // cudaMemcpy(qold, qtFields_d +currentIdx,sizeof(cudaReal),cudaMemcpyDeviceToHost);
                // cudaMemcpy(qnew, qtFields_d +currentIdx+nx,sizeof(cudaReal),cudaMemcpyDeviceToHost);
                // std::cout<< qold[0] << " -> " << qnew[0] << "\n";
            }

            pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
            (qtFields_d + (NB-1)*nx, qr_.cDField(), nx);

            fft_.forwardTransform(qr_, qk_);
            
            scaleComplex<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
            (qk_.cDField(), expAB_.cDField(), kSize_);

            fft_.inverseTransform(qk_, qr_);
        
            scaleReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
            (qr_.cDField(), expWA_.cDField(), nx);    

            pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
            (qr_.cDField(), qtFields_d +  NB*nx, nx);

            // cudaMemcpy(qold, qtFields_d +(NB-1)*nx,sizeof(cudaReal),cudaMemcpyDeviceToHost);
            // cudaMemcpy(qnew, qtFields_d +NB*nx,sizeof(cudaReal),cudaMemcpyDeviceToHost);
            // std::cout<< qold[0] << " -> " << qnew[0] << "\n";
            // double qold[nx], qnew[nx];
            // cudaMemcpy(qolder, qtFields_d +(NB-2)*nx,sizeof(cudaReal)*nx,cudaMemcpyDeviceToHost);
            // cudaMemcpy(qold, qtFields_d +(NB-1)*nx,sizeof(cudaReal)*nx,cudaMemcpyDeviceToHost);
            // cudaMemcpy(qnew, qtFields_d +NB*nx,sizeof(cudaReal)*nx,cudaMemcpyDeviceToHost);
            // for(int i = 0; i < nx; i++)
            //     std::cout << qolder[i] << std::endl;
            // std::cout << "===========================" <<"\n";
            // for(int i = 0; i < nx; i++)
            //     std::cout << qold[i] << std::endl;
            // std::cout << "===========================" <<"\n";
            // for(int i = 0; i < nx; i++)
            //     std::cout << qnew[i] << std::endl;
            // std::cout << "===========================" <<"\n";

            for(int s = NB; s < N_-1; ++s)
            {
                currentIdx = s * nx;
                pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (qtFields_d + currentIdx, qr_.cDField(), nx);

                fft_.forwardTransform(qr_, qk_);

                scaleComplex<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (qk_.cDField(), expA_.cDField(), kSize_);

                fft_.inverseTransform(qk_, qr_);

                scaleReal<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (qr_.cDField(), expWA_.cDField(), nx);

                pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (qr_.cDField(), qtFields_d + currentIdx + nx, nx);

                // cudaMemcpy(qold, qtFields_d +currentIdx,sizeof(cudaReal),cudaMemcpyDeviceToHost);
                // cudaMemcpy(qnew, qtFields_d +currentIdx+nx,sizeof(cudaReal),cudaMemcpyDeviceToHost);
                // std::cout<< qold[0] << " -> " << qnew[0] << "\n";
            }

            cudaMemset(cFields[0].cDField(), 0, sizeof(cudaReal)*nx);
            cudaMemset(cFields[1].cDField(), 0, sizeof(cudaReal)*nx);

            for(int s = 0; s < NA; ++s)
            {
                int currentIdx1 = s * nx;
                int currentIdx2 = (N_-s-1) * nx;

                pointwiseTriple<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (cFields[0].cDField(), expWAp_.cDField(),
                 qFields_d+currentIdx1, qtFields_d+currentIdx2, nx);
            }

            for(int s = NA; s < N_; ++s)
            {
                int currentIdx1 = s * nx;
                int currentIdx2 = (N_-s-1) * nx;

                pointwiseTriple<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (cFields[1].cDField(), expWBp_.cDField(),
                 qFields_d+currentIdx1, qtFields_d+currentIdx2, nx);
            }

            double scale = 1.0/(Q[0].x*N_);
            scaleRealData<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
            (cFields[0].cDField(), scale, nx);
            scaleRealData<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
            (cFields[1].cDField(), scale, nx);

        //     double c1[nx], c2[nx];
        //    cudaMemcpy(c1, cFields[0].cDField(),
        //               sizeof(cudaReal)*nx,
        //               cudaMemcpyDeviceToHost);
        //    cudaMemcpy(c2, cFields[1].cDField(),
        //               sizeof(cudaReal)*nx,
        //               cudaMemcpyDeviceToHost);
        //    for(int i = 0; i < nx; ++i)
        //        std::cout << "  c[" << i << "] = "
        //                  << c1[i] << "   " << c2[i] << "\n";
            
        }

        template <int D>
        void DPDdiblock<D>::computeStress(Basis<D> const& basis, UnitCell<D>& unitCell, FieldIo<D> & fieldIo,
                                          DArray<RDField<D>>& cFields)
        {
            unitCell.setLattice();
            int ns = basis.nStar();
            int nx = mesh().size(); 
            int NA = blocks_[0].length();
            int NB = blocks_[1].length();
            int hkltmp[D];
            double  factor = -1.0/double(N_);
            double sqk, k, q;
            double *dk,
                   *dbu0,
                   *dPhiA, *dPhiAB, *dPhiB;
            
            cudaReal *dk_d, *dbu0_d, *dPhiA_d, *dPhiAB_d, *dPhiB_d;

            cudaReal *qt, *qts;
            RDField<D> temp_b, temp_r;
            RDFieldDft<D> temp_k;

            dk     = new double[nParams_*ns];
            dbu0   = new double[nParams_*ns];
            dPhiA  = new double[nParams_*ns];
            dPhiAB = new double[nParams_*ns];
            dPhiB  = new double[nParams_*ns];

            cudaMalloc((void**)&qt, sizeof(cudaReal)* ns * N_);
            cudaMalloc((void**)&qts, sizeof(cudaReal)* ns * N_);
            cudaMalloc((void**)&dk_d, sizeof(cudaReal)* ns * nParams_);
            cudaMalloc((void**)&dbu0_d, sizeof(cudaReal)* ns * nParams_);
            cudaMalloc((void**)&dPhiA_d, sizeof(cudaReal)* ns * nParams_);
            cudaMalloc((void**)&dPhiAB_d, sizeof(cudaReal)* ns * nParams_);
            cudaMalloc((void**)&dPhiB_d, sizeof(cudaReal)* ns * nParams_);

            temp_b.allocate(ns);
            temp_r.allocate(mesh().dimensions());
            temp_k.allocate(mesh().dimensions());

            // Propagators: r_grid => basis
            for(int i = 0; i < N_; ++i)
            {
                pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (qFields_d + i*nx, temp_r.cDField(), nx);
                
                fft_.forwardTransform(temp_r, temp_k);

                fieldIo.convertKGridToBasis(temp_k, temp_b);

                pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (temp_b.cDField(), qt + i*ns, ns);

                pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (qtFields_d + i*nx, temp_r.cDField(), nx);
                
                fft_.forwardTransform(temp_r, temp_k);

                fieldIo.convertKGridToBasis(temp_k, temp_b);

                pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (temp_b.cDField(), qts + i*ns, ns);
            }

            // stress_ : initialized to be 0.0;
            for (int i = 0; i < 6; ++i)
            {
                stress_[i] = 0.0;
            }

            for(int n = 0; n < nParams_; ++n)
            {
                for(int i = 0; i < ns; ++i)
                {
                    if (!basis.star(i).cancel)
                    {
                        for(int j= 0; j < D; ++j)
                        {
                            hkltmp[j] = basis.star(i).waveBz[j];
                            // std::cout<< basis.star(i).waveBz[j];
                        }
                        // std::cout<<std::endl;
                        IntVec<D> hkl(hkltmp);
                        sqk = unitCell.ksq(hkl);
                        k = sqrt(sqk);
                        dPhiA[i + ns*n] = 2.0*factor*k*eps()*eps()*exp(sqk*factor*eps()*eps());
                        dPhiAB[i + ns*n] = 2.0*factor*k*eps()*exp(sqk*factor*eps());
                        dPhiB[i + ns*n] = 2.0*factor*k*exp(sqk*factor);
                        if(sqk != 0)
                        {
                            q = sigma()*k;
                            dk[i + ns*n] = 0.5*unitCell.dksq(hkl, n)/sqrt(unitCell.ksq(hkl));
                            dbu0[i + ns*n] = -60.0*((q*q -15.0)*sin(q) + 8*q + 7*q*cos(q)) / pow(q, 6.0);
                        }
                        else
                        {
                            dk[i + ns*n]   = 0.0;
                            dbu0[i + ns*n] = 0.0;
                        }
                    }
                    else
                    {
                        dk[i + ns*n]     = 0.0;
                        dbu0[i + ns*n]   = 0.0;
                        dPhiA[i + ns*n]  = 0.0;
                        dPhiAB[i + ns*n] = 0.0;
                        dPhiB[i + ns*n]  = 0.0;
                    }
                }
            }

            cudaMemcpy(dk_d,     dk, sizeof(cudaReal)*ns*nParams_,     cudaMemcpyHostToDevice);
            cudaMemcpy(dbu0_d,   dbu0, sizeof(cudaReal)*ns*nParams_,   cudaMemcpyHostToDevice);
            cudaMemcpy(dPhiA_d,  dPhiA, sizeof(cudaReal)*ns*nParams_,  cudaMemcpyHostToDevice);
            cudaMemcpy(dPhiAB_d, dPhiAB, sizeof(cudaReal)*ns*nParams_, cudaMemcpyHostToDevice);
            cudaMemcpy(dPhiB_d,  dPhiB, sizeof(cudaReal)*ns*nParams_,  cudaMemcpyHostToDevice);

            for(int n = 0; n < nParams_; ++n)
            {
                cudaReal *stress_tmp;
                cudaMalloc((void**)&stress_tmp, sizeof(cudaReal)* ns);
                cudaMemset(stress_tmp, 0, sizeof(cudaReal)*ns);

                dpdStress_c<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (cFields[0].cDField(), cFields[1].cDField(),
                 dk_d + n*ns, dbu0_d + n*ns,
                 chiN_, kpN_, sigma_,
                 stress_tmp,
                 ns);

                for(int s = 0; s < NA -1; ++s)
                {
                    dpdStress_q<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (qt + s*ns, qts + (N_-2-s)*ns,
                     dk_d + n*ns, dPhiA_d + n*ns,
                     Q_,
                     stress_tmp,
                     ns);
                }

                dpdStress_q<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (qt + (NA-1)*ns, qts + (N_-1 -NA)*ns,
                 dk_d +  n*ns, dPhiAB_d + n*ns,
                 Q_,
                 stress_tmp,
                 ns);

                for(int s = NA; s < N_ -1; ++s)
                {
                    dpdStress_q<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (qt + s*ns, qts + (N_-2 -s)*ns,
                     dk_d + n*ns, dPhiB_d + n*ns,
                     Q_,
                     stress_tmp,
                     ns);
                }

                double* temp;
                temp = new double[ns];
                cudaMemcpy(temp, stress_tmp, sizeof(double)*ns, cudaMemcpyDeviceToHost);
                for(int i = 0; i < ns; ++i)
                    stress_[n] += temp[i];
                // std::cout << stress_[n] << "\n";
                // std::cout << nParams_ << "\n";
                cudaFree(stress_tmp);
                delete [] temp;
            }

            delete [] dk;
            delete [] dbu0;
            delete [] dPhiA;
            delete [] dPhiAB;
            delete [] dPhiB;

            temp_b.deallocate();
            temp_r.deallocate();
            temp_k.deallocate();

            cudaFree(qt);
            cudaFree(qts);
            cudaFree(dk_d);
            cudaFree(dbu0_d);
            cudaFree(dPhiA_d);
            cudaFree(dPhiAB_d);
            cudaFree(dPhiB_d);
        }

        template <int D>
        void DPDdiblock<D>::computeStress_incmp(Basis<D> const& basis, UnitCell<D>& unitCell, FieldIo<D> & fieldIo,
                                          DArray<RDField<D>>& cFields)
        {
            unitCell.setLattice();
            int ns = basis.nStar();
            int nx = mesh().size(); 
            int NA = blocks_[0].length();
            int NB = blocks_[1].length();
            int hkltmp[D];
            double  factor = -1.0/double(N_);
            double sqk, k, q;
            double *dk,
                   *dbu0,
                   *dPhiA, *dPhiAB, *dPhiB;
            
            cudaReal *dk_d, *dbu0_d, *dPhiA_d, *dPhiAB_d, *dPhiB_d;

            cudaReal *qt, *qts;
            RDField<D> temp_b, temp_r;
            RDFieldDft<D> temp_k;

            dk     = new double[nParams_*ns];
            dbu0   = new double[nParams_*ns];
            dPhiA  = new double[nParams_*ns];
            dPhiAB = new double[nParams_*ns];
            dPhiB  = new double[nParams_*ns];

            cudaMalloc((void**)&qt, sizeof(cudaReal)* ns * N_);
            cudaMalloc((void**)&qts, sizeof(cudaReal)* ns * N_);
            cudaMalloc((void**)&dk_d, sizeof(cudaReal)* ns * nParams_);
            cudaMalloc((void**)&dbu0_d, sizeof(cudaReal)* ns * nParams_);
            cudaMalloc((void**)&dPhiA_d, sizeof(cudaReal)* ns * nParams_);
            cudaMalloc((void**)&dPhiAB_d, sizeof(cudaReal)* ns * nParams_);
            cudaMalloc((void**)&dPhiB_d, sizeof(cudaReal)* ns * nParams_);

            temp_b.allocate(ns);
            temp_r.allocate(mesh().dimensions());
            temp_k.allocate(mesh().dimensions());

            // Propagators: r_grid => basis
            for(int i = 0; i < N_; ++i)
            {
                pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (qFields_d + i*nx, temp_r.cDField(), nx);
                
                fft_.forwardTransform(temp_r, temp_k);

                fieldIo.convertKGridToBasis(temp_k, temp_b);

                pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (temp_b.cDField(), qt + i*ns, ns);

                pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (qtFields_d + i*nx, temp_r.cDField(), nx);
                
                fft_.forwardTransform(temp_r, temp_k);

                fieldIo.convertKGridToBasis(temp_k, temp_b);

                pointwiseEqual<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>>
                (temp_b.cDField(), qts + i*ns, ns);
            }

            // stress_ : initialized to be 0.0;
            for (int i = 0; i < 6; ++i)
            {
                stress_[i] = 0.0;
            }

            for(int n = 0; n < nParams_; ++n)
            {
                for(int i = 0; i < ns; ++i)
                {
                    if (!basis.star(i).cancel)
                    {
                        for(int j= 0; j < D; ++j)
                        {
                            hkltmp[j] = basis.star(i).waveBz[j];
                            // std::cout<< basis.star(i).waveBz[j];
                        }
                        // std::cout<<std::endl;
                        IntVec<D> hkl(hkltmp);
                        sqk = unitCell.ksq(hkl);
                        k = sqrt(sqk);
                        dPhiA[i + ns*n] = 2.0*factor*k*eps()*eps()*exp(sqk*factor*eps()*eps());
                        dPhiAB[i + ns*n] = 2.0*factor*k*eps()*exp(sqk*factor*eps());
                        dPhiB[i + ns*n] = 2.0*factor*k*exp(sqk*factor);
                        if(sqk != 0)
                        {
                            q = sigma()*k;
                            dk[i + ns*n] = 0.5*unitCell.dksq(hkl, n)/sqrt(unitCell.ksq(hkl));
                            dbu0[i + ns*n] = -60.0*((q*q -15.0)*sin(q) + 8*q + 7*q*cos(q)) / pow(q, 6.0);
                        }
                        else
                        {
                            dk[i + ns*n]   = 0.0;
                            dbu0[i + ns*n] = 0.0;
                        }
                    }
                    else
                    {
                        dk[i + ns*n]     = 0.0;
                        dbu0[i + ns*n]   = 0.0;
                        dPhiA[i + ns*n]  = 0.0;
                        dPhiAB[i + ns*n] = 0.0;
                        dPhiB[i + ns*n]  = 0.0;
                    }
                }
            }

            cudaMemcpy(dk_d,     dk, sizeof(cudaReal)*ns*nParams_,     cudaMemcpyHostToDevice);
            cudaMemcpy(dbu0_d,   dbu0, sizeof(cudaReal)*ns*nParams_,   cudaMemcpyHostToDevice);
            cudaMemcpy(dPhiA_d,  dPhiA, sizeof(cudaReal)*ns*nParams_,  cudaMemcpyHostToDevice);
            cudaMemcpy(dPhiAB_d, dPhiAB, sizeof(cudaReal)*ns*nParams_, cudaMemcpyHostToDevice);
            cudaMemcpy(dPhiB_d,  dPhiB, sizeof(cudaReal)*ns*nParams_,  cudaMemcpyHostToDevice);

            for(int n = 0; n < nParams_; ++n)
            {
                cudaReal *stress_tmp;
                cudaMalloc((void**)&stress_tmp, sizeof(cudaReal)* ns);
                cudaMemset(stress_tmp, 0, sizeof(cudaReal)*ns);

                dpdStress_inc<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (cFields[0].cDField(), cFields[1].cDField(),
                 dk_d + n*ns, dbu0_d + n*ns,
                 chiN_, sigma_,
                 stress_tmp,
                 ns);

                for(int s = 0; s < NA -1; ++s)
                {
                    dpdStress_q<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (qt + s*ns, qts + (N_-2-s)*ns,
                     dk_d + n*ns, dPhiA_d + n*ns,
                     Q_,
                     stress_tmp,
                     ns);
                }

                dpdStress_q<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                (qt + (NA-1)*ns, qts + (N_-1 -NA)*ns,
                 dk_d +  n*ns, dPhiAB_d + n*ns,
                 Q_,
                 stress_tmp,
                 ns);

                for(int s = NA; s < N_ -1; ++s)
                {
                    dpdStress_q<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>
                    (qt + s*ns, qts + (N_-2 -s)*ns,
                     dk_d + n*ns, dPhiB_d + n*ns,
                     Q_,
                     stress_tmp,
                     ns);
                }

                double* temp;
                temp = new double[ns];
                cudaMemcpy(temp, stress_tmp, sizeof(double)*ns, cudaMemcpyDeviceToHost);
                for(int i = 0; i < ns; ++i)
                    stress_[n] += temp[i];
                // std::cout << stress_[n] << "\n";
                // std::cout << nParams_ << "\n";
                cudaFree(stress_tmp);
                delete [] temp;
            }

            delete [] dk;
            delete [] dbu0;
            delete [] dPhiA;
            delete [] dPhiAB;
            delete [] dPhiB;

            temp_b.deallocate();
            temp_r.deallocate();
            temp_k.deallocate();

            cudaFree(qt);
            cudaFree(qts);
            cudaFree(dk_d);
            cudaFree(dbu0_d);
            cudaFree(dPhiA_d);
            cudaFree(dPhiAB_d);
            cudaFree(dPhiB_d);
        }
    }
}
}

#endif