#include "fct.h"
#define Pi 3.1415926535897932384626

static
__global__
void cuMul(cufftDoubleReal* q, const double * K, int size_ph)
{
    int nThreads = blockDim.x * gridDim.x;
    int startId = blockIdx.x * blockDim.x + threadIdx.x;
    for(int idx = startId; idx < size_ph; idx += nThreads ) 
    {
        q[idx] *= K[idx]; 
    }
}

__global__
void K_mat_generator_dev(double *K, 
                         double ds, 
                         double Lx, 
                         double Ly, 
                         double Lz,
                         int Nx_2,
                         int Ny_2,
                         int Nz_2,
                         int size_ph)
{
    int nThreads = blockDim.x * gridDim.x;
    int startId = blockIdx.x * blockDim.x + threadIdx.x;
    for(int idx = startId; idx < size_ph; idx += nThreads ) 
    {
        int i, j, k ,index = idx;
        i = index/(Ny_2*Nz_2); 
		index %= Ny_2*Nz_2; 
		j = index/Nz_2; 
		k = index%Nz_2;  
        double ki, kj, kk;
        ki = 2 * Pi * i * 1.0 / Lx;
        ki *= ki;
        kj = 2 * Pi * j * 1.0 / Ly;
        kj *= kj;
        kk = 2 * Pi * k * 1.0 / Lz;
        kk *= kk;
        K[idx] = exp(-(ki + kj + kk) * ds);
    }
}

static
__global__
void expW_generator_dev(double * expW, 
                       double const * w, 
                       double factor, 
                       int size_ph)
{
    int nThreads = blockDim.x * gridDim.x;
    int startId = blockIdx.x * blockDim.x + threadIdx.x;
    for(int idx = startId; idx < size_ph; idx += nThreads ) 
    {
        expW[idx] = exp(-w[idx]*factor);
    }
}

class cuPmmm
{
public:

    cuPmmm();

    ~cuPmmm();

    void setSolver(int * mesh, 
                   double const * wA,
                   double const * wB,
                   double fA,
                   double ds, 
                   double Lx, 
                   double Ly, 
                   double Lz);
    
    void solve();

    void step(cufftDoubleReal * q);

    void expW_generator(double *expW, double const *w, double factor, int size_ph);

private:

    void K_mat_generator(double *K, 
                         double ds, 
                         double Lx_, 
                         double Ly_, 
                         double Lz,
                         int Nx_2,
                         int Ny_2,
                         int Nz_2,
                         int size_ph);

    int mesh_[3];
    int size_;
    int mesh_ph_[3];
    int size_ph_;
    double fA_, fB_;
    int nsA_, nsB_; 
    double ds_;

    double Lx_, Ly_, Lz_;

    FCT fct_;

    double *K_;

};

cuPmmm::cuPmmm()
{
    
}

cuPmmm::~cuPmmm()
{
    cudaFree(K_);
}

void
cuPmmm::setSolver(int * mesh, 
                  double const * wA,
                  double const * wB,
                  double fA,
                  double ds, 
                  double Lx, double Ly, double Lz)
{
    mesh_[0] = mesh[0];
    mesh_[1] = mesh[1];
    mesh_[2] = mesh[2];
    size_ = mesh_[0]*mesh_[1]*mesh_[2];
    mesh_ph_[0] = mesh[0];
    mesh_ph_[1] = mesh[1];
    mesh_ph_[2] = mesh[2];
    size_ph_ = mesh_ph_[0]*mesh_ph_[1]*mesh_ph_[2];
    ds_ = ds;
    Lx_ = Lx;  Ly_ = Ly;   Lz_ = Lz;

    fA_ = fA;
    fB_ = 1.0 - fA_;
    nsA_ = ceil(fA_ / ds_);
    nsB_ = ceil(fB_ / ds_);

    fct_.setup(mesh_ph_);

    cudaMalloc((void**)&K_, size_ph_ * sizeof(double));
    K_mat_generator(K_, 
                    ds_, 
                    Lx_, Ly_, Lz,
                    mesh_ph_[0],mesh_ph_[1],mesh_ph_[2],
                    size_ph_);

    
}

void 
cuPmmm::step(cufftDoubleReal * q)
{
    double t_tot;
    clock_t time1 = clock();
    fct_.forwardTransform(q);
    clock_t time2 = clock();
    double t1 = ((double)(time2 - time1)) / CLOCKS_PER_SEC ;
    // std::cout << "fct_forward running time: " << t1 << "s" << std::endl;

    cuMul<<<mesh_[1]*mesh_[2], 32>>>(q, K_, size_ph_);

    clock_t time3 = clock();
    fct_.inverseTransform(q);
    clock_t time4 = clock();
    double t2 = ((double)(time4 - time3)) / CLOCKS_PER_SEC ;
    // std::cout << "fct_inverse running time: " << t2 << "s" << std::endl;
    t_tot = t1+t2;
    std::cout << "step() running time: " << t_tot << "s" << std::endl;
    
}

void 
cuPmmm::K_mat_generator(double *K, 
                        double ds, 
                        double Lx, 
                        double Ly, 
                        double Lz,
                        int Nx_2,
                        int Ny_2,
                        int Nz_2,
                        int size_ph)
{
    K_mat_generator_dev<<<Nz_2*Ny_2,32>>>(K, ds,
                                          Lx, Ly, Lz, 
                                          Nx_2, Ny_2, Nx_2,
                                          size_ph);
}

void 
cuPmmm::expW_generator(double *expW, 
                       double const *w, 
                       double factor, 
                       int size_ph)
{
    expW_generator_dev<<<mesh_ph_[2]*mesh_ph_[1],32>>>
    (expW, w, factor, size_ph);
}