#ifndef FCT_H
#define FCT_H
#include <iostream>
#include <cmath>
#include "cufft.h"
#include "matrixTranspose.h"
#include <stdio.h>
#include <stdlib.h>

static
__global__
void normalizationFFT(cufftDoubleReal   *data,
                          cufftDoubleReal   scale,
                          int               NX,
                          int               NY,
                          int               NZ);
static
__global__
void normalizationForward(cufftDoubleReal   *data,
                          cufftDoubleReal   scale,
                          int               NX,
                          int               NY,
                          int               NZ);
static
__global__
void normalizationInverse(cufftDoubleReal   *data,
                          cufftDoubleReal   scale,
                          int               NX,
                          int               NY,
                          int               NZ);

static
__global__
void maketri(cufftDoubleReal *sin_,
             cufftDoubleReal *cos_,
             int              NX);
static
__global__
void preForwardTransform(cufftDoubleComplex *c_data, 
                         cufftDoubleReal    *data, 
                         int                 Nx, 
                         int                 Ny, 
                         int                 Nz_2);
static
__global__
void preInverseTransform(cufftDoubleReal  *data,
                         cufftDoubleReal  *r_data,
                         double           *sin_,
                         double           *cos_,
                         int               NX,
                         int               NY,
                         int               NZ);

static
__global__
void postForwardTransform(cufftDoubleReal  *data,
                          cufftDoubleReal  *r_data,
                          double           *sin_,
                          double           *cos_,
                          int               NX,
                          int               NY,
                          int               NZ);
static
__global__
void postInverseTransform(cufftDoubleReal    *data, 
                          cufftDoubleComplex *c_data, 
                          int                 Nx, 
                          int                 Ny, 
                          int                 Nz_2);
template <int D>
class FCT
{
public:

    FCT();
    virtual ~FCT();

    void setup(int * mesh);

    bool isSetup() const;

    void forwardTransform(cufftDoubleReal * data);

    void inverseTransform(cufftDoubleReal * data);

private:

    int mesh_[3];

    cufftDoubleReal * sinX_;
    cufftDoubleReal * sinY_;
    cufftDoubleReal * sinZ_;

    cufftDoubleReal * cosX_;
    cufftDoubleReal * cosY_;
    cufftDoubleReal * cosZ_;

    cufftDoubleComplex * c_data_;
    cufftDoubleReal * r_data_;

    cufftHandle fPlanX_;
    cufftHandle fPlanY_;
    cufftHandle fPlanZ_;
    cufftHandle iPlanX_;
    cufftHandle iPlanY_;
    cufftHandle iPlanZ_;

    void makePlans(int * mesh);

    bool isSetup_;
};

template <int D>
inline 
bool 
FCT<D>::isSetup() const
{ 
    return isSetup_; 
}



static
__global__
void normalizationFFT(cufftDoubleReal   *data,
                      cufftDoubleReal   scale,
                      int               Nx,
                      int               Ny,
                      int               Nz)
{
    int nThreads = blockDim.x * gridDim.x;
    int startId = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = startId; i < Nx*Ny*Nz; i += nThreads ) {
        data[i] *= scale;
    }
}

static
__global__
void maketri(cufftDoubleReal *sin_,
             cufftDoubleReal *cos_,
             int              N)
{
    int nThreads = blockDim.x * gridDim.x;
    int startId = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = startId; i < N; i += nThreads ) {
        sin_[i] = sin(0.5*i*M_PI/N);
        cos_[i] = cos(0.5*i*M_PI/N);
    }
}


static
__global__
void preForwardTransform(cufftDoubleComplex *c, 
                         cufftDoubleReal    *r, 
                         int                 Nx, 
                         int                 Ny, 
                         int                 Nz_2) 
{
    int nThreads = blockDim.x * gridDim.x;
    int startId = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = startId; i < (Nz_2+1)*Ny*Nx; i += nThreads ) 
    {
        int Nz = 2*Nz_2;
        int x, y, z ,index = i;
        x = index/(Ny*(Nz_2+1)); 
		index %= Ny*(Nz_2+1); 
		y = index/(Nz_2+1); 
		z = index%(Nz_2+1);  
        if (z ==0)
        {
            c[z + (Nz_2+1)*y + Ny*(Nz_2+1)*x].x = r[z + Nz*y+ Ny*Nz*x];
            c[z + (Nz_2+1)*y + Ny*(Nz_2+1)*x].y = 0;  
            // printf("(%d, %d, %d)\n", x, y, z);
        }
        else if (z == Nz_2)
        {
            c[z + (Nz_2+1)*y + Ny*(Nz_2+1)*x].x = r[(2*z - 1) + Nz*y+ Ny*Nz*x];
            c[z + (Nz_2+1)*y + Ny*(Nz_2+1)*x].y = 0;  
            // printf("(%d, %d, %d)\n", x, y, z);
        }
        else
        {
            c[z + (Nz_2+1)*y + Ny*(Nz_2+1)*x].x = 0.5*(r[(2*z - 1) + Nz*y+ Ny*Nz*x] + r[2*z + Nz*y+ Ny*Nz*x]);
            c[z + (Nz_2+1)*y + Ny*(Nz_2+1)*x].y = 0.5*(r[2*z + Nz*y+ Ny*Nz*x] - r[(2*z - 1) + Nz*y+ Ny*Nz*x]);
            // printf("(%d, %d, %d)\n", x, y, z);
        }    
    }
}

static
__global__
void postForwardTransform(cufftDoubleReal  *data,
                          cufftDoubleReal  *rdata,
                          double           *sin_,
                          double           *cos_,
                          int               Nx,
                          int               Ny,
                          int               Nz)
{
    int nThreads = blockDim.x * gridDim.x;
    int startId = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = startId; i < Nx*Ny*Nz; i += nThreads ) 
    {
        int x, y, z ,index = i;
        x = index/(Ny*Nz); 
		index %= Ny*Nz; 
		y = index/Nz;
		z = index%Nz; 
        if (z > 0)
        {
            data[z + Nz*y + Ny*Nz*x] = 0.5*(
                                            (rdata[z + Nz*y + Ny*Nz*x]+rdata[(Nz-z) + Nz*y + Ny*Nz*x])*cos_[z]
                                           +(rdata[z + Nz*y + Ny*Nz*x]-rdata[(Nz-z) + Nz*y + Ny*Nz*x])*sin_[z]
                                           )*(1.0/double(Nz));
        }
        else
        {
            data[Nz*Ny*x + Nz*y] = rdata[Nz*Ny*x + Nz*y]*(1.0/double(Nz));
        }
    }
}

static
__global__
void preInverseTransform(cufftDoubleReal  *r_data,
                         cufftDoubleReal  *data,
                         double           *sin_,
                         double           *cos_,
                         int               Nx,
                         int               Ny,
                         int               Nz)
{
    int nThreads = blockDim.x * gridDim.x;
    int startId = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = startId; i < Nx*Ny*Nz; i += nThreads ) 
    {
        int x, y, z ,index = i;
        x = index/(Ny*Nz); 
		index %= Ny*Nz; 
		y = index/Nz;
		z = index%Nz; 
        if (z != 0)
            r_data[z + Nz*y + Ny*Nz*x] = ((((data[z + Nz*y + Ny*Nz*x] + data[(Nz-z) + Nz*y + Ny*Nz*x])*sin_[z]
                                         +(data[z + Nz*y + Ny*Nz*x] - data[(Nz-z) + Nz*y + Ny*Nz*x])*cos_[z])))
                                         *2;
        else 
            r_data[Nz*y + Ny*Nz*x] = data[Nz*y + Ny*Nz*x]*2.0;
            // r_data[Nz*y + Ny*Nx*x] = data[Nz*y + Ny*Nz*x] + data[(Nz-1) + Nz*y + Ny*Nz*x];                          
    }
}

static
__global__
void postInverseTransform(cufftDoubleReal    *data, 
                          cufftDoubleComplex *c_data, 
                          int                 Nx, 
                          int                 Ny, 
                          int                 Nz_2)
{
    int nThreads = blockDim.x * gridDim.x;
    int startId = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = startId; i < (Nz_2+1)*Ny*Nx; i += nThreads ) 
    {
        int Nz = 2*Nz_2;
        int x, y, z ,index = i;
        x = index/(Ny*(Nz_2+1)); 
		index %= Ny*(Nz_2+1); 
		y = index/(Nz_2+1); 
		z = index%(Nz_2+1); 
        if (z == 0)
        {
            data[Nz*y+ Ny*Nz*x] 
                = 0.5*c_data[(Nz_2+1)*y + Ny*(Nz_2+1)*x].x;
        }
        else if (z == Nz_2)
        {
            data[(Nz-1) + Nz*y+ Ny*Nz*x] 
                = 0.5*c_data[Nz_2 + (Nz_2+1)*y + Ny*(Nz_2+1)*x].x;
        }
        else
        {
            data[2*z + Nz*y+ Ny*Nz*x] 
                = 0.5*(c_data[z + (Nz_2+1)*y + Ny*(Nz_2+1)*x].x
                      +c_data[z + (Nz_2+1)*y + Ny*(Nz_2+1)*x].y);
            data[(2*z-1) + Nz*y+ Ny*Nz*x] 
                = 0.5*(c_data[z + (Nz_2+1)*y + Ny*(Nz_2+1)*x].x
                      -c_data[z + (Nz_2+1)*y + Ny*(Nz_2+1)*x].y);
        }    
    }
}

   #ifndef FCT_TPP
   // Suppress implicit instantiation
   extern template class FCT<1>;
   extern template class FCT<2>;
   extern template class FCT<3>;
   #endif

#endif