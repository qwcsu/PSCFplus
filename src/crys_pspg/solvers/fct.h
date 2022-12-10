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
class FCT
{
public:

    FCT();
    virtual ~FCT();

    void setup(int * mesh);

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
};

FCT::FCT()
{

}

FCT::~FCT()
{
    cudaFree(sinX_);
    cudaFree(sinY_);
    cudaFree(sinZ_);
    cudaFree(cosX_);
    cudaFree(cosY_);
    cudaFree(cosZ_);

}

void FCT::setup(int * mesh)
{
    mesh_[0] = mesh[0];
    mesh_[1] = mesh[1];
    mesh_[2] = mesh[2];

    cudaMalloc((void**)&sinX_, 
                mesh_[0] * sizeof(cufftDoubleReal));
    cudaMalloc((void**)&cosX_, 
                mesh_[0] * sizeof(cufftDoubleReal));
    cudaMalloc((void**)&sinY_, 
                mesh_[1] * sizeof(cufftDoubleReal));
    cudaMalloc((void**)&cosY_, 
                mesh_[1] * sizeof(cufftDoubleReal));
    cudaMalloc((void**)&sinZ_, 
                mesh_[2] * sizeof(cufftDoubleReal));
    cudaMalloc((void**)&cosZ_, 
                mesh_[2] * sizeof(cufftDoubleReal));

    makePlans(mesh_);
}

void FCT::makePlans(int * mesh)
{
    maketri<<<32,32>>>(sinX_, cosX_, mesh[0]);
    maketri<<<32,32>>>(sinY_, cosY_, mesh[1]);
    maketri<<<32,32>>>(sinZ_, cosZ_, mesh[2]);

    int inembed[] = {0};
    int onembed[] = {0};
    int meshX[] = {mesh[0]};
    int meshY[] = {mesh[1]};
    int meshZ[] = {mesh[2]};
    cufftPlanMany(&fPlanX_, 1, meshX, 
                  inembed, 1, mesh[0] / 2 + 1,
                  onembed, 1, mesh[0], 
                  CUFFT_Z2D, mesh[1]*mesh[2]);
    
    cufftPlanMany(&fPlanY_, 1, meshY, 
                  inembed, 1, mesh[1] / 2 + 1,
                  onembed, 1, mesh[1], 
                  CUFFT_Z2D, mesh[0]*mesh[2]);
    
    cufftPlanMany(&fPlanZ_, 1, meshZ, 
                  inembed, 1, mesh[2] / 2 + 1,
                  onembed, 1, mesh[2], 
                  CUFFT_Z2D, mesh[0]*mesh[1]);
    
    cufftPlanMany(&iPlanX_, 1, meshX, 
                  inembed, 1, mesh[0],
                  onembed, 1, mesh[0] / 2 + 1, 
                  CUFFT_D2Z, mesh[1]*mesh[2]);

    cufftPlanMany(&iPlanY_, 1, meshY, 
                  inembed, 1, mesh[1],
                  onembed, 1, mesh[1] / 2 + 1, 
                  CUFFT_D2Z, mesh[0]*mesh[2]);

    cufftPlanMany(&iPlanZ_, 1, meshZ, 
                  inembed, 1, mesh[2],
                  onembed, 1, mesh[2] / 2 + 1, 
                  CUFFT_D2Z, mesh[0]*mesh[1]);
}

void FCT::forwardTransform(cufftDoubleReal * data)
{
    int permutation[3];
    int perm_mesh[3];
    // z
    cudaMalloc((void**)&c_data_, 
                (mesh_[2]/2+1) * mesh_[1] * mesh_[0] * sizeof(cufftDoubleComplex));
    cudaMalloc((void**)&r_data_, 
                mesh_[2] * mesh_[1] * mesh_[0] * sizeof(cufftDoubleReal));
    
    
    // 
    //cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, data, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < mesh_[2]; z++)
    // {
    //     for(int y = 0; y < mesh_[1]; y++)
    //     {
    //         for(int x = 0; x < (mesh_[0]); x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    preForwardTransform<<<mesh_[0]*mesh_[1], 32>>>(c_data_, data, mesh_[0], mesh_[1], mesh_[2]/2);    
    //
    // cufftDoubleComplex *c_data_c;
    // c_data_c = new cufftDoubleComplex [(mesh_[2]/2+1) * mesh_[1] * mesh_[0] * sizeof(cufftDoubleComplex)];
    // cudaMemcpy(c_data_c, c_data_, (mesh_[2]/2+1) * mesh_[1] * mesh_[0] * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < (mesh_[2]/2+1); z++)
    // {
    //     for(int y = 0; y < mesh_[1]; y++)
    //     {
    //         for(int x= 0; x < mesh_[0]; x++)
    //         {
    //             std::cout << "(" << c_data_c[z + (mesh_[2]/2+1)*y + mesh_[1]*(mesh_[2]/2+1)*x].x << ", "
    //                              << c_data_c[z + (mesh_[2]/2+1)*y + mesh_[1]*(mesh_[2]/2+1)*x].y << ")  ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // exit(1);
    if (cufftExecZ2D(fPlanZ_, c_data_, r_data_)!= CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: ExecZ2D Forward failed");
        exit(0);
    } 
    // 
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, r_data_, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < mesh_[2]; z++)
    // {
    //     for(int y = 0; y < mesh_[1]; y++)
    //     {
    //         for(int x = 0; x < (mesh_[0]); x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // exit(1);
    postForwardTransform<<<mesh_[0]*mesh_[1], 32>>>(data, r_data_, sinZ_, cosZ_, mesh_[0], mesh_[1], mesh_[2]);
    normalizationFFT<<<mesh_[0]*mesh_[1],32>>>(data, 2.0/double(mesh_[2]), mesh_[0], mesh_[1], mesh_[2]);
    // 
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, data, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < mesh_[2]; z++)
    // {
    //     for(int y = 0; y < mesh_[1]; y++)
    //     {
    //         for(int x = 0; x < (mesh_[0]); x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // exit(1);
    cudaFree(c_data_);
    // Transpose data in z and y direction
    permutation[0] = 1;  
    permutation[1] = 0;  
    permutation[2] = 2; 
    perm_mesh[0] = mesh_[0];
    perm_mesh[1] = mesh_[1];
    perm_mesh[2] = mesh_[2];
    cut_transpose3d( r_data_,
                     data,
                     perm_mesh,
                     permutation,
                     1);
    data = r_data_;
    perm_mesh[0] = mesh_[1];
    perm_mesh[1] = mesh_[0];
    perm_mesh[2] = mesh_[2];
    // 
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, data, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < mesh_[2]; z++)
    // {
    //     for(int y = 0; y < mesh_[1]; y++)
    //     {
    //         for(int x = 0; x < mesh_[0]; x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // exit(1);

    // y
    cudaMalloc((void**)&c_data_, 
                (mesh_[1]/2+1) * mesh_[0] * mesh_[2] * sizeof(cufftDoubleComplex));

    preForwardTransform<<<mesh_[0]*mesh_[2], 32>>>(c_data_, data, mesh_[0], mesh_[2], mesh_[1]/2);
    if (cufftExecZ2D(fPlanY_, c_data_, r_data_)!= CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: ExecZ2D Forward failed");
        exit(0);
    }
    // 
    // cudaMemcpy(r_data_c, r_data_, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < mesh_[2]; z++)
    // {
    //     for(int y = 0; y < mesh_[1]; y++)
    //     {
    //         for(int x = 0; x < (mesh_[0]); x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    postForwardTransform<<<mesh_[0]*mesh_[2], 32>>>(data, r_data_, sinY_, cosY_, mesh_[0], mesh_[2], mesh_[1]);
    normalizationFFT<<<mesh_[0]*mesh_[2],32>>>(data, 2.0/double(mesh_[1]), mesh_[0], mesh_[1], mesh_[2]);
    // 
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, data, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < mesh_[2]; z++)
    // {
    //     for(int y = 0; y < mesh_[1]; y++)
    //     {
    //         for(int x = 0; x < (mesh_[0]); x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // exit(1);
    cudaFree(c_data_);
    // Transpose data in z and y direction
    permutation[0] = 1;
    permutation[1] = 0;
    permutation[2] = 2;  
    cut_transpose3d( r_data_,
                     data,
                     perm_mesh,
                     permutation,
                     1);
    data = r_data_;
    perm_mesh[0] = mesh_[0];
    perm_mesh[1] = mesh_[1];
    perm_mesh[2] = mesh_[2];
    // 
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, data, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < mesh_[2]; z++)
    // {
    //     for(int y = 0; y < mesh_[1]; y++)
    //     {
    //         for(int x = 0; x < (mesh_[0]); x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // exit(1);
    
    // x
    // Transpose data in x and z direction
    permutation[0] = 2;
    permutation[1] = 1;
    permutation[2] = 0;  
    cut_transpose3d( r_data_,
                     data,
                     perm_mesh,
                     permutation,
                     1);
    data = r_data_;
    // 
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, data, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < mesh_[2]; z++)
    // {
    //     for(int y = 0; y < mesh_[1]; y++)
    //     {
    //         for(int x = 0; x < (mesh_[0]); x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }exit(1);
    perm_mesh[0] = mesh_[2];
    perm_mesh[1] = mesh_[1];
    perm_mesh[2] = mesh_[0];
    // 
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, data, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < mesh_[2]; z++)
    // {
    //     for(int y = 0; y < mesh_[1]; y++)
    //     {
    //         for(int x = 0; x < (mesh_[0]); x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    cudaMalloc((void**)&c_data_, 
                (mesh_[0]/2+1) * mesh_[1] * mesh_[2] * sizeof(cufftDoubleComplex));
    //
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, data, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < mesh_[2]; z++)
    // {
    //     for(int y = 0; y < mesh_[1]; y++)
    //     {
    //         for(int x = 0; x < (mesh_[0]); x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // exit(1);    
    preForwardTransform<<<mesh_[1]*mesh_[2], 32>>>(c_data_, data, mesh_[2], mesh_[1], mesh_[0]/2);
    //
    // cufftDoubleComplex *c_data_c;
    // c_data_c = new cufftDoubleComplex [(mesh_[2]/2+1) * mesh_[1] * mesh_[0] * sizeof(cufftDoubleComplex)];
    // cudaMemcpy(c_data_c, c_data_, (mesh_[2]/2+1) * mesh_[1] * mesh_[0] * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < (mesh_[2]/2+1); z++)
    // {
    //     for(int y = 0; y < mesh_[1]; y++)
    //     {
    //         for(int x= 0; x < mesh_[0]; x++)
    //         {
    //             std::cout << "(" << c_data_c[z + (mesh_[2]/2+1)*y + mesh_[1]*(mesh_[2]/2+1)*x].x << ", "
    //                              << c_data_c[z + (mesh_[2]/2+1)*y + mesh_[1]*(mesh_[2]/2+1)*x].y << ")  ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    if (cufftExecZ2D(fPlanX_, c_data_, r_data_)!= CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: ExecZ2D Forward failed");
        exit(0);
    }
    //
    // cudaMemcpy(r_data_c, r_data_, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < mesh_[2]; z++)
    // {
    //     for(int y = 0; y < mesh_[1]; y++)
    //     {
    //         for(int x = 0; x < (mesh_[0]); x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    postForwardTransform<<<mesh_[1]*mesh_[2], 32>>>(data, r_data_, sinX_, cosX_, mesh_[2], mesh_[1], mesh_[0]);
    normalizationFFT<<<mesh_[1]*mesh_[2],32>>>(data, 2.0/double(mesh_[0]), mesh_[0], mesh_[1], mesh_[2]);
    // 
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, data, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < mesh_[2]; z++)
    // {
    //     for(int y = 0; y < mesh_[1]; y++)
    //     {
    //         for(int x = 0; x < (mesh_[0]); x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }exit(1);
    cudaFree(c_data_);

    // Transpose data in z and x direction
    permutation[0] = 2;
    permutation[1] = 1;
    permutation[2] = 0;  
    cut_transpose3d( r_data_,
                     data,
                     perm_mesh,
                     permutation,
                     1);
    data = r_data_;
    //
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, data, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < mesh_[2]; z++)
    // {
    //     for(int y = 0; y < mesh_[1]; y++)
    //     {
    //         for(int x = 0; x < (mesh_[0]); x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // exit(1);
    cudaFree(r_data_);
}

void FCT::inverseTransform(cufftDoubleReal * data)
{
    int permutation[3];
    int perm_mesh[3];
    // z
    cudaMalloc((void**)&c_data_, 
                (mesh_[2]/2+1) * mesh_[1] * mesh_[0] * sizeof(cufftDoubleComplex));
    cudaMalloc((void**)&r_data_, 
                mesh_[2] * mesh_[1] * mesh_[0] * sizeof(cufftDoubleReal));
    
    //    
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, data, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < mesh_[2]; z++)
    // {
    //     for(int y = 0; y < mesh_[1]; y++)
    //     {
    //         for(int x = 0; x < (mesh_[0]); x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    preInverseTransform<<<mesh_[0]*mesh_[1], 32>>>(data, r_data_, sinZ_, cosZ_, mesh_[0], mesh_[1], mesh_[2]);  
    // 
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, r_data_, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < mesh_[2]; z++)
    // {
    //     for(int y = 0; y < mesh_[1]; y++)
    //     {
    //         for(int x = 0; x < (mesh_[0]); x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // exit(1);
    // normalizationFFT<<<32, 32>>>(data, 1.0/double(mesh_[2]), mesh_[0], mesh_[1], mesh_[2]);
    if (cufftExecD2Z(iPlanZ_, r_data_, c_data_)!= CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: ExecD2Z in Z Forward failed\n");
        exit(0);
    } 
    // 
    // cufftDoubleComplex *c_data_c;
    // c_data_c = new cufftDoubleComplex [(mesh_[2]/2+1) * mesh_[1] * mesh_[0] * sizeof(cufftDoubleComplex)];
    // cudaMemcpy(c_data_c, c_data_, (mesh_[2]/2+1) * mesh_[1] * mesh_[0] * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < (mesh_[2]/2+1); z++)
    // {
    //     for(int y = 0; y < mesh_[1]; y++)
    //     {
    //         for(int x= 0; x < mesh_[0]; x++)
    //         {
    //             std::cout << "(" << c_data_c[z + (mesh_[2]/2+1)*y + mesh_[1]*(mesh_[2]/2+1)*x].x << ", "
    //                              << c_data_c[z + (mesh_[2]/2+1)*y + mesh_[1]*(mesh_[2]/2+1)*x].y << ")  ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // exit(1);
    postInverseTransform<<<mesh_[0]*mesh_[1], 32>>>(data, c_data_, mesh_[0], mesh_[1], mesh_[2]/2);
    // normalizationFFT<<<32,32>>>(data, 1.0/double(2*mesh_[0]), mesh_[0], mesh_[1], mesh_[2]);
    // 
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, data, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < mesh_[2]; z++)
    // {
    //     for(int y = 0; y < mesh_[1]; y++)
    //     {
    //         for(int x = 0; x < (mesh_[0]); x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // exit(1);
    cudaFree(c_data_);
    cudaFree(r_data_);
    permutation[0] = 1;
    permutation[1] = 0;
    permutation[2] = 2; 
    perm_mesh[0] = mesh_[0];
    perm_mesh[1] = mesh_[1];
    perm_mesh[2] = mesh_[2];
    cut_transpose3d( r_data_,
                     data,
                     perm_mesh,
                     permutation,
                     1);
    data = r_data_;
    perm_mesh[0] = mesh_[1];
    perm_mesh[1] = mesh_[0];
    perm_mesh[2] = mesh_[2];
    // cudaMemcpy(r_data_c, data, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < mesh_[2]; z++)
    // {
    //     for(int y = 0; y < mesh_[1]; y++)
    //     {
    //         for(int x = 0; x < (mesh_[0]); x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // y
    cudaMalloc((void**)&c_data_, 
                (mesh_[1]/2+1) * mesh_[0] * mesh_[2] * sizeof(cufftDoubleComplex));
    cudaMalloc((void**)&r_data_, 
                mesh_[1] * mesh_[0] * mesh_[2] * sizeof(cufftDoubleReal));
    preInverseTransform<<<mesh_[0]*mesh_[2], 32>>>(data, r_data_, sinY_, cosY_, mesh_[0], mesh_[2], mesh_[1]);  
    //
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, r_data_, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < mesh_[2]; z++)
    // {
    //     for(int y = 0; y < mesh_[1]; y++)
    //     {
    //         for(int x = 0; x < (mesh_[0]); x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // exit(1);
    // normalizationFFT<<<32, 32>>>(data, 1.0/double(mesh_[1]), mesh_[0], mesh_[1], mesh_[2]);
    if (cufftExecD2Z(iPlanY_, r_data_, c_data_)!= CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: ExecD2Z in Y Forward failed\n");
        exit(0);
    } 
    // 
    // cufftDoubleComplex *c_data_c;
    // c_data_c = new cufftDoubleComplex [(mesh_[2]/2+1) * mesh_[1] * mesh_[0] * sizeof(cufftDoubleComplex)];
    // cudaMemcpy(c_data_c, c_data_, (mesh_[2]/2+1) * mesh_[1] * mesh_[0] * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < (mesh_[2]/2+1); z++)
    // {
    //     for(int y = 0; y < mesh_[1]; y++)
    //     {
    //         for(int x= 0; x < mesh_[0]; x++)
    //         {
    //             std::cout << "(" << c_data_c[z + (mesh_[2]/2+1)*y + mesh_[1]*(mesh_[2]/2+1)*x].x << ", "
    //                              << c_data_c[z + (mesh_[2]/2+1)*y + mesh_[1]*(mesh_[2]/2+1)*x].y << ")  ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    postInverseTransform<<<mesh_[0]*mesh_[2], 32>>>(data, c_data_, mesh_[0], mesh_[2], mesh_[1]/2);
    // normalizationFFT<<<32,32>>>(data, 1.0/double(2*mesh_[0]), mesh_[2], mesh_[0], mesh_[1]);
    // 
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, data, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < mesh_[2]; z++)
    // {
    //     for(int y = 0; y < mesh_[1]; y++)
    //     {
    //         for(int x = 0; x < (mesh_[0]); x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    cudaFree(c_data_);
    cudaFree(r_data_);
    // Transpose data in z and y direction
    permutation[0] = 1;
    permutation[1] = 0;
    permutation[2] = 2;  
    cut_transpose3d( r_data_,
                     data,
                     perm_mesh,
                     permutation,
                     1);
    data = r_data_;
    perm_mesh[0] = mesh_[1];
    perm_mesh[1] = mesh_[0];
    perm_mesh[2] = mesh_[2];
    // x
    // Transpose data in x and z direction
    permutation[0] = 2;
    permutation[1] = 1;
    permutation[2] = 0;  
    cut_transpose3d( r_data_,
                     data,
                     perm_mesh,
                     permutation,
                     1);
    data = r_data_;
    perm_mesh[0] = mesh_[2];
    perm_mesh[1] = mesh_[1];
    perm_mesh[2] = mesh_[0];
    cudaMalloc((void**)&c_data_, 
                (mesh_[0]/2+1) * mesh_[1] * mesh_[2] * sizeof(cufftDoubleComplex));
    cudaMalloc((void**)&r_data_, 
                mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal));
    // normalizationInverse<<<32,32>>>(data, 0.5, mesh_[2], mesh_[1], mesh_[0]);
    preInverseTransform<<<mesh_[2]*mesh_[1], 32>>>(data, r_data_, sinX_, cosX_, mesh_[2], mesh_[1], mesh_[0]);  
    if (cufftExecD2Z(iPlanX_, r_data_, c_data_)!= CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: ExecD2Z in X Forward failed\n");
        exit(0);
    } 
    postInverseTransform<<<mesh_[2]*mesh_[1], 32>>>(data, c_data_, mesh_[2], mesh_[1], mesh_[0]/2);
    cudaFree(c_data_);
    cudaFree(r_data_);
    permutation[0] = 2;
    permutation[1] = 1;
    permutation[2] = 0;  
    cut_transpose3d( r_data_,
                     data,
                     perm_mesh,
                     permutation,
                     1);
    data = r_data_;
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
                                           );
        }
        else
        {
            data[Nz*Ny*x + Nz*y] = rdata[Nz*Ny*x + Nz*y];
        }
    }
}

static
__global__
void preInverseTransform(cufftDoubleReal  *data,
                         cufftDoubleReal  *r_data,
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
            r_data[z + Nz*y + Ny*Nz*x] = (data[z + Nz*y + Ny*Nz*x] + data[(Nz-z) + Nz*y + Ny*Nz*x])*sin_[z]
                                        +(data[z + Nz*y + Ny*Nz*x] - data[(Nz-z) + Nz*y + Ny*Nz*x])*cos_[z];
        else 
            r_data[Nz*y + Ny*Nz*x] = data[Nz*y + Ny*Nz*x];
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