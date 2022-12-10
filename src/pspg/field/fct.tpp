#ifndef FCT_TPP
#define FCT_TPP

#include "fct.h"

template <int D>
FCT<D>::FCT()
{

}

template <int D>
FCT<D>::~FCT()
{
    cudaFree(sinX_);
    cudaFree(sinY_);
    cudaFree(sinZ_);
    cudaFree(cosX_);
    cudaFree(cosY_);
    cudaFree(cosZ_);
    cudaFree(r_data_);
}

template <int D>
void FCT<D>::setup(int * mesh)
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

    cudaMalloc((void**)&r_data_, 
                mesh_[2] * mesh_[1] * mesh_[0] * sizeof(cufftDoubleReal));


    makePlans(mesh_);

    isSetup_ = true;
}

template <int D>
void FCT<D>::makePlans(int * mesh)
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

template <int D>
void FCT<D>::forwardTransform(cufftDoubleReal * data)
{
    int permutation[3];
    int perm_mesh[3];
    // z
    cudaMalloc((void**)&c_data_, 
                (mesh_[2]/2+1) * mesh_[1] * mesh_[0] * sizeof(cufftDoubleComplex));
    
    // 
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, data, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < 3; z++)
    // {
    //     for(int y = 0; y < 3; y++)
    //     {
    //         for(int x = 0; x < 3; x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // exit(1);
    preForwardTransform<<<mesh_[0]*mesh_[1], 32>>>(c_data_, data, mesh_[0], mesh_[1], mesh_[2]/2);    
    //
    // cufftDoubleComplex *c_data_c;
    // c_data_c = new cufftDoubleComplex [(mesh_[2]/2+1) * mesh_[1] * mesh_[0] * sizeof(cufftDoubleComplex)];
    // cudaMemcpy(c_data_c, c_data_, (mesh_[2]/2+1) * mesh_[1] * mesh_[0] * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < 3; z++)
    // {
    //     for(int y = 0; y < 2; y++)
    //     {
    //         for(int x= 0; x < 2; x++)
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
    // for(int z = 0; z < 2; z++)
    // {
    //     for(int y = 0; y < 2; y++)
    //     {
    //         for(int x = 0; x < 2; x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // exit(1);
    postForwardTransform<<<mesh_[0]*mesh_[1], 32>>>(data, r_data_, sinZ_, cosZ_, mesh_[0], mesh_[1], mesh_[2]);
    // normalizationFFT<<<mesh_[0]*mesh_[1],32>>>(data, 2.0/double(mesh_[2]), mesh_[0], mesh_[1], mesh_[2]);
    // 
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, data, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < 2; z++)
    // {
    //     for(int y = 0; y < 32; y++)
    //     {
    //         for(int x = 0; x < 4; x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // exit(1);
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
    cudaFree(c_data_);
    // y
    cudaMalloc((void**)&c_data_, 
                (mesh_[1]/2+1) * mesh_[0] * mesh_[2] * sizeof(cufftDoubleComplex));
    preForwardTransform<<<mesh_[0]*mesh_[2], 32>>>(c_data_, r_data_, mesh_[0], mesh_[2], mesh_[1]/2);
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
    // normalizationFFT<<<mesh_[0]*mesh_[2],32>>>(data, 2.0/double(mesh_[1]), mesh_[0], mesh_[1], mesh_[2]);
    // 
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, data, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < 2; z++)
    // {
    //     for(int y = 0; y < 32; y++)
    //     {
    //         for(int x = 0; x < 4; x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // exit(1);
    // Transpose data in z and y direction
    permutation[0] = 1;
    permutation[1] = 0;
    permutation[2] = 2;  
    cut_transpose3d( r_data_,
                     data,
                     perm_mesh,
                     permutation,
                     1);

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
    cudaFree(c_data_);
    // x
    // Transpose data in x and z direction
    permutation[0] = 2;
    permutation[1] = 1;
    permutation[2] = 0;  
    cut_transpose3d( data,
                     r_data_,
                     perm_mesh,
                     permutation,
                     1);
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
    cudaMalloc((void**)&c_data_, 
                (mesh_[0]/2+1) * mesh_[1] * mesh_[2] * sizeof(cufftDoubleComplex)); 
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
    if (cufftExecZ2D(fPlanX_, c_data_, data)!= CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: ExecZ2D Forward failed");
        exit(0);
    }
    
    postForwardTransform<<<mesh_[1]*mesh_[2], 32>>>(r_data_, data, sinX_, cosX_, mesh_[2], mesh_[1], mesh_[0]);
    // normalizationFFT<<<mesh_[1]*mesh_[2],32>>>(r_data_, 2.0/double(mesh_[0]), mesh_[0], mesh_[1], mesh_[2]);
    //
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, data, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < 2; z++)
    // {
    //     for(int y = 0; y < 32; y++)
    //     {
    //         for(int x = 0; x < 4; x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // exit(1);
    cudaFree(c_data_);

    // Transpose data in z and x direction
    permutation[0] = 2;
    permutation[1] = 1;
    permutation[2] = 0;  
    cut_transpose3d( data,
                     r_data_,
                     perm_mesh,
                     permutation,
                     1);
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
}

template <int D>
void FCT<D>::inverseTransform(cufftDoubleReal * data)
{
    int permutation[3];
    int perm_mesh[3];
    // z
    cudaMalloc((void**)&c_data_, 
                (mesh_[2]/2+1) * mesh_[1] * mesh_[0] * sizeof(cufftDoubleComplex));
    
    //    
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, data, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < 3; z++)
    // {
    //     for(int y = 0; y < 3; y++)
    //     {
    //         for(int x = 0; x < 3; x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // exit(1);
    preInverseTransform<<<mesh_[0]*mesh_[1], 32>>>(r_data_, data, sinZ_, cosZ_, mesh_[0], mesh_[1], mesh_[2]);  
    // 
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, r_data_, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < 32; z++)
    // {
    //     for(int y = 0; y < 32; y++)
    //     {
    //         for(int x = 0; x < 3; x++)
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
    // for(int z = 0; z < 2; z++)
    // {
    //     for(int y = 0; y < 3; y++)
    //     {
    //         for(int x= 0; x < 3; x++)
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
    //  
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, data, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < 3; z++)
    // {
    //     for(int y = 0; y < 3; y++)
    //     {
    //         for(int x = 0; x < 3; x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // exit(1);
    cudaFree(c_data_);
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

    perm_mesh[0] = mesh_[1];
    perm_mesh[1] = mesh_[0];
    perm_mesh[2] = mesh_[2];
    //
    // cudaMemcpy(r_data_c, data, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < 3; z++)
    // {
    //     for(int y = 0; y < 3; y++)
    //     {
    //         for(int x = 0; x < 3; x++)
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
    preInverseTransform<<<mesh_[0]*mesh_[2], 32>>>(data, r_data_, sinY_, cosY_, mesh_[0], mesh_[2], mesh_[1]);  
    //
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, r_data_, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < 3; z++)
    // {
    //     for(int y = 0; y < 3; y++)
    //     {
    //         for(int x = 0; x < 3; x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    // exit(1);
    // normalizationFFT<<<32, 32>>>(data, 1.0/double(mesh_[1]), mesh_[0], mesh_[1], mesh_[2]);
    if (cufftExecD2Z(iPlanY_, data, c_data_)!= CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: ExecD2Z in Y Forward failed\n");
        exit(0);
    } 
    // 
    // cufftDoubleComplex *c_data_c;
    // c_data_c = new cufftDoubleComplex [(mesh_[2]/2+1) * mesh_[1] * mesh_[0] * sizeof(cufftDoubleComplex)];
    // cudaMemcpy(c_data_c, c_data_, (mesh_[2]/2+1) * mesh_[1] * mesh_[0] * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < 2; z++)
    // {
    //     for(int y = 0; y < 3; y++)
    //     {
    //         for(int x= 0; x < 3; x++)
    //         {
    //             std::cout << "(" << c_data_c[z + (mesh_[2]/2+1)*y + mesh_[1]*(mesh_[2]/2+1)*x].x << ", "
    //                              << c_data_c[z + (mesh_[2]/2+1)*y + mesh_[1]*(mesh_[2]/2+1)*x].y << ")  ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
    postInverseTransform<<<mesh_[0]*mesh_[2], 32>>>(data, c_data_, mesh_[0], mesh_[2], mesh_[1]/2);
    // 
    // 
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, data, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < 3; z++)
    // {
    //     for(int y = 0; y < 3; y++)
    //     {
    //         for(int x = 0; x < 3; x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
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
    perm_mesh[0] = mesh_[1];
    perm_mesh[1] = mesh_[0];
    perm_mesh[2] = mesh_[2];
    // x
    // Transpose data in x and z direction
    permutation[0] = 2;
    permutation[1] = 1;
    permutation[2] = 0;  
    cut_transpose3d( data,
                     r_data_,
                     perm_mesh,
                     permutation,
                     1);
    perm_mesh[0] = mesh_[2];
    perm_mesh[1] = mesh_[1];
    perm_mesh[2] = mesh_[0];
    cudaMalloc((void**)&c_data_, 
                (mesh_[0]/2+1) * mesh_[1] * mesh_[2] * sizeof(cufftDoubleComplex));
    preInverseTransform<<<mesh_[2]*mesh_[1], 32>>>(r_data_, data, sinX_, cosX_, mesh_[2], mesh_[1], mesh_[0]);  
    if (cufftExecD2Z(iPlanX_, r_data_, c_data_)!= CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: ExecD2Z in X Forward failed\n");
        exit(0);
    } 
    postInverseTransform<<<mesh_[2]*mesh_[1], 32>>>(r_data_, c_data_, mesh_[2], mesh_[1], mesh_[0]/2);
    cudaFree(c_data_);
    permutation[0] = 2;
    permutation[1] = 1;
    permutation[2] = 0;  
    cut_transpose3d( data,
                     r_data_,
                     perm_mesh,
                     permutation,
                     1);
    // 
    // cufftDoubleReal *r_data_c;
    // r_data_c = new cufftDoubleReal [mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal)];
    // cudaMemcpy(r_data_c, data, mesh_[0] * mesh_[1] * mesh_[2] * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
    // for(int z = 0; z < 3; z++)
    // {
    //     for(int y = 0; y < 3; y++)
    //     {
    //         for(int x = 0; x < 3; x++)
    //         {
    //             std::cout << r_data_c[z + mesh_[2]*y + mesh_[1]*mesh_[2]*x] << "    ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }
}



#endif