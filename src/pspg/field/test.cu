#include "fct.h"

#include <iomanip>
#include<time.h>

int main()
{
    FCT<3> fct;
    int mesh[3];
    int Nx = 4,
        Ny = 4,
        Nz = 4;
    int size = Nx * Ny * Nz;
    mesh[0] = Nx;
    mesh[1] = Ny;
    mesh[2] = Nz;
    double *data_c, *data;
    data_c = new double[size];
    cudaMalloc((void**)&data, size * sizeof(double));

    srand((unsigned int)time(NULL));
    for(int i = 0; i < size; ++i)
        data_c[i] = rand()%10;
    for (int z = 0; z < Nz; z++)
    {
        for (int y = 0; y < Ny; y++)
        {
            for (int x = 0; x < Nx; x++)
            {
                std::cout << std::setw(8) << std::scientific
                << data_c[z + y*Nz + Nz*Ny*x] << "   ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "----------------------------------------------------------------" << std::endl;

    cudaMemcpy(data, data_c, size * sizeof(double), cudaMemcpyHostToDevice);

    fct.setup(mesh);
    fct.forwardTransform(data);

    cudaMemcpy(data_c, data, size * sizeof(double), cudaMemcpyDeviceToHost);

    for (int z = 0; z < Nz; z++)
    {
        for (int y = 0; y < Ny; y++)
        {
            for (int x = 0; x < Nx; x++)
            {
                std::cout << std::setw(8) << std::scientific
                << data_c[z + y*Nz + Nz*Ny*x] << "   ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "----------------------------------------------------------------" << std::endl;
    fct.inverseTransform(data);

    cudaMemcpy(data_c, data, size * sizeof(double), cudaMemcpyDeviceToHost);

    for (int z = 0; z < Nz; z++)
    {
        for (int y = 0; y < Ny; y++)
        {
            for (int x = 0; x < Nx; x++)
            {
                std::cout << std::setw(8) << std::scientific
                << data_c[z + y*Nz + Nz*Ny*x] << "   ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    free(data_c);
    cudaFree(data);

    return 0;
}