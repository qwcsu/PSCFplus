#ifndef GPU_RESOURCES_H
#define GPU_RESOURCES_H
// #define double float
#include <cufft.h>

extern int THREADS_PER_BLOCK;
extern int NUMBER_OF_BLOCKS;
// #define SINGLE_PRECISION
#define DOUBLE_PRECISION


#ifdef SINGLE_PRECISION
typedef cufftReal cudaReal;
typedef cufftComplex cudaComplex;
typedef cufftReal hostReal;
typedef cufftComplex hostComplex;
#else
#ifdef DOUBLE_PRECISION
typedef cufftDoubleReal cudaReal;
typedef cufftDoubleComplex cudaComplex;
typedef cufftDoubleReal hostReal;
typedef cufftDoubleComplex hostComplex;
#endif
#endif


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if(abort) exit(code);
    }
}

static __global__ void helmholtzHelper(cudaReal* result, const cudaReal* composition,
   const cudaReal* pressure, double chi, int size) 
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads) 
    {
        result[i] = composition[i] * composition[i] / chi - pressure[i];
    }
}

static __global__ void reformField(cudaReal* Wa, cudaReal* Wb,
                    const cudaReal* pressureF,const cudaReal* compositionF, int size) 
    {
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads) 
    {
        Wa[i] = pressureF[i] + compositionF[i];
        Wb[i] = pressureF[i] - compositionF[i];
    }
}

static __global__ void mcftsStepHelper(cudaReal* result, const cudaReal* A2, const cudaReal* sqrtSq, int size) {
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads) 
    {
         result[i] += A2[i] * sqrtSq[i];
    }
}
static __global__ void mcftsScale(cudaReal* result, cudaReal scale, int size) {
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads) 
    {
         result[i] = result[i] * 2 * scale - scale;
    }
}

static __global__ void pointWiseAdd(cudaReal* result, const cudaReal* rhs, int size) {
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads) 
    {
         result[i] += rhs[i];
    }
}

static __global__ void subtractUniform(cudaReal* result, cudaReal rhs, int size) {
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads) 
    {
        result[i] -= rhs;
    }
}

static __global__ void pointWiseSubtract(cudaReal* result, const cudaReal* rhs, int size)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads) 
    {
        result[i] -= rhs[i];
    }
}

static __global__ void pointWiseSubtractFloat(cudaReal* result, const double rhs, int size) {
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads) 
    {
        result[i] -= rhs;
    }   
}

static __global__ void pointWiseBinarySubtract(const cudaReal* a, const cudaReal* b, cudaReal* result, int size) {
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads) 
    {
         result[i] = a[i] - b[i];
    }
}

static __global__ void pointWiseBinaryAdd(const cudaReal* a, const cudaReal* b, cudaReal* result, int size) {
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads) 
    {
        result[i] = a[i] + b[i];
    }
}

static __global__ void pointWiseAddScale(cudaReal* result, const cudaReal* rhs, double scale, int size) 
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads) 
    {
        result[i] += scale * rhs[i];
    }
}

static __global__ void pointWiseScale(cudaReal* result, double scale, int size) 
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads) 
    {
        result[i] *= scale;
    }
}

//the 1 is a placeholder for dr
static __global__ void AmIsConvergedHelper(cudaReal* out, int size) 
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    cudaReal temp;
    for (int i = startID; i < size; i += nThreads) 
    {
        temp = (out[i] - 1) * (out[i] - 1) * 1;
        out[i] = temp;
    }
}

static __global__ void AmHelper(cudaReal* out, cudaReal* present, cudaReal* iPast, cudaReal* jPast, int size) {
   int nThreads = blockDim.x * gridDim.x;
   int startID = blockIdx.x * blockDim.x + threadIdx.x;
   for (int i = startID; i < size; i += nThreads) {
      out[i] += (present[i] - iPast[i]) * (present[i] - jPast[i]);
   }
}

static __global__ void AmHelperVm(cudaReal* out, cudaReal* present, cudaReal* iPast, int size) 
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = startID; i < size; i += nThreads) 
    {
        out[i] += (present[i] - iPast[i]) * (present[i]);
    }
}

static __global__ void reduction(cudaReal* c, cudaReal* a, int size) 
{
    //int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * (blockDim.x*2) + threadIdx.x;
//    printf("a[%2d] = %6.4f\n", startID, a[startID]);
//    printf("a[%2d] = %6.4f\n", startID+ blockDim.x, a[startID+ blockDim.x]);
    volatile extern __shared__ cudaReal cache[];
    cache[threadIdx.x] = 0.0;
    cudaReal temp1 = 0.0;
    cudaReal temp2 = 0.0;
    for (int i = startID; i < size; i += 2*blockDim.x*gridDim.x)
    {
        temp1 += a[startID];
        temp2 += a[startID + blockDim.x];
//        printf("temp2 = %12.10e\n", temp1);
    }

    /*cache[threadIdx.x] = 0;
    if (startID < size)
    {

    }*/

    cache[threadIdx.x] = temp1 + temp2;
//    printf("startID = %2d cache[%2d] = %6.4f\n", startID, threadIdx.x, cache[threadIdx.x]);
    __syncthreads();

    //reduce operation
    //256/2 -- needs to be power of two
    for (int i = blockDim.x/2; i != 0; i/=2)
    {
        if(threadIdx.x < i)
            cache[threadIdx.x] += cache[threadIdx.x + i];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        c[blockIdx.x] = cache[0];
//        printf("c[%2d] = %12.10e\n", blockIdx.x, c[blockIdx.x]);
    }
}

//dpdpg
static __global__
void deviceRI4(cudaReal *f, int size,
               cudaReal *I0,
               cudaReal *I1,
               cudaReal *I2,
               cudaReal *I3,
               cudaReal *I4)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    __shared__ cudaReal cache0[32];
    __shared__ cudaReal cache1[32];
    __shared__ cudaReal cache2[32];
    __shared__ cudaReal cache3[32];
    __shared__ cudaReal cache4[32];
    cudaReal temp0 = 0.0;
    cudaReal temp1 = 0.0;
    cudaReal temp2 = 0.0;
    cudaReal temp3 = 0.0;
    cudaReal temp4 = 0.0;

    for(int i = startID; i < size; i += nThreads) 
    {
        temp0 += f[i];  
        if (i/2*2 == i)
            temp1 += f[i]; 
        if (i/4*4 == i)
            temp2 += f[i]; 
        if (i/8*8 == i)
            temp3 += f[i]; 
        if (i/16*16 == i)
            temp4 += f[i]; 
    } 

    cache0[cacheIndex] = temp0;
    cache1[cacheIndex] = temp1;
    cache2[cacheIndex] = temp2;
    cache3[cacheIndex] = temp3;
    cache4[cacheIndex] = temp4;

    __syncthreads();

    int i = blockDim.x/2;
    while (i != 0)
    {
        if (cacheIndex < i)
        {
            cache0[cacheIndex] += cache0[cacheIndex + i];
            cache1[cacheIndex] += cache1[cacheIndex + i];
            cache2[cacheIndex] += cache2[cacheIndex + i];
            cache3[cacheIndex] += cache3[cacheIndex + i];
            cache4[cacheIndex] += cache4[cacheIndex + i];
        }
            
        __syncthreads();
        i /= 2;
    }
    if (cacheIndex == 0)
    {
        I0[blockIdx.x] = cache0[0];
        I1[blockIdx.x] = cache1[0];
        I2[blockIdx.x] = cache2[0];
        I3[blockIdx.x] = cache3[0];
        I4[blockIdx.x] = cache4[0];
    }

}

// crystallographic fft
static __global__
void device_RI4(cudaReal *f, int size,
               cudaReal *I0,
               cudaReal *I1,
               cudaReal *I2,
               cudaReal *I3,
               cudaReal *I4)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    __shared__ cudaReal cache0[32];
    __shared__ cudaReal cache1[32];
    __shared__ cudaReal cache2[32];
    __shared__ cudaReal cache3[32];
    __shared__ cudaReal cache4[32];
    cudaReal temp0 = 0.0;
    cudaReal temp1 = 0.0;
    cudaReal temp2 = 0.0;
    cudaReal temp3 = 0.0;
    cudaReal temp4 = 0.0;

    for(int i = startID; i < size; i += nThreads) 
    {
        if (i == 0)
        {
            temp0 += (f[0])/2; 
            temp1 += (f[0])/2; 
            temp2 += (f[0])/2; 
            temp3 += (f[0])/2; 
            temp4 += (f[0])/2; 
        }
        else{
            temp0 += f[i];  
            if (i/2*2 == i)
                temp1 += f[i]; 
            if (i/4*4 == i)
                temp2 += f[i]; 
            if (i/8*8 == i)
                temp3 += f[i]; 
            if (i/16*16 == i)
                temp4 += f[i]; 
        }
        
    } 

    cache0[cacheIndex] = temp0;
    cache1[cacheIndex] = temp1;
    cache2[cacheIndex] = temp2;
    cache3[cacheIndex] = temp3;
    cache4[cacheIndex] = temp4;

    __syncthreads();

    int i = blockDim.x/2;
    while (i != 0)
    {
        if (cacheIndex < i)
        {
            cache0[cacheIndex] += cache0[cacheIndex + i];
            cache1[cacheIndex] += cache1[cacheIndex + i];
            cache2[cacheIndex] += cache2[cacheIndex + i];
            cache3[cacheIndex] += cache3[cacheIndex + i];
            cache4[cacheIndex] += cache4[cacheIndex + i];
        }
            
        __syncthreads();
        i /= 2;
    }
    if (cacheIndex == 0)
    {
        I0[blockIdx.x] = cache0[0];
        I1[blockIdx.x] = cache1[0];
        I2[blockIdx.x] = cache2[0];
        I3[blockIdx.x] = cache3[0];
        I4[blockIdx.x] = cache4[0];
    }

}

static __global__
void deviceRI3(cudaReal *f, int size,
               cudaReal *I0,
               cudaReal *I1,
               cudaReal *I2,
               cudaReal *I3)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    __shared__ cudaReal cache0[32];
    __shared__ cudaReal cache1[32];
    __shared__ cudaReal cache2[32];
    __shared__ cudaReal cache3[32];
    cudaReal temp0 = 0.0;
    cudaReal temp1 = 0.0;
    cudaReal temp2 = 0.0;
    cudaReal temp3 = 0.0;

    for(int i = startID; i < size; i += nThreads) 
    {
        temp0 += f[i];  
        if (i/2*2 == i)
            temp1 += f[i]; 
        if (i/4*4 == i)
            temp2 += f[i]; 
        if (i/8*8 == i)
            temp3 += f[i]; 
    } 

    cache0[cacheIndex] = temp0;
    cache1[cacheIndex] = temp1;
    cache2[cacheIndex] = temp2;
    cache3[cacheIndex] = temp3;

    __syncthreads();

    int i = blockDim.x/2;
    while (i != 0)
    {
        if (cacheIndex < i)
        {
            cache0[cacheIndex] += cache0[cacheIndex + i];
            cache1[cacheIndex] += cache1[cacheIndex + i];
            cache2[cacheIndex] += cache2[cacheIndex + i];
            cache3[cacheIndex] += cache3[cacheIndex + i];
        }
            
        __syncthreads();
        i /= 2;
    }
    if (cacheIndex == 0)
    {
        I0[blockIdx.x] = cache0[0];
        I1[blockIdx.x] = cache1[0];
        I2[blockIdx.x] = cache2[0];
        I3[blockIdx.x] = cache3[0];
    }

}

static __global__
void deviceRI2(cudaReal *f, int size,
               cudaReal *I0,
               cudaReal *I1,
               cudaReal *I2)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    __shared__ cudaReal cache0[32];
    __shared__ cudaReal cache1[32];
    __shared__ cudaReal cache2[32];

    cudaReal temp0 = 0.0;
    cudaReal temp1 = 0.0;
    cudaReal temp2 = 0.0;

    for(int i = startID; i < size; i += nThreads) 
    {
        temp0 += f[i];  
        if (i/2*2 == i)
            temp1 += f[i]; 
        if (i/4*4 == i)
            temp2 += f[i]; 
    } 

    cache0[cacheIndex] = temp0;
    cache1[cacheIndex] = temp1;
    cache2[cacheIndex] = temp2;

    __syncthreads();

    int i = blockDim.x/2;
    while (i != 0)
    {
        if (cacheIndex < i)
        {
            cache0[cacheIndex] += cache0[cacheIndex + i];
            cache1[cacheIndex] += cache1[cacheIndex + i];
            cache2[cacheIndex] += cache2[cacheIndex + i];
        }
            
        __syncthreads();
        i /= 2;
    }
    if (cacheIndex == 0)
    {
        I0[blockIdx.x] = cache0[0];
        I1[blockIdx.x] = cache1[0];
        I2[blockIdx.x] = cache2[0];
    }

}


static __global__
void deviceRI1(cudaReal *f, int size,
               cudaReal *I0,
               cudaReal *I1)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    __shared__ cudaReal cache0[32];
    __shared__ cudaReal cache1[32];

    cudaReal temp0 = 0.0;
    cudaReal temp1 = 0.0;

    for(int i = startID; i < size; i += nThreads) 
    {
        temp0 += f[i];  
        if (i/2*2 == i)
            temp1 += f[i]; 
    } 

    cache0[cacheIndex] = temp0;
    cache1[cacheIndex] = temp1;

    __syncthreads();

    int i = blockDim.x/2;
    while (i != 0)
    {
        if (cacheIndex < i)
        {
            cache0[cacheIndex] += cache0[cacheIndex + i];
            cache1[cacheIndex] += cache1[cacheIndex + i];
        }
            
        __syncthreads();
        i /= 2;
    }
    if (cacheIndex == 0)
    {
        I0[blockIdx.x] = cache0[0];
        I1[blockIdx.x] = cache1[0];
    }

}

static __global__
void deviceRI0(cudaReal *f, int size,
               cudaReal *I0)
{
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    __shared__ cudaReal cache0[32];

    cudaReal temp0 = 0.0;

    for(int i = startID; i < size; i += nThreads) 
    {
        temp0 += f[i];  
    } 

    cache0[cacheIndex] = temp0;

    __syncthreads();

    int i = blockDim.x/2;
    while (i != 0)
    {
        if (cacheIndex < i)
        {
            cache0[cacheIndex] += cache0[cacheIndex + i];
        }
            
        __syncthreads();
        i /= 2;
    }
    if (cacheIndex == 0)
    {
        I0[blockIdx.x] = cache0[0];
    }

}

static __global__ void setUniformReal(cudaReal* result, cudaReal uniform, int size) {
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = startID; i < size; i += nThreads) {
        result[i] = uniform;
        // printf("uniform = %f\n", uniform);
    }
}

static __global__ void normalization(cudaReal* result, cudaReal* a, cudaReal* b, int size) {
    int nThreads = blockDim.x * gridDim.x;
    int startID = blockIdx.x * blockDim.x + threadIdx.x;
    
    for(int i = startID; i < size; i += nThreads) {
        cudaReal tmp;
        tmp = a[i] + b[i];
        result[i] = a[i]/tmp;
        // printf("uniform = %f\n", uniform);
    }
}


#endif
