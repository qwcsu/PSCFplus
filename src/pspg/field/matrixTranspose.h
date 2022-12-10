#ifndef MATRIX_TRANS_H
#define MATRIX_TRANS_H
#include <cufft.h>
#include <iostream>

#define TILE_SIZE 16
#define BRICK_SIZE 8  

/********************************************
 * Private function prototypes              *
 ********************************************/
static void set_grid_dims( const int* size,
                           int        d2,
                           dim3*      block_size,
                           dim3*      num_blocks,
                           int        elements_per_thread);

static void set_grid_dims_cube( const int* size,
                                dim3*      block_size,
                                dim3*      num_blocks,
                                int        elements_per_thread );

static int valid_parameters( int        in_place,
                             const int* size,
                             const int* permutation,
                             int        elements_per_thread );
/********************************************
 * Device functions                         *
 ********************************************/
static
__global__
void transpose_zy( cufftDoubleReal*       out,
                   const cufftDoubleReal* in,
                   int                    np0,
                   int                    np1,
                   int                    np2 )
{
	__shared__ cufftDoubleReal tile[TILE_SIZE][TILE_SIZE + 1];

	int x, y, z,
	    ind_in,
	    ind_out;

	int lx = threadIdx.x,
	    ly = threadIdx.y,
	    bx = blockIdx.x,
	    by = blockIdx.y;

	x = lx + TILE_SIZE * bx;
	y = ly + TILE_SIZE * by;
	z = blockIdx.z;

	ind_in = x + (y + z * np1) * np0;
	ind_out = x + (z + y * np2) * np0;

	if( x < np0 && y < np1 )
	{
		tile[lx][ly] = in[ind_in];
	}

	__syncthreads();

	if( x < np0 && y < np1	 )
	{
		out[ind_out] = tile[lx][ly];
	}
}

static
__global__
void transpose_xz( cufftDoubleReal*       out,
                   const cufftDoubleReal* in,
                   int                    np0,
                   int                    np1,
                   int                    np2 )
{

	__shared__ cufftDoubleReal tile[TILE_SIZE][TILE_SIZE + 1];
	
	int x_in, y, z_in,
	    x_out, z_out,
	    ind_in,
	    ind_out;
	
	int lx = threadIdx.x,
	    ly = threadIdx.y,
	    bx = blockIdx.x,
	    by = blockIdx.y;

	x_in = lx + TILE_SIZE * bx;
	z_in = ly + TILE_SIZE * by;

	y = blockIdx.z;

	x_out = ly + TILE_SIZE * bx;
	z_out = lx + TILE_SIZE * by;


	ind_in = x_in + (y + z_in * np1) * np0;
	ind_out = z_out + (y + x_out * np1) * np2;

	if( x_in < np0 && z_in < np2 )
	{
			tile[lx][ly] = in[ind_in];
	}	

	__syncthreads();

	if( z_out < np2 && x_out < np0 )
	{
		out[ind_out] = tile[ly][lx];
	}

}

static
 __global__
 void transpose_yx( cufftDoubleReal*       out,
                    const cufftDoubleReal* in,
                    int           		   np0,
                    int           		   np1,
                    int                    np2 )
{
	__shared__ cufftDoubleReal tile[TILE_SIZE][TILE_SIZE + 1];

	int x_in, y_in, z,
	    x_out, y_out,
	    ind_in,
	    ind_out;

	int lx = threadIdx.x,
	    ly = threadIdx.y,
	    bx = blockIdx.x,
	    by = blockIdx.y;

	x_in = lx + TILE_SIZE * bx;
	y_in = ly + TILE_SIZE * by;

	z = blockIdx.z;

	x_out = ly + TILE_SIZE * bx;
	y_out = lx + TILE_SIZE * by;


	ind_in = x_in + (y_in + z * np1) * np0;
	ind_out = y_out + (x_out + z * np0) * np1;

	if( x_in < np0 && y_in < np1 )
	{
		tile[lx][ly] = in[ind_in];
	}

	__syncthreads();

	if( x_out < np0 && y_out < np1 )
	{
		out[ind_out] = tile[ly][lx];
	}
}



/********************************************
 * Exported functions                         *
 ********************************************/
static
int cut_transpose3d( cufftDoubleReal *out,
					 cufftDoubleReal *in,
                     const int*    size,
                     const int*    permutation,
                     int           elements_per_thread )

{
	
	dim3 num_blocks,
	     block_size;
	set_grid_dims( size,
		           permutation[0],
		           &block_size,
		           &num_blocks,
		           elements_per_thread );


	if( permutation[0] == 1 && permutation[1] == 0 && permutation[2] == 2 ){
		transpose_yx<<< num_blocks, block_size >>>(out,
												   in,
                  							       size[0],
												   size[1],
											       size[2]);
	}
	else if ( permutation[0] == 2 && permutation[1] == 1 && permutation[2] == 0 ){
		transpose_xz<<< num_blocks, block_size >>>(out,
												   in,
                  							       size[0],
												   size[1],
											       size[2]);
	}
	else if ( permutation[0] == 0 && permutation[1] == 2 && permutation[2] == 1 ){
		transpose_zy<<< num_blocks, block_size >>>(out,
												   in,
                  							       size[0],
												   size[1],
											       size[2]);
	}
	
	return 0;
}


/********************************************
 * Private functions                        *
 ********************************************/
static void set_grid_dims( const int* size,
                           int        d2,
                           dim3*      block_size,
                           dim3*      num_blocks,
                           int        elements_per_thread)
{
	block_size->x = TILE_SIZE;
	block_size->y = TILE_SIZE / elements_per_thread;
	block_size->z = 1;
	num_blocks->x = size[0] / TILE_SIZE;
	if( size[0] % TILE_SIZE != 0 )
		num_blocks->x++;
	if( d2 == 0 )
		d2 = 1;
	num_blocks->y = size[d2] / TILE_SIZE;
	if( size[d2] % TILE_SIZE != 0 )
		num_blocks->y++;
	num_blocks->z = size[(d2 == 1) ? 2 : 1];
}

static void set_grid_dims_cube( const int* size,
                                dim3*      block_size,
                                dim3*      num_blocks,
                                int        elements_per_thread )
{
	block_size->x = BRICK_SIZE;
	block_size->y = BRICK_SIZE;
	block_size->z = BRICK_SIZE / elements_per_thread;
	num_blocks->x = size[0] / BRICK_SIZE;
	if( size[0] % BRICK_SIZE != 0 )
		num_blocks->x++;
	num_blocks->y = size[1] / BRICK_SIZE;
	if( size[1] % BRICK_SIZE != 0 )
		num_blocks->y++;
	num_blocks->z = size[2] / BRICK_SIZE;
	if( size[2] % BRICK_SIZE != 0 )
		num_blocks->z++;
}

static int valid_parameters( int        in_place,
                             const int* size,
                             const int* permutation,
                             int        elements_per_thread )
{
	int dims[] = { 0, 0, 0 },
        i;

	if( in_place && elements_per_thread != 1 )
		return 0;
	if( size == NULL || permutation == NULL )
		return 0;
	if( size[0] < 2 || size[1] < 2 || size[2] < 2 )
		return 0;

	for( i = 0; i < 3; i++ )
	{
		if( permutation[i] < 0 || permutation[i] > 2 )
			return 0;
		else if( dims[permutation[i]] == 1 )
			return 0;
		else
			dims[permutation[i]] = 1;
	}

	return 1;
}

#endif