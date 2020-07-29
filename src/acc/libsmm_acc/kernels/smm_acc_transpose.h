/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

/*****************************************************************************
 *  Authors: Peter Messmer <pmessmer@nvidia.com>,                            *
 *           Nikolay Markovskiy <nmarkovskiy@nvidia.com>                     *
 *****************************************************************************/

#include "smm_acc_common.h"

/*
 * Execution configuration:
 * gridDim.x = number of matrix blocks in this batched matrix transpose
 *           = length of the batched transpose stack
 * blockIdx.x = {0, ..., gridDim.x-1}
 * blockDim.x = 128 or the smallest multiple of warp_size larger or equal to m*n
 * threadIdx.x = {0, ..., blockDim.x-1}

 * Execute batched matrix transpose in place

 * Template parameters
 * --- m, n: pair of integers characterising the dimensions of the matrix to transpose

 * Function arguments
 * --- trs_stack: transpose stack (pointer to global memory):
 *     array of stack entries (indices), indicating where each matrix to transpose starts in the "mat" array
 * --- mat (pointer to global memory):
 *     arrays containing the values of the matrix to be transposed
 *     mat is column major: m = number of rows, n = number of columns

 * Algorithm specificities:
 * - the temporary buffer (of size m * n * 8 bytes) in which matrix elements are stored has to fit entirely into shared memory. Therefore, this kernel cannot be run for mattrix sizes such that m * n * 8 bytes > available shared memory per block.
 */

#define TILE_DIM 16


template <int m, int n>
__global__ void transpose_d(int *trs_stack, double* mat){

#ifdef TR_OLD
 __shared__ double buf[m*n];

 /* Get the offset in the transpose-stack that this block ID should handle */
 int offset = trs_stack[blockIdx.x];

 /* Loop over m*n matrix elements */
 for(int i=threadIdx.x; i < m*n; i+=blockDim.x){
     /* Load matrix elements into a temporary buffer */
     buf[i] = mat[offset + i];
 }
 syncthreads();

 /* Loop over elements of the matrix to be overwritten */
 for(int i=threadIdx.x; i < m*n; i+=blockDim.x){
     /* Compute old row and column index of matrix element */
     int r_out = i % n;
     int c_out = i / n;
     /* Compute the corresponding old 1D index of matrix element */
     int idx = r_out * m + c_out;
     /* Overwrite the matrix element */
     mat[offset + i] = buf[idx];
 }

#else

 __shared__ double buf[TILE_DIM][TILE_DIM];

 /* Get the offset in the transpose-stack that this block ID should handle */
 int num_tiles_row = (m + TILE_DIM - 1) / TILE_DIM;
 int num_tiles_col = (n + TILE_DIM - 1) / TILE_DIM;
 int num_tiles = num_tiles_row * num_tiles_col;
 int trs_stack_offset = trs_stack[blockIdx.x / num_tiles];

 /* Get indices in the matrix */
 int block_id_local = blockIdx.x % num_tiles;
 int block_id_local_row = block_id_local % num_tiles_row;
 int block_id_local_col = block_id_local / num_tiles_row;
 int i = threadIdx.x;
 int irow_tile = threadIdx.x % TILE_DIM;
 int icol_tile = threadIdx.x / TILE_DIM;
 int irow_mat = block_id_local_row * TILE_DIM + irow_tile;
 int icol_mat = block_id_local_col * TILE_DIM + icol_tile;

 /* Loop over the elements in this matrix tile */
 if((irow_mat < m) && (icol_mat < n)){
     /* Convert to 2D index */
     int mat_idx = icol_mat * m + irow_mat;
     /* Load matrix elements into a temporary buffer */
     buf[irow_tile][icol_tile] = mat[trs_stack_offset + mat_idx];
//     if(icol_mat == 0 or icol_mat == 60 or icol_mat == 80 or icol_mat == 99 or irow_mat == 0 or irow_mat == 60 or irow_mat == 80 or irow_mat == 99){
       printf("[t=%i,b=%i,offset=%i]{%g} block_id_local = (%ix%i=%i), itile = (%ix%i) <-- imat = (%ix%i)%c", threadIdx.x, blockIdx.x, trs_stack_offset, mat[trs_stack_offset + mat_idx], block_id_local_row, block_id_local_col, block_id_local, irow_tile, icol_tile, irow_mat, icol_mat, 0x0A);
//     }
 }

 syncthreads();

 int irow_tile_trs = icol_tile;
 int icol_tile_trs = irow_tile;
 int irow_mat_trs = block_id_local_row * TILE_DIM + icol_tile;
 int icol_mat_trs = block_id_local_col * TILE_DIM + irow_tile;

 /* Loop over elements of the matrix to be overwritten */
 if((irow_mat_trs < m) && (icol_mat_trs < n)){
     /* Overwrite the matrix element */
     int mat_idx = irow_mat_trs * n + icol_mat_trs;
     mat[trs_stack_offset + mat_idx] = buf[irow_tile_trs][icol_tile_trs];
//     if(icol_mat == 0 or icol_mat == 60 or icol_mat == 80 or icol_mat == 99 or irow_mat == 0 or irow_mat == 60 or irow_mat == 80 or irow_mat == 99){
       printf("[t=%i,b=%i,offset=%i]{%g} block_id_local = (%ix%i=%i), itile = (%ix%i) --> imat = (%ix%i)%c", threadIdx.x, blockIdx.x, trs_stack_offset, buf[irow_tile_trs][icol_tile_trs], block_id_local_row, block_id_local_col, block_id_local, irow_tile_trs, icol_tile_trs, irow_mat_trs, icol_mat_trs, 0x0A);
//     }
}

#endif

}
