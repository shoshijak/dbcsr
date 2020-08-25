/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#include "cublas.h"


/****************************************************************************/
int cublas_create(cublasHandle_t **handle)
{
  *handle = (cublasHandle_t*)malloc(sizeof(cublasHandle_t));
  cublasStatus_t cStatus = cublasCreate(*handle);
  if (cStatus != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS initialization failed\n");
    return(-1);
  }
  if (acc_error_check(cudaGetLastError())) return(-1);
  return(0);
}

/****************************************************************************/
int cublas_destroy(cublasHandle_t *handle)
{
  cublasStatus_t cStatus = cublasDestroy(*handle);
  free(handle);
  if (cStatus != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS finalization failed\n");
    return(-1);
  }
  if (acc_error_check(cudaGetLastError())) return(-1);
  return(0);
}


//===========================================================================
double checkSum(double* mat_c, int mat_m, int mat_n){
   double res = 0;
   for(int i=0; i < mat_m * mat_n; i++){
     res += mat_c[i];
   }
   return res;
}


//===========================================================================
void print_matrix_val(int m, int n, const double* mat_trs_a){
    int index = 0;
    for(int mi=0; mi < m; mi++){
        for(int ni=0; ni < n; ni++){
            index = ni * m + mi;
            printf("(m=%i,n=%i,i=%i) %g", mi, ni, index, mat_trs_a[index]);
        }
        printf("\n");
    }
    printf("\n");
}


__global__ void
print_matrices_on_gpu_before(int m, int n, int k,
			 const double *a_data, const double *b_data, const double *c_data)
{

  int index = 0;

  printf ("[print matrices before - %ix%i] BEFORE MATRIX A\n", m, k);
  for(int mi=0; mi<m; mi++){
    for(int ki=0; ki<k; ki++){
      index = ki * m + mi;
      printf("(m=%i,n=%i,i=%i) %g", mi, ki, index, a_data[index]);
    }
    printf("\n");
  }
  printf("\n");

  printf ("[print matrices before - %ix%i] BEFORE MATRIX B\n", k, n);
  for(int ki=0; ki<k; ki++){
    for(int ni=0; ni<n; ni++){
      index = ni * k + ki;
      printf("(m=%i,n=%i,i=%i) %g", ki, ni, index, b_data[index]);
    }
    printf("\n");
  }
  printf("\n");

  printf ("[print matrices before - %ix%i] BEFORE MATRIX C\n", m, n);
  for(int mi=0; mi<m; mi++){
    for(int ni=0; ni<n; ni++){
      index = ni * m + mi;
      printf("(m=%i,n=%i,i=%i) %g", mi, ni, index, c_data[index]);
    }
    printf("\n");
  }
  printf("\n");

}

__global__ void
print_matrices_on_gpu_after(int m, int n, int k,
			 const double *a_data, const double *b_data, const double *c_data)
{
  int index = 0;

//  printf ("[print matrices after - %ix%i] MATRIX A\n", m, k);
//  for(int mi=0; mi<m; mi++){
//    for(int ki=0; ki<k; ki++){
//      index = ki * m + mi;
//      printf("(m=%i,n=%i,i=%i) %g", mi, ki, index, a_data[index]);
//    }
//    printf("\n");
//  }
//  printf("\n");
///
////  printf ("[print matrices after - %ix%i] AFTER MATRIX B\n", k, n);
////  for(int ki=0; ki<k; ki++){
////    for(int ni=0; ni<n; ni++){
////      index = ni * k + ki;
////      printf("(m=%i,n=%i,i=%i) %g", ki, ni, index, b_data[index]);
////    }
////    printf("\n");
////  }
//  printf("\n");

  printf ("[print matrices after - %ix%i] AFTER MATRIX C\n", m, n);
  for(int mi=0; mi<m; mi++){
    for(int ni=0; ni<n; ni++){
      index = ni * m + mi;
      printf("(m=%i,n=%i,i=%i) %g", mi, ni, index, c_data[index]);
    }
    printf("\n");
  }
  printf("\n");

}

__global__ void
check_kernel(cublasHandle_t handle, char transa, char transb,
		     int m, int n, int k,
			 const double *alpha,
			 const double *a_data,
             int lda,
             const double *b_data,
             int ldb,
             const double *beta,
             const double *c_data,
             int ldc)
{
  printf ("[check_kernel - %ix%ix%i] start!\n", m, n, k);

  printf ("[check_kernel - %ix%ix%i] leading dims = (%i, %i)\n", m, n, k, lda, ldb);
  printf ("[check_kernel - %ix%ix%i] matrix element  0 (a = %g, b = %g, c = %g)\n", m, n, k, a_data[0], b_data[0], c_data[0]);
  printf ("[check_kernel - %ix%ix%i] matrix element 15 (a = %g, b = %g, c = %g)\n", m, n, k, a_data[15], b_data[15], c_data[15]);
  //printf ("[check_kernel - %ix%ix%i] parameters: alpha = %g, beta = %g)\n", m, n, k, *alpha, *beta);
  printf ("[check_kernel - %ix%ix%i] exiting!\n", m, n, k);

}

/****************************************************************************/
int cublas_dgemm(cublasHandle_t *handle, char transa, char transb,
		            int m, int n, int k,
			    int a_offset, int b_offset, int c_offset,
			    const double *a_data, const double *b_data, double *c_data,
			    double alpha, double beta, cudaStream_t *stream)
{
  // printf ("[cublas_dgemm - %ix%ix%i] start!\n", m, n, k);

  cublasOperation_t cTransa = transa=='N' ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cTransb = transb=='N' ? CUBLAS_OP_N : CUBLAS_OP_T;
  int &lda = transa=='N' ? m : k;
  int &ldb = transb=='N' ? k : n;

  printf ("[cublas_dgemm - %ix%ix%i] leading dims = (%i, %i)\n", m, n, k, lda, ldb);
  printf ("[cublas_dgemm - %ix%ix%i] offsets = (%i, %i, %i)\n", m, n, k, a_offset, b_offset, c_offset);
  printf ("[cublas_dgemm - %ix%ix%i] parameters: alpha = %g, beta = %g)\n", m, n, k, alpha, beta);

  //printf ("[cublas_dgemm - %ix%ix%i] get version ...\n", m, n, k);
  int cublasVersion = 0;
  cublasGetVersion(*handle, &cublasVersion);
  //printf ("[cublas_dgemm - %ix%ix%i] version %i\n", m, n, k, cublasVersion);

  cublasStatus_t cStatus = cublasSetStream(*handle, *stream);
  //printf ("[cublas_dgemm - %ix%ix%i] stream set!\n", m, n, k);
  if (cStatus != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS SetStream failed\n");
    return(-1);
  }

  check_kernel<<<1,1>>>(*handle, cTransa, cTransb,
				    m, n, k,
				    &alpha, &a_data[ a_offset ], lda,
				    &b_data[ b_offset], ldb,
				    &beta, &c_data[ c_offset], lda);

// FIXME translate this part of the ftn code!!!
//-      ! Call Cublas/Hipblas for big kernels
//-      IF (datatype .EQ. dbcsr_type_real_8 .AND. istat .EQ. -100) THEN
//-         ithread = 0
//-!$       ithread = omp_get_thread_num()
//-         DO istack = 1, stack_size
//-            IF (param_stack_host(2, istack) .LE. max_kernel_dim .AND. &
//-                param_stack_host(3, istack) .LE. max_kernel_dim) THEN
//-               ! Transpose for B-blocks
//-               transb = 'T'
//-            ELSE
//-               transb = 'N'
//-            ENDIF
//-#if (__CUDA)
//-            istat = cublas_dgemm_cu(cublas_handles(ithread + 1)%handle_ptr, &
//-                                    'N', transb, &
//-                                    INT(param_stack_host(1, istack), KIND=C_INT), &
//-                                    INT(param_stack_host(2, istack), KIND=C_INT), &
//-                                    INT(param_stack_host(3, istack), KIND=C_INT), &
//-                                    INT(param_stack_host(4, istack) - 1, KIND=C_INT), &
//-                                    INT(param_stack_host(5, istack) - 1, KIND=C_INT), &
//-                                    INT(param_stack_host(6, istack) - 1, KIND=C_INT), &
//+      ! Call batched matrix-matrix multiplication in libsmm_acc
//+      istat = libsmm_acc_process_cu(acc_devmem_cptr(param_stack_dev), &
//+                                    INT(stack_size, KIND=C_INT), &
//+                                    INT(dbcsr_ps_width, KIND=C_INT), &
//+                                    INT(datatype, KIND=C_INT), &
//                                     acc_devmem_cptr(a_data), &
//                                     acc_devmem_cptr(b_data), &
//                                     acc_devmem_cptr(c_data), &
//-                                    1.0_dp, 1.0_dp, &
//-                                    acc_stream_cptr(stream))
//-            IF (istat /= 0) DBCSR_ABORT("failed to run CUBLAS.")

  print_matrices_on_gpu_before<<<1,1>>>(
				    m, n, k,
				    &a_data[a_offset],
				    &b_data[b_offset],
				    &c_data[c_offset]);

  // TMP VALIDATION STEP:
  double* val_mat_a = (double*) malloc(m * k * sizeof(double));
  //printf ("[cublas_dgemm - %ix%ix%i] 1\n", m, n, k);
  ACC_API_CALL(Memcpy, (val_mat_a, &a_data[a_offset], m * k * sizeof(double), ACC(MemcpyDeviceToHost)));
  //printf ("[cublas_dgemm - %ix%ix%i] 2\n", m, n, k);
  double* val_mat_b = (double*) malloc(k * n * sizeof(double));
  //printf ("[cublas_dgemm - %ix%ix%i] 3\n", m, n, k);
  ACC_API_CALL(Memcpy, (val_mat_b, &b_data[b_offset], k * n * sizeof(double), ACC(MemcpyDeviceToHost)));
  //printf ("[cublas_dgemm - %ix%ix%i] 4\n", m, n, k);
  double* val_mat_c = (double*) malloc(m * n * sizeof(double));
  ACC_API_CALL(Memcpy, (val_mat_c, &c_data[c_offset], m * n * sizeof(double), ACC(MemcpyDeviceToHost)));

  for(int ni=0; ni<n; ni++){
    for(int mi=0; mi<m; mi++){
      double res = 0.;
      for(int ki=0; ki<k; ki++){
       int a_ind = ki * m + mi;
       int b_ind = ki * n + ni;
       res += val_mat_a[a_ind] * val_mat_b[b_ind];
      }
      int c_ind = ni * m + mi;
      val_mat_c[c_ind] += res;
    }
  }

  //printf ("[cublas_dgemm - %ix%ix%i] about to launch!\n", m, n, k);

  cublasStatus_t stat = cublasDgemm(*handle, cTransa, cTransb,
				    m, n, k,
				    &alpha, &a_data[a_offset], lda,
				    &b_data[ b_offset], ldb,
				    &beta, &c_data[ c_offset], lda);
  //printf ("[cublas_dgemm - %ix%ix%i] launched!\n", m, n, k);
  print_matrices_on_gpu_after<<<1,1>>>(
				    m, n, k,
				    &a_data[a_offset],
				    &b_data[b_offset],
				    &c_data[c_offset]);

  double checkSum_CPU = checkSum(val_mat_c, m, n);
  //printf ("[cublas_dgemm - %ix%ix%i] 5\n", m, n, k);
  ACC_API_CALL(Memcpy, (val_mat_c, &c_data[c_offset], m * n * sizeof(double), ACC(MemcpyDeviceToHost)));
  //printf ("[cublas_dgemm - %ix%ix%i] 6\n", m, n, k);
  double checkSum_GPU = checkSum(val_mat_c, m, n);

  printf ("[cublas_dgemm validation- %ix%i] CPU MATRIX C\n", m, n);
  print_matrix_val(m, n, val_mat_c);

  if(checkSum_GPU != checkSum_CPU){
      printf("Kernel validation FAIL for multiplication kernel %ix%ix%i\nchecksum CPU: %g, checksum GPU: %g\nchecksum_diff: %g\n", m, n, k, checkSum_CPU, checkSum_GPU, checkSum_GPU-checkSum_CPU);
      //exit(1);
  } else {
      printf("Kernel validation SUCCESS for multiplication kernel %ix%ix%i\nchecksum CPU: %g, checksum GPU: %g\nchecksum_diff: %g\n", m, n, k, checkSum_CPU, checkSum_GPU, checkSum_GPU-checkSum_CPU);
  }

  printf ("[cublas_dgemm - %ix%ix%i] exiting!\n", m, n, k);
  if (stat != CUBLAS_STATUS_SUCCESS) return(-1);
  if (acc_error_check(cudaGetLastError())) return(-1);
  return(0);
}
