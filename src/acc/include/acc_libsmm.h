/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/
#ifndef DBCSR_ACC_LIBSMM_H
#define DBCSR_ACC_LIBSMM_H

#include "acc.h"

#ifdef __cplusplus
extern "C" {
#endif

extern "C" int libsmm_acc_init(void);
int libsmm_acc_is_thread_safe(void);

int libsmm_acc_transpose(const int* dev_trs_stack, int offset, int nblks,
  void* dev_data, acc_data_t datatype, int m, int n, acc_stream_t* stream);

//int libsmm_acc_process(const int* param_stack_host,
//  const libsmm_acc_stack_descriptor_type *param_stack_dev, int stack_size,
//  int nparams, acc_data_t datatype, const void *a_data, const void *b_data,
//  void *c_data, int m, int n, int k, int def_mnk, acc_stream_t *stream,
//  cublasHandle_t *handle);
extern "C" int libsmm_acc_process(const int* host_param_stack, const int* dev_param_stack, int stack_size,
  int nparams, acc_data_t datatype, const void* dev_a_data, const void* dev_b_data, void* dev_c_data,
  int m_max, int n_max, int k_max, acc_bool_t def_mnk, acc_stream_t* stream);

#ifdef __cplusplus
}
#endif

#endif /*DBCSR_ACC_LIBSMM_H*/
