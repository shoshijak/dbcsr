/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#include "include/libsmm_acc.h"
#include "parameters.h"
#include "parameters_utils.h"
#include "libsmm.h"
#include "libsmm_benchmark.h"
#include "cusmm_kernels.h"

#include <sstream>
#include <fstream>
#include <string>
#include <cstring>
#include <algorithm>
#include <array>
#include <iostream>

#if defined _OPENMP
#include <omp.h>
#endif

#define dbcsr_type_real_4     1
#define dbcsr_type_real_8     3
#define dbcsr_type_complex_4  5
#define dbcsr_type_complex_8  7


//===========================================================================
inline int launch_kernel_from_handle(ACC_DRV(function) const& kern_func, int nblks, int threads, ACC_DRV(stream) stream, void** args){

    std::cout << "[launch_kernel_from_handle]  start " << std::endl;
    ACC_DRV_CALL(
        LaunchJITKernel, (kern_func,      // kernel function,
                          nblks, 1, 1,    // grid dimension x, y, z
                          threads, 1, 1,  // block dimension x, y, z
                          0, stream,      // shared memory size and stream
                          args, NULL));   // arguments
    std::cout << "[launch_kernel_from_handle]  done " << std::endl;
    return 0;

}


//===========================================================================
inline void validate_kernel(ACC_DRV(function)& kern_func, ACC_DRV(stream) stream, int threads, int grouping, int m, int n, int k){

    std::cout << "[validate_kernel]  start " << std::endl;
    libsmm_benchmark_t* h;
    libsmm_benchmark_init(&h, test, m, n, k);

    // Run the matrix-matrix multiplication on the CPU
    std::cout << "[validate_kernel] CPU-multiplication - setup " << std::endl;
    memset(h->mat_c, 0, h->n_c * m * n * sizeof(double));
    matInit(h->mat_a, h->n_a, m, k, 42);
    matInit(h->mat_b, h->n_b, k, n, 24);
    stackInit(h->stack, h->n_stack, h->n_c, h->mat_c, h->n_a, h->mat_a, h->n_b, h->mat_b, m, n, k);
    std::cout << "[validate_kernel] CPU-multiplication - setup done" << std::endl;

    std::cout << "[validate_kernel] CPU-multiplication - calc start " << std::endl;
    stackCalc(h->stack, h->n_stack, h->mat_c, h->mat_a, h->mat_b, m, n, k);
    std::cout << "[validate_kernel] CPU-multiplication - calc done " << std::endl;
    double sumCPU = checkSum(h->mat_c, h->n_c, m, n);
    std::cout << "[validate_kernel] CPU-checksum - sumCPU="<< sumCPU << std::endl;

    // Run the matrix-matrix multiplication kernel on the GPU
    std::cout << "[validate_kernel] GPU-multiplication - setup start " << std::endl;
    ACC_API_CALL(Memcpy, (h->d_mat_a, h->mat_a, h->n_a * m * k * sizeof(double), ACC(MemcpyHostToDevice)));
    ACC_API_CALL(Memcpy, (h->d_mat_b, h->mat_b, h->n_b * k * n * sizeof(double), ACC(MemcpyHostToDevice)));
    ACC_API_CALL(Memcpy, (h->d_stack, h->stack, h->n_stack * 3 * sizeof(int), ACC(MemcpyHostToDevice)));
    ACC_API_CALL(Memset, (h->d_mat_c, 0, h->n_c * m * n * sizeof(double)));
    std::cout << "[validate_kernel] GPU-multiplication - setup done " << std::endl;

    void *args[] = { &h->d_stack, &h->n_stack, &h->d_mat_a, &h->d_mat_b, &h->d_mat_c };
    std::cout << "[validate_kernel] GPU-multiplication - calc start " << std::endl;
    int res = launch_kernel_from_handle(kern_func, ((h->n_stack + grouping - 1) / grouping), threads, stream, args);
    std::cout << "[validate_kernel] GPU-multiplication - calc done " << std::endl;
    ACC_API_CALL(Memcpy, (h->mat_c, h->d_mat_c, h->n_c * m * n * sizeof(double), ACC(MemcpyDeviceToHost)));

    // Validate the kernel based on results
    std::cout << "[validate_kernel] GPU-multiplication - checksum start " << std::endl;
    double sumGPU =  checkSum(h->mat_c, h->n_c, m, n);
    std::cout << "[validate_kernel] GPU-checksum - sumGPU="<< sumGPU << std::endl;
    if(sumGPU != sumCPU){
        printf("Kernel validation failed for kernel %ix%ix%i\nchecksum_diff: %g\nthreads: %i, grouping: %i\n", m, n, k, sumGPU-sumCPU, threads, grouping);
        exit(1);
    }
    std::cout << "[validate_kernel] GPU-multiplication - benchmark finalize... " << std::endl;
    libsmm_benchmark_finalize(h);
    std::cout << "[validate_kernel]  done " << std::endl;
}


//===========================================================================
inline void jit_kernel(ACC_DRV(function)& kern_func, libsmm_algo algo, int tile_m, int tile_n, int w, int v, int threads, int grouping, int minblocks, int m, int n, int k){

    // Get the code and the lowered name corresponding the kernel to launch
    std::cout << "[jit_kernel]  start " << std::endl;
    std::string kernel_code = cusmm_common; // prepend include file content to code
    std::string kernel_name;
    switch(algo) {
        case 1:
            kernel_code += cusmm_dnt_largeDB1;
            kernel_name = "cusmm_dnt_largeDB1<" +
                          std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                          std::to_string(tile_m) + ", " + std::to_string(tile_n) + ", " +
                          std::to_string(w) + ", " + std::to_string(v) + ", " +
                          std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
            break;
        case 2:
            kernel_code += cusmm_dnt_largeDB2;
            kernel_name = "cusmm_dnt_largeDB2<" +
                          std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                          std::to_string(tile_m) + ", " + std::to_string(tile_n) + ", " +
                          std::to_string(w) + ", " + std::to_string(v) + ", " +
                          std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
            break;
        case 3:
            kernel_code += cusmm_dnt_medium;
            kernel_name = "cusmm_dnt_medium<" +
                          std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                          std::to_string(tile_m) + ", " + std::to_string(tile_n) + ", " +
                          std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
            break;
        case 4:
            kernel_code += cusmm_dnt_small;
            kernel_name = "cusmm_dnt_small<" +
                          std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                          std::to_string(tile_m) + ", " + std::to_string(tile_n) + ", " +
                          std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
            break;
        case 5:
            kernel_code += cusmm_dnt_tiny;
            kernel_name = "cusmm_dnt_tiny<" +
                          std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                          std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
            break;
        default:
            printf("\nerror: algorithm number %i is not encoded.", algo);
            exit(1);
    }

    // Create JIT program
    ACC_RTC(Program) kernel_program;
    std::cout << "[jit_kernel]  CreateProgram " << std::endl;
    ACC_RTC_CALL(CreateProgram, (&kernel_program, kernel_code.c_str(), "smm_kernel.cu", 0, NULL, NULL));
    std::cout << "[jit_kernel] kernel code:" << kernel_code.c_str() << std::endl;

    // Add lowered name
    ACC_RTC_CALL(AddNameExpression, (kernel_program, kernel_name.c_str()));
    std::cout << "[jit_kernel] kernel name:" << kernel_name.c_str() << std::endl;

    // (JIT-)compile kernel program
#if defined(__CUDA) || defined(__HIP_PLATFORM_NVCC__)
    const std::string arch_opt = "--gpu-architecture=compute_" + std::to_string(ARCH_NUMBER);
    const char *compileOptions[] = {"-D__CUDA", "-w", arch_opt.c_str()};
    size_t nOptions = 3;
#else
    const char *compileOptions[] = {"-D__HIP"};
    size_t nOptions = 1;
#endif
    std::cout << "[jit_kernel] CompileProgram - start" << std::endl;
    ACC_RTC(Result) compileResult = ACC_RTC(CompileProgram)(kernel_program, nOptions, compileOptions);
    std::cout << "[jit_kernel] CompileProgram - done" << std::endl;

    // Obtain compilation log from the program.
    size_t logSize;
    ACC_RTC_CALL(GetProgramLogSize, (kernel_program, &logSize));
    char *log = new char[logSize];
    ACC_RTC_CALL(GetProgramLog, (kernel_program, log));
    std::cout << log << '\n';
    delete[] log;
    if (compileResult != RTC_SUCCESS) {
        exit(1);
    }

    // Obtain PTX from the program.
    size_t codeSize;
    std::cout << "[jit_kernel] GetLowLevelCodeSize - start" << std::endl;
    ACC_RTC_CALL(GetLowLevelCodeSize, (kernel_program, &codeSize));
    std::cout << "[jit_kernel] GetLowLevelCodeSize - done" << std::endl;
    char *code = new char[codeSize];
    std::cout << "[jit_kernel] GetLowLevelCode - start" << std::endl;
    ACC_RTC_CALL(GetLowLevelCode, (kernel_program, code));
    std::cout << "[jit_kernel] GetLowLevelCode - done" << std::endl;

    // Get lowered name
    const char *lowered_kernel_name;
    std::cout << "[jit_kernel] GetLoweredName - start" << std::endl;
    ACC_RTC_CALL(GetLoweredName, (kernel_program, kernel_name.c_str(), &lowered_kernel_name));
    std::cout << "[jit_kernel] GetLoweredName - done" << std::endl;

    // Get pointer to kernel from PTX
    ACC_DRV(module) module;
    std::cout << "[jit_kernel] ModuleLoadDataEx - start" << std::endl;
    ACC_DRV_CALL(ModuleLoadDataEx, (&module, code, 0, 0, 0));
    std::cout << "[jit_kernel] ModuleLoadDataEx - done" << std::endl;
    delete[] code;
    std::cout << "[jit_kernel] ModuleLoadGetFunction - start" << std::endl;
    ACC_DRV_CALL(ModuleGetFunction, (&kern_func, module, lowered_kernel_name));
    std::cout << "[jit_kernel] ModuleLoadGetFunction - done" << std::endl;

    // Set shared memory configuration
#if defined(__CUDA) || defined(__HIP_PLATFORM_NVCC__)
    ACC_DRV_CALL(FuncSetSharedMemConfig, (kern_func, ACC_DRV(SharedMemBankSizeEightByte)));
#else
    std::cout << "[jit_kernel] ... hip version" << std::endl;
    ACC_DRV_CALL(CtxSetSharedMemConfig, (ACC_DRV(SharedMemBankSizeEightByte)));
#endif
    std::cout << "[jit_kernel] SetSharedMemConfig - done" << std::endl;

    // Destroy program
    std::cout << "[jit_kernel] DestroyProgram - start" << std::endl;
    ACC_RTC_CALL(DestroyProgram, (&kernel_program));
    std::cout << "[jit_kernel] DestroyProgram - done" << std::endl;
}


void add_kernel_handle_to_jitted_kernels(ACC_DRV(function) kern_func, ACC_DRV(stream) stream, Triplet h_mnk, int& threads, int& grouping, bool& cpu_fallback){

    std::cout << "[add_kernel_handle_to_jitted_kernels]  start " << std::endl;
    // Check whether autotuned parameters are given for this kernel, and if so, retrieve them
    if (ht.find(h_mnk) != ht.end()){

        // Retrieve launching parameters
        const KernelParameters params = ht.at(h_mnk);
        libsmm_algo algo = libsmm_algo(params[0]); // enum {largeDB1, largeDB2, medium, small, tiny}
        int tile_m = params[1];
        int tile_n = params[2];
        int w = params[3];
        int v = params[4];
        threads = params[5];
        grouping = params[6];
        int minblocks =  params[7];

        // JIT and validate the kernel
        std::cout << "[add_kernel_handle_to_jitted_kernels] jit_kernel - start " << std::endl;
        jit_kernel(kern_func, algo, tile_m, tile_n, w, v, threads, grouping, minblocks, h_mnk[0], h_mnk[1], h_mnk[2]);
        std::cout << "[add_kernel_handle_to_jitted_kernels] jit_kernel - done " << std::endl;
        std::cout << "[add_kernel_handle_to_jitted_kernels] validate_kernel - start " << std::endl;
        validate_kernel(kern_func, stream, threads, grouping, h_mnk[0], h_mnk[1], h_mnk[2]);
        std::cout << "[add_kernel_handle_to_jitted_kernels] validate_kernel - done " << std::endl;

        // Store the handle to the JIT-ed kernel
        std::cout << "[add_kernel_handle_to_jitted_kernels] emplace - start " << std::endl;
        kernel_handles.emplace(h_mnk, kernel_launcher(kern_func, threads, grouping));
        std::cout << "[add_kernel_handle_to_jitted_kernels] emplace - done " << std::endl;

    } else { // there exist no autotuned parameters for this (m, n, k)-triplet, fall back to CPU

        cpu_fallback = true;

    }
    std::cout << "[add_kernel_handle_to_jitted_kernels]  done " << std::endl;

}


//===========================================================================
int libsmm_process_d(int *param_stack, int stack_size, ACC_DRV(stream) stream, int m, int n, int k, double *a_data, double *b_data, double *c_data){

    std::cout << "[libsmm_process_d] - start " << std::endl;
    ACC_DRV(function) kern_func = NULL;
    int threads, grouping;
    Triplet h_mnk = { m, n, k };
    static bool cpu_fallback = false;
    std::unordered_map<std::array<int, 3>, kernel_launcher>::iterator kernel_it;

#if defined _OPENMP
#pragma omp critical (jit_multiplication)
{
#endif

    // Look up the kernel in the table of already JITed kernels
    kernel_it = kernel_handles.find(h_mnk);
    if (kernel_it == kernel_handles.end()){  // the kernel has not been JIT-ed yet

        std::cout << "[libsmm_process_d] add_kernel_handle_to_jitted_kernels - start " << std::endl;
        add_kernel_handle_to_jitted_kernels(kern_func, stream, h_mnk, threads, grouping, cpu_fallback);
        std::cout << "[libsmm_process_d] add_kernel_handle_to_jitted_kernels - done " << std::endl;
        kernel_it = kernel_handles.find(h_mnk);
        std::cout << "[libsmm_process_d] got kernel iterator " << std::endl;

    }  // now the kernel has been jitted

#if defined _OPENMP
}
#endif

    std::cout << "[libsmm_process_d] after critical region " << std::endl;
    std::cout << "[libsmm_process_d] CPU fallback = " << cpu_fallback << std::endl;

    if(cpu_fallback){
        std::cout << "[libsmm_process_d] - CPU fallback " << std::endl;
        return -2; // fall back to CPU
    } else {

        // Retrieve kernel launching parameters
        std::cout << "[libsmm_process_d] - retrieve launching parameters " << std::endl;
        kern_func = kernel_it->second.kernel_function;
        threads = kernel_it->second.threads;
        grouping = kernel_it->second.grouping;

        // Construct argument pointer list and launch kernel
        void *args[] = { &param_stack, &stack_size, &a_data, &b_data, &c_data };
        int res = launch_kernel_from_handle(kern_func, ((stack_size + grouping - 1) / grouping), threads, stream, args);

        std::cout << "[libsmm_process_d] - done " << std::endl;
        return res;
    }

}


//===========================================================================
extern "C" int libsmm_acc_process (void *param_stack, int stack_size, int nparams, int datatype, void *a_data, void *b_data, void *c_data, int m, int n, int k, int def_mnk, void *stream){
    if(def_mnk!=1)
        return -1; // inhomogeneous stacks not supported
    if(datatype==dbcsr_type_real_8) {
      if(m>MAX_BLOCK_DIM || n>MAX_BLOCK_DIM || k>MAX_BLOCK_DIM)
        return -1; // maximum size over any dimension
      else
        return (libsmm_process_d ((int *) param_stack, stack_size, *((ACC_DRV(stream) *) stream), m, n, k, (double *) a_data, (double *) b_data, (double *) c_data));
    }
    return -1; // datatype not supported
};


//===========================================================================
void jit_transpose_handle(ACC_DRV(function)& kern_func, int m, int n){

    std::cout << "[jit_transpose_handle] start " << std::endl;
    // Create nvrtcProgram
    ACC_RTC(Program) kernel_program;
<<<<<<< HEAD
    std::string transpose_code = cusmm_common + cusmm_transpose;
||||||| merged common ancestors
    std::string transpose_code = cusmm_common + cusmm_transpose; 
=======
    std::string transpose_code = cusmm_common + cusmm_transpose; 
    std::cout << "[jit_transpose_handle] create program " << std::endl;
>>>>>>> [TORM] Debug prints and print JIT compilation errors
    ACC_RTC_CALL(CreateProgram, (&kernel_program, transpose_code.c_str(), "transpose_kernel.cu", 0, NULL, NULL));
    std::cout << "[jit_transpose_handle] transpose code:" << transpose_code.c_str() << std::endl;

    // Add lowered name
    std::string kernel_name = "transpose_d<" + std::to_string(m) + ", " + std::to_string(n) + ">";
    ACC_RTC_CALL(AddNameExpression, (kernel_program, kernel_name.c_str()));
    std::cout << "[jit_transpose_handle] transpose name:" << kernel_name.c_str() << std::endl;

    // (JIT-)compile
#if defined(__CUDA) || defined(__HIP_PLATFORM_NVCC__)
    const std::string arch_opt = "--gpu-architecture=compute_" + std::to_string(ARCH_NUMBER);
    const char *compileOptions[] = {"-D__CUDA", "-w", arch_opt.c_str()};
    size_t nOptions = 3;
#else
    const char *compileOptions[] = {"-D__HIP"};
    size_t nOptions = 1;
#endif
    ACC_RTC_CALL(CompileProgram, (kernel_program, nOptions, compileOptions));

    // Obtain PTX from the program.
    size_t codeSize;
    ACC_RTC_CALL(GetLowLevelCodeSize, (kernel_program, &codeSize));
    char *code = new char[codeSize];
    ACC_RTC_CALL(GetLowLevelCode, (kernel_program, code));

    // Get lowered name
    const char *lowered_kernel_name;
    ACC_RTC_CALL(GetLoweredName, (kernel_program, kernel_name.c_str(), &lowered_kernel_name));

    // Get pointer to kernel from PTX
    ACC_DRV(module) module;
    ACC_DRV_CALL(ModuleLoadDataEx, (&module, code, 0, 0, 0));
    delete[] code;
    ACC_DRV_CALL(ModuleGetFunction, (&kern_func, module, lowered_kernel_name));

    // Set shared memory configuration
#if defined(__CUDA) || defined(__HIP_PLATFORM_NVCC__)
    ACC_DRV_CALL(FuncSetSharedMemConfig, (kern_func, ACC_DRV(SharedMemBankSizeEightByte)));
#else
    ACC_DRV_CALL(CtxSetSharedMemConfig, (ACC_DRV(SharedMemBankSizeEightByte)));
#endif

    // Destroy program
    ACC_RTC_CALL(DestroyProgram, (&kernel_program));
    std::cout << "[jit_transpose_handle] done " << std::endl;
}


//===========================================================================
int libsmm_transpose_d(int *trs_stack, int offset, int nblks,
                       double *buffer, int m, int n, ACC_DRV(stream) stream) {

    std::cout << "[libsmm_transpose_d] start " << std::endl;
    ACC_DRV(function) kern_func;

    // Look up the kernel in the table of already JITed kernels
    Triplet h_mnk = { m, n, 0 };
    std::unordered_map<std::array<int, 3>, ACC_DRV(function)>::iterator kernel_it;

#if defined _OPENMP
#pragma omp critical (jit_transpose)
{
#endif

    std::cout << "[libsmm_transpose_d] find handle " << std::endl;
    kernel_it = transpose_handles.find(h_mnk);
    if(kernel_it == transpose_handles.end()){  // the kernel has not been JIT-ed yet

        std::cout << "[libsmm_transpose_d] not jitted yet " << std::endl;
        // JIT and store a kernel for this transposition
        jit_transpose_handle(kern_func, m, n);
        transpose_handles.emplace(h_mnk, kern_func);
        kernel_it = transpose_handles.find(h_mnk);

    }

#if defined _OPENMP
}
#endif

    std::cout << "[libsmm_transpose_d] jitting done " << std::endl;
    // Construct argument pointer list and launch function
    kern_func = kernel_it->second; // retrieve handle
    int* trs_stack_ = trs_stack + offset;
    void *args[] = { &trs_stack_, &buffer};

    std::cout << "[libsmm_transpose_d] done " << std::endl;
    return launch_kernel_from_handle(kern_func, nblks, 128, stream, args);

}


//===========================================================================
extern "C" int libsmm_acc_transpose (void *trs_stack, int offset, int nblks, void *buffer,int datatype, int m, int n, void* stream) {
    if(datatype != dbcsr_type_real_8)
        return 0; // transpose not needed
    if(m>MAX_BLOCK_DIM || n>MAX_BLOCK_DIM)
      return 0; // maximum size over any dimension
    return libsmm_transpose_d((int *) trs_stack, offset, nblks, (double *) buffer, m, n, *((ACC_DRV(stream) *) stream));
}


//===========================================================================
extern "C" int libsmm_acc_libsmm_is_thread_safe() {
#if defined _OPENMP
    return 1;  // i.e. true, libsmm is threaded
#else
    return 0;  // i.e. false, libsmm is not threaded
#endif
}
