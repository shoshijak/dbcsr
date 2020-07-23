/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/

#include "parameters.h"
#include "parameters_utils.h"
#include "libsmm_acc.h"
#include "libsmm_acc_benchmark.h"
#include "smm_acc_kernels.h"

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

// MACRO HELPERS
#define STRINGIFY_NX(x) #x
#define STRINGIFY(x) STRINGIFY_NX(x)
#define CONCAT_NX(A, B) A ## B
#define CONCAT(A, B) CONCAT_NX(A, B)

#define TILE_DIM 16

// The macro ARCH_OPTION, when expanded, is a string literal containing the
// jit compiler option specifying the target architecture
#if defined(__CUDA) || defined(__HIP_PLATFORM_NVCC__)
#define ARCH_OPTION_NAME --gpu-architecture=compute_
#else
#define ARCH_OPTION_NAME --amdgpu-target=
#endif
#define ARCH_OPTION STRINGIFY(CONCAT(ARCH_OPTION_NAME, ARCH_NUMBER))


//===========================================================================
void timeset(std::string routine_name, int& handle){
    const char* routine_name_ = routine_name.c_str();
    int routine_name_length  = routine_name.length();
    dbcsr_timeset(&routine_name_, &routine_name_length, &handle);
}

void timestop(int handle){
    dbcsr_timestop(&handle);
}

//===========================================================================
inline int launch_kernel_from_handle(ACC_DRV(function) const& kern_func, int nblks, int threads, ACC_DRV(stream) stream, void** args){

    ACC_DRV_CALL(
        LaunchJITKernel, (kern_func,      // kernel function,
                          nblks, 1, 1,    // grid dimension x, y, z
                          threads, 1, 1,  // block dimension x, y, z
                          0, stream,      // shared memory size and stream
                          args, NULL));   // arguments
    return 0;

}


//===========================================================================
inline void validate_kernel(ACC_DRV(function)& kern_func, ACC_DRV(stream) stream, int threads, int grouping, int m, int n, int k){

    libsmm_acc_benchmark_t* h;
    libsmm_acc_benchmark_init(&h, test, m, n, k);

    // Run the matrix-matrix multiplication on the CPU
    memset(h->mat_c, 0, h->n_c * m * n * sizeof(double));
    matInit(h->mat_a, h->n_a, m, k, 42);
    matInit(h->mat_b, h->n_b, k, n, 24);
    stackInit(h->stack, h->n_stack, h->n_c, h->mat_c, h->n_a, h->mat_a, h->n_b, h->mat_b, m, n, k);

    stackCalc(h->stack, h->n_stack, h->mat_c, h->mat_a, h->mat_b, m, n, k);
    double sumCPU = checkSum(h->mat_c, h->n_c, m, n);

    // Run the matrix-matrix multiplication kernel on the GPU
    ACC_API_CALL(Memcpy, (h->d_mat_a, h->mat_a, h->n_a * m * k * sizeof(double), ACC(MemcpyHostToDevice)));
    ACC_API_CALL(Memcpy, (h->d_mat_b, h->mat_b, h->n_b * k * n * sizeof(double), ACC(MemcpyHostToDevice)));
    ACC_API_CALL(Memcpy, (h->d_stack, h->stack, h->n_stack * 3 * sizeof(int), ACC(MemcpyHostToDevice)));
    ACC_API_CALL(Memset, (h->d_mat_c, 0, h->n_c * m * n * sizeof(double)));

    void *args[] = { &h->d_stack, &h->n_stack, &h->d_mat_a, &h->d_mat_b, &h->d_mat_c };
    launch_kernel_from_handle(kern_func, ((h->n_stack + grouping - 1) / grouping), threads, stream, args);
    ACC_API_CALL(Memcpy, (h->mat_c, h->d_mat_c, h->n_c * m * n * sizeof(double), ACC(MemcpyDeviceToHost)));

    // Validate the kernel based on results
    double sumGPU = checkSum(h->mat_c, h->n_c, m, n);
    libsmm_acc_benchmark_finalize(h);
    if(sumGPU != sumCPU){
        printf("Kernel validation failed for multiplication kernel %ix%ix%i\nchecksum CPU: %g, checksum GPU: %g\nchecksum_diff: %g\n", m, n, k, sumCPU, sumGPU, sumGPU-sumCPU);
        exit(1);
    }
}


//===========================================================================
inline void jit_kernel(ACC_DRV(function)& kern_func, libsmm_acc_algo algo, int tile_m, int tile_n, int w, int v, int threads, int grouping, int minblocks, int m, int n, int k){
    std::string routineN = "jit_kernel_multiply";
    int handle;
    timeset(routineN, handle);

    // Get the code and the lowered name corresponding the kernel to launch
    std::string kernel_code = smm_acc_common; // prepend include file content to code
    std::string kernel_name;
    switch(algo) {
        case 1:
            kernel_code += smm_acc_dnt_largeDB1;
            kernel_name = "smm_acc_dnt_largeDB1<" +
                          std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                          std::to_string(tile_m) + ", " + std::to_string(tile_n) + ", " +
                          std::to_string(w) + ", " + std::to_string(v) + ", " +
                          std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
            break;
        case 2:
            kernel_code += smm_acc_dnt_largeDB2;
            kernel_name = "smm_acc_dnt_largeDB2<" +
                          std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                          std::to_string(tile_m) + ", " + std::to_string(tile_n) + ", " +
                          std::to_string(w) + ", " + std::to_string(v) + ", " +
                          std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
            break;
        case 3:
            kernel_code += smm_acc_dnt_medium;
            kernel_name = "smm_acc_dnt_medium<" +
                          std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                          std::to_string(tile_m) + ", " + std::to_string(tile_n) + ", " +
                          std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
            break;
        case 4:
            kernel_code += smm_acc_dnt_small;
            kernel_name = "smm_acc_dnt_small<" +
                          std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                          std::to_string(tile_m) + ", " + std::to_string(tile_n) + ", " +
                          std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
            break;
        case 5:
            kernel_code += smm_acc_dnt_tiny;
            kernel_name = "smm_acc_dnt_tiny<" +
                          std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + ", " +
                          std::to_string(threads) + ", " + std::to_string(grouping) + ", " + std::to_string(minblocks) + ">";
            break;
        default:
            printf("\nERROR: algorithm number %i is not encoded.\n", algo);
            exit(1);
    }

    // Create JIT program
    ACC_RTC(Program) kernel_program;
    ACC_RTC_CALL(CreateProgram, (&kernel_program, kernel_code.c_str(), "smm_acc_kernel.cpp", 0, NULL, NULL));

    // Add lowered name
    ACC_RTC_CALL(AddNameExpression, (kernel_program, kernel_name.c_str()));

    // (JIT-)compile kernel program
#if defined(__CUDA) || defined(__HIP_PLATFORM_NVCC__)
    const char *compileOptions[] = {"-D__CUDA", "-w", ARCH_OPTION};
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
#endif

    // Destroy program
    ACC_RTC_CALL(DestroyProgram, (&kernel_program));

    timestop(handle);
}


kernel_map_iterator add_kernel_handle_to_jitted_kernels(ACC_DRV(function) kern_func, ACC_DRV(stream) stream, Triplet h_mnk, int& threads, int& grouping){

    kernel_map_iterator kernel_it = kernel_handles.end();

    // Check whether autotuned parameters are given for this kernel, and if so, retrieve them
    if (ht.find(h_mnk) != ht.end()){

        // Retrieve launching parameters
        const KernelParameters params = ht.at(h_mnk);
        libsmm_acc_algo algo = libsmm_acc_algo(params[0]); // enum {largeDB1, largeDB2, medium, small, tiny}
        int tile_m = params[1];
        int tile_n = params[2];
        int w = params[3];
        int v = params[4];
        threads = params[5];
        grouping = params[6];
        int minblocks =  params[7];

        // JIT and validate the kernel
        jit_kernel(kern_func, algo, tile_m, tile_n, w, v, threads, grouping, minblocks, h_mnk[0], h_mnk[1], h_mnk[2]);
        validate_kernel(kern_func, stream, threads, grouping, h_mnk[0], h_mnk[1], h_mnk[2]);

        // Store the handle to the JIT-ed kernel
        auto kernel_it_emplaced = kernel_handles.emplace(h_mnk, kernel_launcher(kern_func, threads, grouping));
        kernel_it = kernel_it_emplaced.first;

    }

    return kernel_it;

}


//===========================================================================
int libsmm_acc_process_d(const int *param_stack, int stack_size, ACC_DRV(stream) stream, int m, int n, int k, const double *a_data, const double *b_data, double *c_data){

    ACC_DRV(function) kern_func = NULL;
    int threads, grouping;
    Triplet h_mnk = { m, n, k };
    kernel_map_iterator kernel_it;

#pragma omp critical (jit_multiplication)
{

    // Look up the kernel in the table of already JITed kernels
    kernel_it = kernel_handles.find(h_mnk);
    if (kernel_it == kernel_handles.end()){  // the kernel has not been JIT-ed yet

        kernel_it = add_kernel_handle_to_jitted_kernels(kern_func, stream, h_mnk, threads, grouping);

    }  // if the kernel could be jited successfully, the kernel_it iterator now points to the kernel_launcher.
       // if this wasn't possible, is set to kernel_handles.end()

}

    if (kernel_it == kernel_handles.end()){  // the kernel could not be JIT-ed, so we should fall back to CPU

        return -2; // fall back to CPU

    } else {

        // Retrieve kernel launching parameters
        kern_func = kernel_it->second.kernel_function;
        threads = kernel_it->second.threads;
        grouping = kernel_it->second.grouping;

        // Construct argument pointer list and launch kernel
        void *args[] = { &param_stack, &stack_size, &a_data, &b_data, &c_data };
        return launch_kernel_from_handle(kern_func, ((stack_size + grouping - 1) / grouping), threads, stream, args);

    }

}


//===========================================================================
extern "C" int libsmm_acc_process (const libsmm_acc_stack_descriptor_type *param_stack, int stack_size, int nparams, acc_data_t datatype, const void *a_data, const void *b_data, void *c_data, int m, int n, int k, int def_mnk, acc_stream_t *stream){
    if(def_mnk!=1)
        return -1; // inhomogeneous stacks not supported
    if(datatype==dbcsr_type_real_8) {
      return (libsmm_acc_process_d ((const int *) param_stack, stack_size, *((ACC_DRV(stream) *) stream), m, n, k, (const double *) a_data, (const double *) b_data, (double *) c_data));
    }
    return -1; // datatype not supported
};


//===========================================================================
void print_matrix_val(int n_a, int m, int n, double* mat_trs_a){
    int index = 0;
    for(int s=0; s < n_a; s++){
        printf("\t[s=%i]\n", s);
        for(int mi=0; mi < m; mi++){
            for(int ni=0; ni < n; ni++){
                index = (s * n * m) + (ni * m + mi);
                printf("(m=%i,n=%i,i=%i) %g", mi, ni, index, mat_trs_a[index]);
            }
            printf("\n");
        }
    }
}


//===========================================================================
inline void validate_transpose_kernel(ACC_DRV(function)& kern_func, int threads, ACC_DRV(stream) stream, int m, int n){

    libsmm_acc_benchmark_t* h;
    libsmm_acc_benchmark_init(&h, test, m, 0, n);

    // Initialize arrays
    matInit(h->mat_a, h->n_a, m, n, 42);
    memset(h->mat_trs_a, 0, h->n_a * m * n * sizeof(double));
    stackInitTransp(h->stack_trs_a, h->n_stack_trs_a, m, n);

    // Run the matrix-matrix multiplication on the CPU
    stackTransp(h->stack_trs_a, h->n_stack_trs_a, h->mat_a, h->mat_trs_a, m, n);
    double sumCPU = checkSumTransp(h->mat_trs_a, h->n_stack_trs_a, m, n);

    // Run the matrix-matrix multiplication kernel on the GPU
    ACC_API_CALL(Memcpy, (h->d_mat_a, h->mat_a, h->n_a * m * n * sizeof(double), ACC(MemcpyHostToDevice)));
    ACC_API_CALL(Memcpy, (h->d_stack_trs_a, h->stack_trs_a, h->n_stack_trs_a * sizeof(int), ACC(MemcpyHostToDevice)));

    void *args[] = {&h->d_stack_trs_a, &h->d_mat_a};
    int num_tiles_row = (n + TILE_DIM - 1) / TILE_DIM;
    int num_tiles_col = (m + TILE_DIM - 1) / TILE_DIM;
    int num_tiles = num_tiles_row * num_tiles_col;

    launch_kernel_from_handle(kern_func, h->n_stack_trs_a * num_tiles, threads, stream, args);
    ACC_API_CALL(Memcpy, (h->mat_trs_a, h->d_mat_a, h->n_a * m * n * sizeof(double), ACC(MemcpyDeviceToHost)));

    // Validate the kernel based on results
    double sumGPU = checkSumTransp(h->mat_trs_a, h->n_stack_trs_a, m, n);
    libsmm_acc_benchmark_finalize(h);
    if(sumGPU != sumCPU){
        printf("Kernel validation failed for transpose kernel %ix%i\nchecksum CPU: %g, checksum GPU: %g\nchecksum_diff: %g\n", m, n, sumCPU, sumGPU, sumGPU-sumCPU);
        exit(1);
    }
}


//===========================================================================
void jit_transpose_handle(ACC_DRV(function)& kern_func, int m, int n){

    std::string routineN = "jit_kernel_transpose";
    int handle;
    timeset(routineN, handle);

    // Create nvrtcProgram
    ACC_RTC(Program) kernel_program;
    std::string transpose_code = smm_acc_common + smm_acc_transpose;
    ACC_RTC_CALL(CreateProgram, (&kernel_program, transpose_code.c_str(), "transpose_kernel.cpp", 0, NULL, NULL));

    // Add lowered name
    std::string kernel_name = "transpose_d<" + std::to_string(m) + ", " + std::to_string(n) + ">";
    ACC_RTC_CALL(AddNameExpression, (kernel_program, kernel_name.c_str()));

    // (JIT-)compile
#if defined(__CUDA) || defined(__HIP_PLATFORM_NVCC__)
    const char *compileOptions[] = {"-D__CUDA", "-w", ARCH_OPTION};
    size_t nOptions = 3;
#else
    const char *compileOptions[] = {"-D__HIP"};
    size_t nOptions = 1;
#endif
    nvrtcResult compileResult = nvrtcCompileProgram(kernel_program, nOptions, compileOptions);

    // Obtain compilation log from the program.
    size_t logSize;
    ACC_RTC_CALL(GetProgramLogSize, (kernel_program, &logSize));
    char *log = new char[logSize];
    ACC_RTC_CALL(GetProgramLog, (kernel_program, log));
    std::cout << log << '\n';
    delete[] log;
    if (compileResult != NVRTC_SUCCESS) {
        exit(1);
    }

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
#endif

    // Destroy program
    ACC_RTC_CALL(DestroyProgram, (&kernel_program));

    timestop(handle);
}


//===========================================================================
int libsmm_acc_transpose_d(const int *trs_stack, int offset, int stack_size,
                           double *buffer, int m, int n, ACC_DRV(stream) stream) {

    ACC_DRV(function) kern_func;
    int threads = 256;
//    if (m * n + warp_size <= 128){
//        threads = m*n  - (m*n % warp_size) + warp_size;
//    }

    // Look up the kernel in the table of already JITed kernels
    Triplet h_mnk = { m, n, 0 };
    std::unordered_map<std::array<int, 3>, ACC_DRV(function)>::iterator kernel_it;

#pragma omp critical (jit_transpose)
{

    kernel_it = transpose_handles.find(h_mnk);
    if(kernel_it == transpose_handles.end()){  // the kernel has not been JIT-ed yet

        // JIT and store a kernel for this transposition
        jit_transpose_handle(kern_func, m, n);
        validate_transpose_kernel(kern_func, threads, stream, m, n);
        transpose_handles.emplace(h_mnk, kern_func);
        kernel_it = transpose_handles.find(h_mnk);

    }

}

    // Construct argument pointer list and launch function
    kern_func = kernel_it->second; // retrieve handle
    const int* trs_stack_ = trs_stack + offset;
    void *args[] = {&trs_stack_, &buffer};

    int num_tiles_row = (n + TILE_DIM - 1) / TILE_DIM;
    int num_tiles_col = (m + TILE_DIM - 1) / TILE_DIM;
    int num_tiles = num_tiles_row * num_tiles_col;
    printf("[%i,%i] offset=%i, stack_size=%i, num_tiles=%i, threads=%i\n",m, n, offset, stack_size, num_tiles, threads);
    return launch_kernel_from_handle(kern_func, stack_size * num_tiles, threads, stream, args);

}


//===========================================================================
extern "C" int libsmm_acc_transpose (const int *trs_stack, int offset, int stack_size, void *buffer, acc_data_t datatype, int m, int n, acc_stream_t* stream) {
    if(datatype != dbcsr_type_real_8)
        return 0; // transpose not needed
    return libsmm_acc_transpose_d(trs_stack, offset, stack_size, (double *) buffer, m, n, *((ACC_DRV(stream) *) stream));
}
