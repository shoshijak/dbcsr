!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief   Routines that affect the DBCSR library as a whole
!> \author  Ole Schuett
!> \date    10-2014
! **************************************************************************************************

MODULE dbcsr_lib
   USE dbcsr_config, ONLY: cublas_handles
   USE dbcsr_cublas, ONLY: cublas_handle_destroy, &
                           cublas_handle_init
   USE dbcsr_kinds, ONLY: int_1_size, &
                          int_2_size, &
                          int_4_size, &
                          int_8_size
   USE dbcsr_machine, ONLY: default_output_unit
   USE dbcsr_mpiwrap, ONLY: add_mp_perf_env, &
                            describe_mp_perf_env, &
                            has_mp_perf_env, &
                            mp_environ
   USE dbcsr_multiply_api, ONLY: dbcsr_multiply_clear_mempools, &
                                 dbcsr_multiply_lib_finalize, &
                                 dbcsr_multiply_lib_init
#include "base/dbcsr_base_uses.f90"

!$ USE OMP_LIB, ONLY: omp_get_thread_num, omp_get_num_threads

#if defined (__DBCSR_ACC)
   USE ISO_C_BINDING, ONLY: C_INT
#endif

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_lib'

   PUBLIC :: dbcsr_init_lib, dbcsr_finalize_lib, dbcsr_clear_mempools

   LOGICAL, PRIVATE, SAVE :: is_initialized = .FALSE.

#if defined (__DBCSR_ACC)
   INTERFACE
      FUNCTION libsmm_acc_libcusmm_is_thread_safe_cu() &
         RESULT(thread_safe) &
         BIND(C, name="libsmm_acc_libcusmm_is_thread_safe")
         IMPORT
         INTEGER(KIND=C_INT)        :: thread_safe

      END FUNCTION libsmm_acc_libcusmm_is_thread_safe_cu
   END INTERFACE
#endif

CONTAINS

! **************************************************************************************************
!> \brief Initialize the DBCSR library
!>
!> Prepares the DBCSR library for use.
! **************************************************************************************************
   SUBROUTINE dbcsr_init_lib()

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_init_lib', routineP = moduleN//':'//routineN
#if defined(__LIBXSMM)
      USE libxsmm, ONLY: libxsmm_init
#endif
      INTEGER                                            :: error_handle, ithread, nthreads

#if defined(__DBCSR_ACC)
      INTEGER :: dbcsr_thread_safe, libcusmm_thread_safe
#endif

#if defined(__LIBXSMM)
      CALL libxsmm_init()
#endif

#if defined(__DBCSR_ACC)
      libcusmm_thread_safe = libsmm_acc_libcusmm_is_thread_safe_cu()  ! 0: not threaded, 1: threaded
      dbcsr_thread_safe = 0  ! not threaded
!$    dbcsr_thread_safe = 1  ! if DBCSR is compiled with openmp, set to threaded
      ! Check whether DBCSR and libcusmm (GPU backend) have the same level of threading
       IF (dbcsr_thread_safe /= libcusmm_thread_safe) then
          IF (dbcsr_thread_safe /= 0) then
              DBCSR_ABORT("DBCSR with threading, libcusmm without threading")
          ELSE
              DBCSR_ABORT("DBCSR without threading, libcusmm with threading")
          ENDIF
       ENDIF
#endif


!n_stack_buffers,mem_type, n_stack_mem_regions, stack_size, nstacks,&
!INTEGER, DIMENSION(3)                    :: nxstacks
!   ---------------------------------------------------------------------------

      IF (is_initialized) RETURN
      CALL timeset(routineN, error_handle)
      !
      IF (int_1_size /= 1) &
         DBCSR_ABORT("Incorrect assumption of an 8-bit integer size!")
      IF (int_2_size /= 2) &
         DBCSR_ABORT("Incorrect assumption of a 16-bit integer size!")
      IF (int_4_size /= 4) &
         DBCSR_ABORT("Incorrect assumption of a 32-bit integer size!")
      IF (int_8_size /= 8) &
         DBCSR_ABORT("Incorrect assumption of a 64-bit integer size!")

      IF (.NOT. has_mp_perf_env()) THEN
         CALL add_mp_perf_env()
      ENDIF

!$OMP     PARALLEL DEFAULT(NONE) PRIVATE(ithread) SHARED(nthreads, cublas_handles)
      CALL dbcsr_multiply_lib_init()
!$OMP MASTER
      nthreads = 1
!$    nthreads = omp_get_num_threads()
      ALLOCATE (cublas_handles(nthreads))
!$OMP END MASTER
!$OMP BARRIER
      ithread = 0
!$    ithread = omp_get_thread_num()
      ! init cublas. Internal check for cublas existing
      CALL cublas_handle_init(cublas_handles(ithread + 1))
!$OMP     END PARALLEL

      is_initialized = .TRUE.

      CALL timestop(error_handle)
   END SUBROUTINE dbcsr_init_lib

! **************************************************************************************************
!> \brief Finalize the DBCSR library
!>
!> Cleans up after the DBCSR library.  Used to deallocate persistent objects.
!> \param group ...
!> \param output_unit ...
! **************************************************************************************************
   SUBROUTINE dbcsr_finalize_lib(group, output_unit)
      INTEGER, INTENT(IN)                                :: group
      INTEGER, INTENT(IN), OPTIONAL                      :: output_unit

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_finalize_lib', &
                                     routineP = moduleN//':'//routineN

      INTEGER                                            :: error_handle, io_unit, ithread, mynode

      IF (.NOT. is_initialized) RETURN
      CALL timeset(routineN, error_handle)

      io_unit = 0
      IF (PRESENT(output_unit)) THEN
         io_unit = output_unit
      ELSE
         CALL mp_environ(taskid=mynode, groupid=group)
         IF (mynode .EQ. 0) io_unit = default_output_unit
      ENDIF

      IF (io_unit > 0) THEN
         WRITE (UNIT=io_unit, FMT="(/,T2,A)") REPEAT("-", 79)
         WRITE (UNIT=io_unit, FMT="(T2,A,T80,A)") "-", "-"
         WRITE (UNIT=io_unit, FMT="(T2,A,T35,A,T80,A)") "-", "DBCSR STATISTICS", "-"
         WRITE (UNIT=io_unit, FMT="(T2,A,T80,A)") "-", "-"
         WRITE (UNIT=io_unit, FMT="(T2,A)") REPEAT("-", 79)
      END IF

!$OMP     PARALLEL DEFAULT(NONE) PRIVATE(ithread) SHARED(io_unit, group, cublas_handles)
      CALL dbcsr_multiply_lib_finalize(group, io_unit)
      ithread = 0
!$    ithread = omp_get_thread_num()
      CALL cublas_handle_destroy(cublas_handles(ithread + 1))
!$OMP BARRIER
!$OMP MASTER
      DEALLOCATE (cublas_handles)
!$OMP END MASTER
!$OMP     END PARALLEL
      IF (io_unit > 0) WRITE (UNIT=io_unit, FMT="(T2,A)") REPEAT("-", 79)

      CALL describe_mp_perf_env(io_unit)

      is_initialized = .FALSE.

      CALL timestop(error_handle)
   END SUBROUTINE dbcsr_finalize_lib

! **************************************************************************************************
!> \brief  Deallocate memory contained in mempools
! **************************************************************************************************
   SUBROUTINE dbcsr_clear_mempools()

!$OMP     PARALLEL DEFAULT(NONE)
      CALL dbcsr_multiply_clear_mempools()
!$OMP     END PARALLEL
   END SUBROUTINE dbcsr_clear_mempools

END MODULE dbcsr_lib
