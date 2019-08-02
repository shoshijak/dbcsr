###############################################################################
# FindHIPcustom.cmake
###############################################################################
# Workaround bug in ROCm's Findhip.cmake

###############################################################################
# MACRO: Separate the options from the sources
###############################################################################
macro(HIPCUSTOM_GET_SOURCES_AND_OPTIONS _sources _cmake_options _hipcc_options _hcc_options _nvcc_options)
    set(${_sources})
    set(${_cmake_options})
    set(${_hipcc_options})
    set(${_hcc_options})
    set(${_nvcc_options})
    set(_hipcc_found_options FALSE)
    set(_hcc_found_options FALSE)
    set(_nvcc_found_options FALSE)
    foreach(arg ${ARGN})
        if("x${arg}" STREQUAL "xHIPCC_OPTIONS")
            set(_hipcc_found_options TRUE)
            set(_hcc_found_options FALSE)
            set(_nvcc_found_options FALSE)
        elseif("x${arg}" STREQUAL "xHCC_OPTIONS")
            set(_hipcc_found_options FALSE)
            set(_hcc_found_options TRUE)
            set(_nvcc_found_options FALSE)
        elseif("x${arg}" STREQUAL "xNVCC_OPTIONS")
            set(_hipcc_found_options FALSE)
            set(_hcc_found_options FALSE)
            set(_nvcc_found_options TRUE)
        elseif(
                "x${arg}" STREQUAL "xEXCLUDE_FROM_ALL" OR
                "x${arg}" STREQUAL "xSTATIC" OR
                "x${arg}" STREQUAL "xOBJECT" OR
                "x${arg}" STREQUAL "xSHARED" OR
                "x${arg}" STREQUAL "xMODULE"
                )
            list(APPEND ${_cmake_options} ${arg})
        else()
            if(_hipcc_found_options)
                list(APPEND ${_hipcc_options} ${arg})
            elseif(_hcc_found_options)
                list(APPEND ${_hcc_options} ${arg})
            elseif(_nvcc_found_options)
                list(APPEND ${_nvcc_options} ${arg})
            else()
                # Assume this is a file
                list(APPEND ${_sources} ${arg})
            endif()
        endif()
    endforeach()
endmacro()

###############################################################################
# HIP_ADD_LIBRARY
###############################################################################
macro(HIPCUSTOM_ADD_LIBRARY hip_target)
    # Separate the sources from the options
    HIPCUSTOM_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _hipcc_options _hcc_options _nvcc_options ${ARGN})
    message(STATUS "sources:" ${_sources})
    HIP_PREPARE_TARGET_COMMANDS(${hip_target} OBJ _generated_files _source_files ${_sources} ${_cmake_options} HIPCC_OPTIONS ${_hipcc_options} HCC_OPTIONS ${_hcc_options} NVCC_OPTIONS ${_nvcc_options})
    if(_source_files)
        list(REMOVE_ITEM _sources ${_source_files})
    endif()
    message(STATUS "options:" ${_cmake_options})
    message(STATUS "genfiles:" ${_generated_files})
    message(STATUS "sources:" ${_sources})
    message(STATUS add_library(${hip_target} ${_cmake_options} ${_generated_files} ${_sources} libhipsmm.cpp libhipsmm_benchmark.cpp))
    message(STATUS set_target_properties(${hip_target} PROPERTIES LINKER_LANGUAGE ${HIP_C_OR_CXX}))
    add_library(${hip_target} ${_cmake_options} ${_generated_files} ${_sources})
    set_target_properties(${hip_target} PROPERTIES LINKER_LANGUAGE ${HIP_C_OR_CXX})
endmacro()
# vim: ts=4:sw=4:expandtab:smartindent
