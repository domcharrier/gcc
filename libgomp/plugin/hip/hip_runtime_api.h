/* CUDA API description.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.

This file is part of GCC.

GCC is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3, or (at your option)
any later version.

GCC is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Under Section 7 of GPL version 3, you are granted additional
permissions described in the GCC Runtime Library Exception, version
3.1, as published by the Free Software Foundation.

You should have received a copy of the GNU General Public License and
a copy of the GCC Runtime Library Exception along with this program;
see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see
<http://www.gnu.org/licenses/>.

This header provides the minimum amount of typedefs, enums and function
declarations to be able to compile plugin-nvptx.c if hip/hip_runtime.h and
libcuda.so.1 are not available.  */

#ifndef GCC_CUDA_H
#define GCC_CUDA_H

#include <stdlib.h>

#define CUDA_VERSION 8000

typedef void *hipCtx_t;
typedef int hipDevice_t;
//#if defined(__LP64__) || defined(_WIN64)
//typedef unsigned long long hipDeviceptr_t;
//#else
//typedef unsigned hipDeviceptr_t;
//#endif
typedef 
typedef void *hipEvent_t;
typedef void *hipFunction_t;
//typedef void *CUlinkState;
typedef void *hipModule_t;
typedef size_t (*CUoccupancyB2DSize)(int);
typedef void *hipStream_t;

typedef enum {
#include "hiperror_t.inc"
} hipError_t;

typedef enum {
#include "hipdeviceattribute_t.inc"
} hipDeviceAttribute_t;

enum {
  hipEventDefault = 0,
  hipEventDisableTiming = 2
};

typedef enum {
  HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
  HIP_FUNC_ATTRIBUTE_NUM_REGS = 4
} hipFunction_attribute;

typedef enum {
  hipJitOptionWallTime = 2,
  hipJitOptionInfoLogBuffer = 3,
  hipJitOptionInfoLogBufferSizeBytes = 4,
  hipJitOptionErrorLogBuffer = 5,
  hipJitOptionErrorLogBufferSizeBytes = 6,
  hipJitOptionOptimizationLevel = 7,
  hipJitOptionLogVerbose = 12
} hipJitOption;

//typedef enum {
//  CU_JIT_INPUT_PTX = 1
//} CUjitInputType;

enum {
  hipDeviceScheduleAuto = 0
};

#define HIP_LAUNCH_PARAM_END ((void *) 0)
#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void *) 1)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE ((void *) 2)

enum {
  hipStreamDefault = 0,
  hipStreamNonBlocking = 1
};

hipError_t hipCtxCreate (hipCtx_t *, unsigned, hipDevice_t);
hipError_t hipCtxDestroy (hipCtx_t);
hipError_t hipCtxGetCurrent (hipCtx_t *);
hipError_t hipCtxGetDevice (hipDevice_t *);
hipError_t hipCtxPopCurrent (hipCtx_t *);
hipError_t hipCtxPushCurrent (hipCtx_t);
hipError_t hipCtxSynchronize (void);
hipError_t hipGetDevice (hipDevice_t *, int);
hipError_t hipDeviceTotalMem (size_t *, hipDevice_t);
hipError_t hipDeviceGetAttribute (int *, hipDeviceAttribute_t, hipDevice_t);
hipError_t hipGetDeviceCount (int *);
hipError_t hipDeviceGetName (char *, int, hipDevice_t);
hipError_t hipEventCreateWithFlags (hipEvent_t *, unsigned);
hipError_t hipEventDestroy (hipEvent_t);
hipError_t hipEventElapsedTime (float *, hipEvent_t, hipEvent_t);
hipError_t hipEventQuery (hipEvent_t);
hipError_t hipEventRecord (hipEvent_t, hipStream_t);
hipError_t hipEventSynchronize (hipEvent_t);
hipError_t hipFuncGetAttribute (int *, hipFunction_attribute, hipFunction_t);
char* hipGetErrorString (hipError_t);
hipError_t hipInit (unsigned);
hipError_t hipDriverGetVersion (int *);
hipError_t hipModuleLaunchKernel (hipFunction_t, unsigned, unsigned, unsigned, unsigned,
			 unsigned, unsigned, unsigned, hipStream_t, void **, void **);
hipError_t cuLinkAddData (CUlinkState, CUjitInputType, void *, size_t, const char *,
			unsigned, hipJitOption *, void **);
hipError_t cuLinkComplete (CUlinkState, void **, size_t *);
hipError_t cuLinkCreate (unsigned, hipJitOption *, void **, CUlinkState *);
hipError_t cuLinkDestroy (CUlinkState);
hipError_t hipMemGetInfo (size_t *, size_t *);
hipError_t hipMalloc (hipDeviceptr_t *, size_t);
hipError_t hipHostMalloc (void **, size_t);
hipError_t hipMemcpy (hipDeviceptr_t, hipDeviceptr_t, size_t);
hipError_t hipMemcpyDtoDAsync (hipDeviceptr_t, hipDeviceptr_t, size_t, hipStream_t);
hipError_t hipMemcpyDtoH (void *, hipDeviceptr_t, size_t);
hipError_t hipMemcpyDtoHAsync (void *, hipDeviceptr_t, size_t, hipStream_t);
hipError_t hipMemcpyHtoD (hipDeviceptr_t, const void *, size_t);
hipError_t hipMemcpyHtoDAsync (hipDeviceptr_t, const void *, size_t, hipStream_t);
hipError_t hipFree (hipDeviceptr_t);
hipError_t hipHostFree (void *);
hipError_t hipMemGetAddressRange (hipDeviceptr_t *, size_t *, hipDeviceptr_t);
hipError_t hipHostGetDevicePointer (hipDeviceptr_t *, void *, unsigned);
hipError_t hipModuleGetFunction (hipFunction_t *, hipModule_t, const char *);
hipError_t hipModuleGetGlobal (hipDeviceptr_t *, size_t *, hipModule_t, const char *);
hipError_t hipModuleLoad (hipModule_t *, const char *);
hipError_t hipModuleLoadData (hipModule_t *, const void *);
hipError_t hipModuleUnload (hipModule_t);
hipError_t hipOccupancyMaxPotentialBlockSize(int *, int *, hipFunction_t,
					  CUoccupancyB2DSize, size_t, int);
typedef void (*hipStreamCallback_t)(hipStream_t, hipError_t, void *);
hipError_t hipStreamAddCallback(hipStream_t, hipStreamCallback_t, void *, unsigned int);
hipError_t hipStreamCreateWithFlags (hipStream_t *, unsigned);
#define hipStreamDestroy hipStreamDestroy
hipError_t hipStreamDestroy (hipStream_t);
hipError_t hipStreamQuery (hipStream_t);
hipError_t hipStreamSynchronize (hipStream_t);
hipError_t hipStreamWaitEvent (hipStream_t, hipEvent_t, unsigned);

#endif /* GCC_CUDA_H */
