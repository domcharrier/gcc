/* Plugin for HIP execution on AMD devices.

   Copyright (C) 2013-2021 Free Software Foundation, Inc.

   Contributed by Mentor Embedded.

   This file is part of the GNU Offloading and Multi Processing Library
   (libgomp).

   Libgomp is free software; you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   Libgomp is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
   more details.

   Under Section 7 of GPL version 3, you are granted additional
   permissions described in the GCC Runtime Library Exception, version
   3.1, as published by the Free Software Foundation.

   You should have received a copy of the GNU General Public License and
   a copy of the GCC Runtime Library Exception along with this program;
   see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see
   <http://www.gnu.org/licenses/>.  */

/* AMD HIP-specific parts of OpenACC support.  The ROCm driver
   library appears to hold some implicit state, but the documentation
   is not clear as to what that state might be.  Or how one might
   propagate it from one thread to another.  */

#define _GNU_SOURCE
#include "openacc.h"
#include "config.h"
#include "symcat.h"
#include "libgomp-plugin.h"
#include "oacc-plugin.h"
#include "gomp-constants.h"
#include "oacc-int.h"

#include <pthread.h>
#include <hip/hip_runtime.h>
#include <stdbool.h>
#include <limits.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <errno.h>

/* An arbitrary fixed limit (128MB) for the size of the OpenMP soft stacks
   block to cache between kernel invocations.  For soft-stacks blocks bigger
   than this, we will free the block before attempting another GPU memory
   allocation (i.e. in GOMP_OFFLOAD_alloc).  Otherwise, if an allocation fails,
   we will free the cached soft-stacks block anyway then retry the
   allocation.  If that fails too, we lose.  */

#define DO_PRAGMA(x) _Pragma (#x)

#if PLUGIN_NVPTX_DYNAMIC
# include <dlfcn.h>

struct hip_lib_s {

# define HIP_ONE_CALL(call)			\
  __typeof (call) *call;
# define HIP_ONE_CALL_MAYBE_NULL(call)		\
  HIP_ONE_CALL (call)
#include "hip-lib.def"
# undef HIP_ONE_CALL
# undef HIP_ONE_CALL_MAYBE_NULL

} hip_lib;

/* -1 if init_hip_lib has not been called yet, false
   if it has been and failed, true if it has been and succeeded.  */
static signed char hip_lib_inited = -1;

/* Dynamically load the HIP runtime library and initialize function
   pointers, return false if unsuccessful, true if successful.  */
static bool
init_hip_lib (void)
{
  if (hip_lib_inited != -1)
    return hip_lib_inited;
  const char *hip_runtime_lib = "libamdhip64.so";
  void *h = dlopen (hip_runtime_lib, RTLD_LAZY);
  hip_lib_inited = false;
  if (h == NULL)
    return false;

# define HIP_ONE_CALL(call) HIP_ONE_CALL_1 (call, false)
# define HIP_ONE_CALL_MAYBE_NULL(call) HIP_ONE_CALL_1 (call, true)
# define HIP_ONE_CALL_1(call, allow_null)		\
  hip_lib.call = dlsym (h, #call);	\
  if (!allow_null && hip_lib.call == NULL)		\
    return false;
#include "hip-lib.def"
# undef HIP_ONE_CALL
# undef HIP_ONE_CALL_1
# undef HIP_ONE_CALL_MAYBE_NULL

  hip_lib_inited = true;
  return true;
}
# define HIP_CALL_PREFIX hip_lib.
#else

# define HIP_ONE_CALL(call)
# define HIP_ONE_CALL_MAYBE_NULL(call) DO_PRAGMA (weak call)
#include "hip-lib.def"
#undef HIP_ONE_CALL_MAYBE_NULL
#undef HIP_ONE_CALL

# define HIP_CALL_PREFIX
# define init_hip_lib() true
#endif

#include "secure_getenv.h"

#undef MIN
#undef MAX
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

/* Convenience macros for the frequently used HIP library call and
   error handling sequence as well as CUDA library calls that
   do the error checking themselves or don't do it at all.  */

#define HIP_CALL_ERET(ERET, FN, ...)		\
  do {						\
    unsigned __r				\
      = HIP_CALL_PREFIX FN (__VA_ARGS__);	\
    if (__r != hipSuccess)			\
      {						\
	GOMP_PLUGIN_error (#FN " error: %s",	\
			   hip_error (__r));	\
	return ERET;				\
      }						\
  } while (0)

#define HIP_CALL(FN, ...)			\
  HIP_CALL_ERET (false, FN, __VA_ARGS__)

#define HIP_CALL_ASSERT(FN, ...)		\
  do {						\
    unsigned __r				\
      = HIP_CALL_PREFIX FN (__VA_ARGS__);	\
    if (__r != hipSuccess)			\
      {						\
	GOMP_PLUGIN_fatal (#FN " error: %s",	\
			   hip_error (__r));	\
      }						\
  } while (0)

#define HIP_CALL_NOCHECK(FN, ...)		\
  HIP_CALL_PREFIX FN (__VA_ARGS__)

#define HIP_CALL_EXISTS(FN)			\
  HIP_CALL_PREFIX FN

static const char *
hip_error (hipError_t r)
{
  const char *fallback = "unknown HIP error";

  if (!HIP_CALL_EXISTS (hipGetErrorString))
    return fallback;
  else
    return hipGetErrorString(r);
}

/* Version of HIP/ROCm in the same MAJOR.MINOR format that is used by
   AMD. */
static char rocm_version_s[30];

static unsigned int instantiated_devices = 0;
static pthread_mutex_t hip_dev_lock = PTHREAD_MUTEX_INITIALIZER;

/* HIP specific definition of asynchronous queues.  */
struct goacc_asyncqueue
{
  hipStream_t hip_stream;
};

struct hip_callback
{
  void (*fn) (void *);
  void *ptr;
  struct goacc_asyncqueue *aq;
  struct hip_callback *next;
};

/* Thread-specific data for HIP.  */

struct hip_thread
{
  /* We currently have this embedded inside the plugin because libgomp manages
     devices through integer target_ids.  This might be better if using an
     opaque target-specific pointer directly from gomp_device_descr.  */
  struct hip_device *hip_dev;
};

/* Target data function launch information.  */

struct targ_fn_launch
{
  const char *fn;
  unsigned short dim[GOMP_DIM_MAX];
};

/* Target HIP object information.  */

struct targ_ptx_obj
{
  const char *code;
  size_t size;
};

/* Target data image information.  */

typedef struct hip_tdata
{
  const struct targ_ptx_obj *hip_objs;
  unsigned hip_num;

  const char *const *var_names;
  unsigned var_num;

  const struct targ_fn_launch *fn_descs;
  unsigned fn_num;
} hip_tdata_t;

/* Descriptor of a loaded function.  */

struct targ_fn_descriptor
{
  hipFunction_t fn;
  const struct targ_fn_launch *launch;
  int regs_per_thread;
  int max_threads_per_block;
};

/* A loaded HIP image.  */
struct hip_image_data
{
  const void *target_data;
  hipModule_t module;

  struct targ_fn_descriptor *fns;  /* Array of functions.  */
  
  struct hip_image_data *next;
};

struct hip_free_block
{
  void *ptr;
  struct hip_free_block *next;
};

struct hip_device
{
  hipCtx_t ctx;
  bool ctx_shared;
  hipDevice_t dev;

  int ord;
  bool overlap;
  bool map;
  bool concur;
  bool mkern;
  int mode;
  int clock_khz;
  int num_sms;
  int regs_per_block;
  int regs_per_sm;
  int warp_size;
  int max_threads_per_block;
  int max_threads_per_multiprocessor;
  int default_dims[GOMP_DIM_MAX];

  /* Length as used by the HIP Runtime API ('struct hipDeviceProp_t').  */
  char name[256];

  struct hip_image_data *images;  /* Images loaded on device.  */
  pthread_mutex_t image_lock;     /* Lock for above list.  */

  struct hip_free_block *free_blocks;
  pthread_mutex_t free_blocks_lock;

  /* OpenMP stacks, cached between kernel invocations.  */
  struct
    {
      hipDeviceptr_t ptr;
      size_t size;
      pthread_mutex_t lock;
    } omp_stacks;

  struct hip_device *next;
};

static struct hip_device **hip_devices;

static inline struct hip_thread *
hip_thread (void)
{
  return (struct hip_thread *) GOMP_PLUGIN_acc_thread ();
}

/* Initialize the device.  Return TRUE on success, else FALSE.  HIP_DEV_LOCK
   should be locked on entry and remains locked on exit.  */

static bool
hip_init (void)
{
  int ndevs;

  if (instantiated_devices != 0)
    return true;

  if (!init_hip_lib ())
    return false;

  HIP_CALL (hipInit, 0);

  int hip_driver_version;
  HIP_CALL_ERET (NULL, hipDriverGetVersion, &hip_driver_version);
  snprintf (hip_driver_version_s, sizeof hip_driver_version_s,
	    "HIP Driver %u.%u",
	    hip_driver_version / 1000, hip_driver_version % 1000 / 10);

  HIP_CALL (hipGetDeviceCount, &ndevs);
  hip_devices = GOMP_PLUGIN_malloc_cleared (sizeof (struct hip_device *)
					    * ndevs);

  return true;
}

/* Select the N'th HIP device for the current host thread.  The device must
   have been previously opened before calling this function.  */

static bool
hip_attach_host_thread_to_device (int n)
{
  hipDevice_t dev;
  hipError_t r;
  struct hip_device *hip_dev;
  hipCtx_t thd_ctx;

  r = HIP_CALL_NOCHECK (hipCtxGetDevice, &dev);
  if (r == HIP_ERROR_NOT_PERMITTED)
    {
      /* Assume we're in a HIP callback, just return true.  */
      return true;
    }
  if (r != hipSuccess && r != hipErrorInvalidContext)
    {
      GOMP_PLUGIN_error ("hipCtxGetDevice error: %s", hip_error (r));
      return false;
    }

  if (r != hipErrorInvalidContext && dev == n)
    return true;
  else
    {
      hipCtx_t old_ctx;

      hip_dev = hip_devices[n];
      if (!hip_dev)
	{
	  GOMP_PLUGIN_error ("device %d not found", n);
	  return false;
	}

      HIP_CALL (hipCtxGetCurrent, &thd_ctx);

      /* We don't necessarily have a current context (e.g. if it has been
         destroyed.  Pop it if we do though.  */
      if (thd_ctx != NULL)
	HIP_CALL (hipCtxPopCurrent, &old_ctx);

      HIP_CALL (hipCtxPushCurrent, hip_dev->ctx);
    }
  return true;
}

static struct hip_device *
hip_open_device (int n)
{
  struct hip_device *hip_dev;
  hipDevice_t dev, ctx_dev;
  hipError_t r;
  int async_engines, pi;

  HIP_CALL_ERET (NULL, hipGetDevice, &dev, n);

  hip_dev = GOMP_PLUGIN_malloc (sizeof (struct hip_device));

  hip_dev->ord = n;
  hip_dev->dev = dev;
  hip_dev->ctx_shared = false;

  r = HIP_CALL_NOCHECK (hipCtxGetDevice, &ctx_dev);
  if (r != hipSuccess && r != hipErrorInvalidContext)
    {
      GOMP_PLUGIN_error ("hipCtxGetDevice error: %s", hip_error (r));
      return NULL;
    }
  
  if (r != hipErrorInvalidContext && ctx_dev != dev)
    {
      /* The current host thread has an active context for a different device.
         Detach it.  */
      hipCtx_t old_ctx;
      HIP_CALL_ERET (NULL, hipCtxPopCurrent, &old_ctx);
    }

  HIP_CALL_ERET (NULL, hipCtxGetCurrent, &hip_dev->ctx);

  if (!hip_dev->ctx)
    HIP_CALL_ERET (NULL, hipCtxCreate, &hip_dev->ctx, hipDeviceScheduleAuto, dev);
  else
    hip_dev->ctx_shared = true;

  HIP_CALL_ERET (NULL, hipDeviceGetAttribute,
		  &pi, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, dev);
  hip_dev->overlap = pi;

  HIP_CALL_ERET (NULL, hipDeviceGetAttribute,
		  &pi, hipDeviceAttributeCanMapHostMemory, dev);
  hip_dev->map = pi;

  HIP_CALL_ERET (NULL, hipDeviceGetAttribute,
		  &pi, hipDeviceAttributeConcurrentKernels, dev);
  hip_dev->concur = pi;

  HIP_CALL_ERET (NULL, hipDeviceGetAttribute,
		  &pi, hipDeviceAttributeComputeMode, dev);
  hip_dev->mode = pi;

  HIP_CALL_ERET (NULL, hipDeviceGetAttribute,
		  &pi, hipDeviceAttributeIntegrated, dev);
  hip_dev->mkern = pi;

  HIP_CALL_ERET (NULL, hipDeviceGetAttribute,
		  &pi, hipDeviceAttributeClockRate, dev);
  hip_dev->clock_khz = pi;

  HIP_CALL_ERET (NULL, hipDeviceGetAttribute,
		  &pi, hipDeviceAttributeMultiprocessorCount, dev);
  hip_dev->num_sms = pi;

  HIP_CALL_ERET (NULL, hipDeviceGetAttribute,
		  &pi, hipDeviceAttributeMaxRegistersPerBlock, dev);
  hip_dev->regs_per_block = pi;

  /* CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR is defined only
     in CUDA 6.0 and newer.  */
  r = HIP_CALL_NOCHECK (hipDeviceGetAttribute, &pi,
			 CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
			 dev);
  /* Fallback: use limit of registers per block, which is usually equal.  */
  if (r == hipErrorInvalidValue)
    pi = hip_dev->regs_per_block;
  else if (r != hipSuccess)
    {
      GOMP_PLUGIN_error ("hipDeviceGetAttribute error: %s", hip_error (r));
      return NULL;
    }
  hip_dev->regs_per_sm = pi;

  HIP_CALL_ERET (NULL, hipDeviceGetAttribute,
		  &pi, hipDeviceAttributeWarpSize, dev);
  if (pi != 64)
    {
      GOMP_PLUGIN_error ("Only warp size 64 is supported");
      return NULL;
    }
  hip_dev->warp_size = pi;

  HIP_CALL_ERET (NULL, hipDeviceGetAttribute, &pi,
		  hipDeviceAttributeMaxThreadsPerBlock, dev);
  hip_dev->max_threads_per_block = pi;

  HIP_CALL_ERET (NULL, hipDeviceGetAttribute, &pi,
		  hipDeviceAttributeMaxThreadsPerMultiProcessor, dev);
  hip_dev->max_threads_per_multiprocessor = pi;

  r = HIP_CALL_NOCHECK (hipDeviceGetAttribute, &async_engines,
			 hipDeviceAttributeAsyncEngineCount, dev);
  if (r != hipSuccess)
    async_engines = 1;

  for (int i = 0; i != GOMP_DIM_MAX; i++)
    hip_dev->default_dims[i] = 0;

  HIP_CALL_ERET (NULL, hipDeviceGetName, hip_dev->name, sizeof hip_dev->name,
		  dev);

  hip_dev->images = NULL;
  pthread_mutex_init (&hip_dev->image_lock, NULL);

  hip_dev->free_blocks = NULL;
  pthread_mutex_init (&hip_dev->free_blocks_lock, NULL);

  hip_dev->omp_stacks.ptr = 0;
  hip_dev->omp_stacks.size = 0;
  pthread_mutex_init (&hip_dev->omp_stacks.lock, NULL);

  return hip_dev;
}

static bool
hip_close_device (struct hip_device *hip_dev)
{
  if (!hip_dev)
    return true;

  for (struct hip_free_block *b = hip_dev->free_blocks; b;)
    {
      struct hip_free_block *b_next = b->next;
      HIP_CALL (hipFree, (hipDeviceptr_t) b->ptr);
      free (b);
      b = b_next;
    }

  pthread_mutex_destroy (&hip_dev->free_blocks_lock);
  pthread_mutex_destroy (&hip_dev->image_lock);

  pthread_mutex_destroy (&hip_dev->omp_stacks.lock);

  if (hip_dev->omp_stacks.ptr)
    HIP_CALL (hipFree, hip_dev->omp_stacks.ptr);

  if (!hip_dev->ctx_shared)
    HIP_CALL (hipCtxDestroy, hip_dev->ctx);

  free (hip_dev);
  return true;
}

static int
hip_get_num_devices (void)
{
  int n;

  /* This function will be called before the plugin has been initialized in
     order to enumerate available devices, but HIP API routines can't be used
     until hipInit has been called.  Just call it now (but don't yet do any
     further initialization).  */
  if (instantiated_devices == 0)
    {
      if (!init_hip_lib ())
	return 0;
      hipError_t r = HIP_CALL_NOCHECK (hipInit, 0);
      /* This is not an error: e.g. we may have HIP libraries installed but
         no devices available.  */
      if (r != hipSuccess)
	{
	  GOMP_PLUGIN_debug (0, "Disabling hipamd offloading; hipInit: %s\n",
			     hip_error (r));
	  return 0;
	}
    }

  HIP_CALL_ERET (-1, hipGetDeviceCount, &n);
  return n;
}

static void
notify_var (const char *var_name, const char *env_var)
{
  if (env_var == NULL)
    GOMP_PLUGIN_debug (0, "%s: <Not defined>\n", var_name);
  else
    GOMP_PLUGIN_debug (0, "%s: '%s'\n", var_name, env_var);
}


static void
process_GOMP_HIP_JIT (intptr_t *gomp_hip_o)
{
//  const char *var_name = "GOMP_HIP_JIT";
//  const char *env_var = secure_getenv (var_name);
//  notify_var (var_name, env_var);
//
//  if (env_var == NULL)
//    return;
//
//  const char *c = env_var;
//  while (*c != '\0')
//    {
//      while (*c == ' ')
//	c++;
//
//      if (c[0] == '-' && c[1] == 'O'
//	  && '0' <= c[2] && c[2] <= '4'
//	  && (c[3] == '\0' || c[3] == ' '))
//	{
//	  *gomp_hip_o = c[2] - '0';
//	  c += 3;
//	  continue;
//	}
//
//      GOMP_PLUGIN_error ("Error parsing %s", var_name);
//      break;
//    }
}

static bool
link_hip (hipModule_t *module, const struct targ_hip_obj *hip_objs,
	  unsigned num_objs)
{
//  hipJitOption opts[7];
//  void *optvals[7];
//  float elapsed = 0.0;
//  char elog[1024];
//  char ilog[16384];
//  CUlinkState linkstate;
//  hipError_t r;
//  void *linkout;
//  size_t linkoutsize __attribute__ ((unused));
//
//  opts[0] = hipJitOptionWallTime;
//  optvals[0] = &elapsed;
//
//  opts[1] = hipJitOptionInfoLogBuffer;
//  optvals[1] = &ilog[0];
//
//  opts[2] = hipJitOptionInfoLogBufferSizeBytes;
//  optvals[2] = (void *) sizeof ilog;
//
//  opts[3] = hipJitOptionErrorLogBuffer;
//  optvals[3] = &elog[0];
//
//  opts[4] = hipJitOptionErrorLogBufferSizeBytes;
//  optvals[4] = (void *) sizeof elog;
//
//  opts[5] = hipJitOptionLogVerbose;
//  optvals[5] = (void *) 1;
//
//  static intptr_t gomp_hip_o = -1;
//
//  static bool init_done = false;
//  if (!init_done)
//    {
//      process_GOMP_HIP_JIT (&gomp_hip_o);
//      init_done = true;
//  }
//
//  int nopts = 6;
//  if (gomp_hip_o != -1)
//    {
//      opts[nopts] = hipJitOptionOptimizationLevel;
//      optvals[nopts] = (void *) gomp_hip_o;
//      nopts++;
//    }
//
//  if (HIP_CALL_EXISTS (cuLinkCreate_v2))
//    HIP_CALL (cuLinkCreate_v2, nopts, opts, optvals, &linkstate);
//  else
//    HIP_CALL (cuLinkCreate, nopts, opts, optvals, &linkstate);
//
//  for (; num_objs--; hip_objs++)
//    {
//      /* cuLinkAddData's 'data' argument erroneously omits the const
//	 qualifier.  */
//      GOMP_PLUGIN_debug (0, "Loading:\n---\n%s\n---\n", hip_objs->code);
//      if (HIP_CALL_EXISTS (cuLinkAddData_v2))
//	r = HIP_CALL_NOCHECK (cuLinkAddData_v2, linkstate, CU_JIT_INPUT_PTX,
//			       (char *) hip_objs->code, hip_objs->size,
//			       0, 0, 0, 0);
//      else
//	r = HIP_CALL_NOCHECK (cuLinkAddData, linkstate, CU_JIT_INPUT_PTX,
//			       (char *) hip_objs->code, hip_objs->size,
//			       0, 0, 0, 0);
//      if (r != hipSuccess)
//	{
//	  GOMP_PLUGIN_error ("Link error log %s\n", &elog[0]);
//	  GOMP_PLUGIN_error ("cuLinkAddData (hip_code) error: %s",
//			     hip_error (r));
//	  return false;
//	}
//    }
//
//  GOMP_PLUGIN_debug (0, "Linking\n");
//  r = HIP_CALL_NOCHECK (cuLinkComplete, linkstate, &linkout, &linkoutsize);
//
//  GOMP_PLUGIN_debug (0, "Link complete: %fms\n", elapsed);
//  GOMP_PLUGIN_debug (0, "Link log %s\n", &ilog[0]);
//
//  if (r != hipSuccess)
//    {
//      GOMP_PLUGIN_error ("Link error log %s\n", &elog[0]);
//      GOMP_PLUGIN_error ("cuLinkComplete error: %s", hip_error (r));
//      return false;
//    }
//
//  HIP_CALL (hipModuleLoadData, module, linkout);
//  HIP_CALL (cuLinkDestroy, linkstate);
//  return true;
}

static void
hip_exec (void (*fn), size_t mapnum, void **hostaddrs, void **devaddrs,
	    unsigned *dims, void *targ_mem_desc,
	    hipDeviceptr_t dp, hipStream_t stream)
{
//  struct targ_fn_descriptor *targ_fn = (struct targ_fn_descriptor *) fn;
//  hipFunction_t function;
//  int i;
//  void *kargs[1];
//  struct hip_thread *nvthd = hip_thread ();
//  int warp_size = nvthd->hip_dev->warp_size;
//
//  function = targ_fn->fn;
//
//  /* Initialize the launch dimensions.  Typically this is constant,
//     provided by the device compiler, but we must permit runtime
//     values.  */
//  int seen_zero = 0;
//  for (i = 0; i != GOMP_DIM_MAX; i++)
//    {
//      if (targ_fn->launch->dim[i])
//       dims[i] = targ_fn->launch->dim[i];
//      if (!dims[i])
//       seen_zero = 1;
//    }
//
//  if (seen_zero)
//    {
//      pthread_mutex_lock (&hip_dev_lock);
//
//      static int gomp_openacc_dims[GOMP_DIM_MAX];
//      if (!gomp_openacc_dims[0])
//	{
//	  /* See if the user provided GOMP_OPENACC_DIM environment
//	     variable to specify runtime defaults.  */
//	  for (int i = 0; i < GOMP_DIM_MAX; ++i)
//	    gomp_openacc_dims[i] = GOMP_PLUGIN_acc_default_dim (i);
//	}
//
//      if (!nvthd->hip_dev->default_dims[0])
//	{
//	  int default_dims[GOMP_DIM_MAX];
//	  for (int i = 0; i < GOMP_DIM_MAX; ++i)
//	    default_dims[i] = gomp_openacc_dims[i];
//
//	  int gang, worker, vector;
//	  {
//	    int block_size = nvthd->hip_dev->max_threads_per_block;
//	    int cpu_size = nvthd->hip_dev->max_threads_per_multiprocessor;
//	    int dev_size = nvthd->hip_dev->num_sms;
//	    GOMP_PLUGIN_debug (0, " warp_size=%d, block_size=%d,"
//			       " dev_size=%d, cpu_size=%d\n",
//			       warp_size, block_size, dev_size, cpu_size);
//
//	    gang = (cpu_size / block_size) * dev_size;
//	    worker = block_size / warp_size;
//	    vector = warp_size;
//	  }
//
//	  /* There is no upper bound on the gang size.  The best size
//	     matches the hardware configuration.  Logical gangs are
//	     scheduled onto physical hardware.  To maximize usage, we
//	     should guess a large number.  */
//	  if (default_dims[GOMP_DIM_GANG] < 1)
//	    default_dims[GOMP_DIM_GANG] = gang ? gang : 1024;
//	  /* The worker size must not exceed the hardware.  */
//	  if (default_dims[GOMP_DIM_WORKER] < 1
//	      || (default_dims[GOMP_DIM_WORKER] > worker && gang))
//	    default_dims[GOMP_DIM_WORKER] = worker;
//	  /* The vector size must exactly match the hardware.  */
//	  if (default_dims[GOMP_DIM_VECTOR] < 1
//	      || (default_dims[GOMP_DIM_VECTOR] != vector && gang))
//	    default_dims[GOMP_DIM_VECTOR] = vector;
//
//	  GOMP_PLUGIN_debug (0, " default dimensions [%d,%d,%d]\n",
//			     default_dims[GOMP_DIM_GANG],
//			     default_dims[GOMP_DIM_WORKER],
//			     default_dims[GOMP_DIM_VECTOR]);
//
//	  for (i = 0; i != GOMP_DIM_MAX; i++)
//	    nvthd->hip_dev->default_dims[i] = default_dims[i];
//	}
//      pthread_mutex_unlock (&hip_dev_lock);
//
//      {
//	bool default_dim_p[GOMP_DIM_MAX];
//	for (i = 0; i != GOMP_DIM_MAX; i++)
//	  default_dim_p[i] = !dims[i];
//
//	if (!HIP_CALL_EXISTS (hipOccupancyMaxPotentialBlockSize))
//	  {
//	    for (i = 0; i != GOMP_DIM_MAX; i++)
//	      if (default_dim_p[i])
//		dims[i] = nvthd->hip_dev->default_dims[i];
//
//	    if (default_dim_p[GOMP_DIM_VECTOR])
//	      dims[GOMP_DIM_VECTOR]
//		= MIN (dims[GOMP_DIM_VECTOR],
//		       (targ_fn->max_threads_per_block / warp_size
//			* warp_size));
//
//	    if (default_dim_p[GOMP_DIM_WORKER])
//	      dims[GOMP_DIM_WORKER]
//		= MIN (dims[GOMP_DIM_WORKER],
//		       targ_fn->max_threads_per_block / dims[GOMP_DIM_VECTOR]);
//	  }
//	else
//	  {
//	    /* Handle the case that the compiler allows the runtime to choose
//	       the vector-length conservatively, by ignoring
//	       gomp_openacc_dims[GOMP_DIM_VECTOR].  TODO: actually handle
//	       it.  */
//	    int vectors = 0;
//	    /* TODO: limit gomp_openacc_dims[GOMP_DIM_WORKER] such that that
//	       gomp_openacc_dims[GOMP_DIM_WORKER] * actual_vectors does not
//	       exceed targ_fn->max_threads_per_block. */
//	    int workers = gomp_openacc_dims[GOMP_DIM_WORKER];
//	    int gangs = gomp_openacc_dims[GOMP_DIM_GANG];
//	    int grids, blocks;
//
//	    HIP_CALL_ASSERT (hipOccupancyMaxPotentialBlockSize, &grids,
//			      &blocks, function, NULL, 0,
//			      dims[GOMP_DIM_WORKER] * dims[GOMP_DIM_VECTOR]);
//	    GOMP_PLUGIN_debug (0, "hipOccupancyMaxPotentialBlockSize: "
//			       "grid = %d, block = %d\n", grids, blocks);
//
//	    /* Keep the num_gangs proportional to the block size.  In
//	       the case were a block size is limited by shared-memory
//	       or the register file capacity, the runtime will not
//	       excessively over assign gangs to the multiprocessor
//	       units if their state is going to be swapped out even
//	       more than necessary. The constant factor 2 is there to
//	       prevent threads from idling when there is insufficient
//	       work for them.  */
//	    if (gangs == 0)
//	      gangs = 2 * grids * (blocks / warp_size);
//
//	    if (vectors == 0)
//	      vectors = warp_size;
//
//	    if (workers == 0)
//	      {
//		int actual_vectors = (default_dim_p[GOMP_DIM_VECTOR]
//				      ? vectors
//				      : dims[GOMP_DIM_VECTOR]);
//		workers = blocks / actual_vectors;
//		workers = MAX (workers, 1);
//		/* If we need a per-worker barrier ... .  */
//		if (actual_vectors > 32)
//		  /* Don't use more barriers than available.  */
//		  workers = MIN (workers, 15);
//	      }
//
//	    for (i = 0; i != GOMP_DIM_MAX; i++)
//	      if (default_dim_p[i])
//		switch (i)
//		  {
//		  case GOMP_DIM_GANG: dims[i] = gangs; break;
//		  case GOMP_DIM_WORKER: dims[i] = workers; break;
//		  case GOMP_DIM_VECTOR: dims[i] = vectors; break;
//		  default: GOMP_PLUGIN_fatal ("invalid dim");
//		  }
//	  }
//      }
//    }
//
//  /* Check if the accelerator has sufficient hardware resources to
//     launch the offloaded kernel.  */
//  if (dims[GOMP_DIM_WORKER] * dims[GOMP_DIM_VECTOR]
//      > targ_fn->max_threads_per_block)
//    {
//      const char *msg
//	= ("The HIP accelerator has insufficient resources to launch '%s'"
//	   " with num_workers = %d and vector_length = %d"
//	   "; "
//	   "recompile the program with 'num_workers = x and vector_length = y'"
//	   " on that offloaded region or '-fopenacc-dim=:x:y' where"
//	   " x * y <= %d"
//	   ".\n");
//      GOMP_PLUGIN_fatal (msg, targ_fn->launch->fn, dims[GOMP_DIM_WORKER],
//			 dims[GOMP_DIM_VECTOR], targ_fn->max_threads_per_block);
//    }
//
//  /* Check if the accelerator has sufficient barrier resources to
//     launch the offloaded kernel.  */
//  if (dims[GOMP_DIM_WORKER] > 15 && dims[GOMP_DIM_VECTOR] > 32)
//    {
//      const char *msg
//	= ("The HIP accelerator has insufficient barrier resources to launch"
//	   " '%s' with num_workers = %d and vector_length = %d"
//	   "; "
//	   "recompile the program with 'num_workers = x' on that offloaded"
//	   " region or '-fopenacc-dim=:x:' where x <= 15"
//	   "; "
//	   "or, recompile the program with 'vector_length = 32' on that"
//	   " offloaded region or '-fopenacc-dim=::32'"
//	   ".\n");
//	GOMP_PLUGIN_fatal (msg, targ_fn->launch->fn, dims[GOMP_DIM_WORKER],
//			   dims[GOMP_DIM_VECTOR]);
//    }
//
//  GOMP_PLUGIN_debug (0, "  %s: kernel %s: launch"
//		     " gangs=%u, workers=%u, vectors=%u\n",
//		     __FUNCTION__, targ_fn->launch->fn, dims[GOMP_DIM_GANG],
//		     dims[GOMP_DIM_WORKER], dims[GOMP_DIM_VECTOR]);
//
//  // OpenACC		HIP
//  //
//  // num_gangs		nctaid.x
//  // num_workers	ntid.y
//  // vector length	ntid.x
//
//  struct goacc_thread *thr = GOMP_PLUGIN_goacc_thread ();
//  acc_prof_info *prof_info = thr->prof_info;
//  acc_event_info enqueue_launch_event_info;
//  acc_api_info *api_info = thr->api_info;
//  bool profiling_p = __builtin_expect (prof_info != NULL, false);
//  if (profiling_p)
//    {
//      prof_info->event_type = acc_ev_enqueue_launch_start;
//
//      enqueue_launch_event_info.launch_event.event_type
//	= prof_info->event_type;
//      enqueue_launch_event_info.launch_event.valid_bytes
//	= _ACC_LAUNCH_EVENT_INFO_VALID_BYTES;
//      enqueue_launch_event_info.launch_event.parent_construct
//	= acc_construct_parallel;
//      enqueue_launch_event_info.launch_event.implicit = 1;
//      enqueue_launch_event_info.launch_event.tool_info = NULL;
//      enqueue_launch_event_info.launch_event.kernel_name = targ_fn->launch->fn;
//      enqueue_launch_event_info.launch_event.num_gangs
//	= dims[GOMP_DIM_GANG];
//      enqueue_launch_event_info.launch_event.num_workers
//	= dims[GOMP_DIM_WORKER];
//      enqueue_launch_event_info.launch_event.vector_length
//	= dims[GOMP_DIM_VECTOR];
//
//      api_info->device_api = acc_device_api_hip;
//
//      GOMP_PLUGIN_goacc_profiling_dispatch (prof_info, &enqueue_launch_event_info,
//					    api_info);
//    }
//
//  kargs[0] = &dp;
//  HIP_CALL_ASSERT (hipModuleLaunchKernel, function,
//		    dims[GOMP_DIM_GANG], 1, 1,
//		    dims[GOMP_DIM_VECTOR], dims[GOMP_DIM_WORKER], 1,
//		    0, stream, kargs, 0);
//
//  if (profiling_p)
//    {
//      prof_info->event_type = acc_ev_enqueue_launch_end;
//      enqueue_launch_event_info.launch_event.event_type
//	= prof_info->event_type;
//      GOMP_PLUGIN_goacc_profiling_dispatch (prof_info, &enqueue_launch_event_info,
//					    api_info);
//    }
//
//  GOMP_PLUGIN_debug (0, "  %s: kernel %s: finished\n", __FUNCTION__,
//		     targ_fn->launch->fn);
}

void * openacc_get_current_hip_context (void);

static void
goacc_profiling_acc_ev_alloc (struct goacc_thread *thr, void *dp, size_t s)
{
  acc_prof_info *prof_info = thr->prof_info;
  acc_event_info data_event_info;
  acc_api_info *api_info = thr->api_info;

  prof_info->event_type = acc_ev_alloc;

  data_event_info.data_event.event_type = prof_info->event_type;
  data_event_info.data_event.valid_bytes = _ACC_DATA_EVENT_INFO_VALID_BYTES;
  data_event_info.data_event.parent_construct = acc_construct_parallel;
  data_event_info.data_event.implicit = 1;
  data_event_info.data_event.tool_info = NULL;
  data_event_info.data_event.var_name = NULL;
  data_event_info.data_event.bytes = s;
  data_event_info.data_event.host_ptr = NULL;
  data_event_info.data_event.device_ptr = dp;

  api_info->device_api = acc_device_api_hip;

  GOMP_PLUGIN_goacc_profiling_dispatch (prof_info, &data_event_info, api_info);
}

/* Free the cached soft-stacks block if it is above the SOFTSTACK_CACHE_LIMIT
   size threshold, or if FORCE is true.  */

static void
hip_stacks_free (struct hip_device *hip_dev, bool force)
{
  pthread_mutex_lock (&hip_dev->omp_stacks.lock);
  if (hip_dev->omp_stacks.ptr
      && (force || hip_dev->omp_stacks.size > SOFTSTACK_CACHE_LIMIT))
    {
      hipError_t r = HIP_CALL_NOCHECK (hipFree, hip_dev->omp_stacks.ptr);
      if (r != hipSuccess)
	GOMP_PLUGIN_fatal ("hipFree error: %s", hip_error (r));
      hip_dev->omp_stacks.ptr = 0;
      hip_dev->omp_stacks.size = 0;
    }
  pthread_mutex_unlock (&hip_dev->omp_stacks.lock);
}

static void *
hip_alloc (size_t s, bool suppress_errors)
{
  hipDeviceptr_t d;

  hipError_t r = HIP_CALL_NOCHECK (hipMalloc, &d, s);
  if (suppress_errors && r == hipErrorOutOfMemory)
    return NULL;
  else if (r != hipSuccess)
    {
      GOMP_PLUGIN_error ("hip_alloc error: %s", hip_error (r));
      return NULL;
    }

  /* NOTE: We only do profiling stuff if the memory allocation succeeds.  */
  struct goacc_thread *thr = GOMP_PLUGIN_goacc_thread ();
  bool profiling_p
    = __builtin_expect (thr != NULL && thr->prof_info != NULL, false);
  if (profiling_p)
    goacc_profiling_acc_ev_alloc (thr, (void *) d, s);

  return (void *) d;
}

static void
goacc_profiling_acc_ev_free (struct goacc_thread *thr, void *p)
{
  acc_prof_info *prof_info = thr->prof_info;
  acc_event_info data_event_info;
  acc_api_info *api_info = thr->api_info;

  prof_info->event_type = acc_ev_free;

  data_event_info.data_event.event_type = prof_info->event_type;
  data_event_info.data_event.valid_bytes = _ACC_DATA_EVENT_INFO_VALID_BYTES;
  data_event_info.data_event.parent_construct = acc_construct_parallel;
  data_event_info.data_event.implicit = 1;
  data_event_info.data_event.tool_info = NULL;
  data_event_info.data_event.var_name = NULL;
  data_event_info.data_event.bytes = -1;
  data_event_info.data_event.host_ptr = NULL;
  data_event_info.data_event.device_ptr = p;

  api_info->device_api = acc_device_api_hip;

  GOMP_PLUGIN_goacc_profiling_dispatch (prof_info, &data_event_info, api_info);
}

static bool
hip_free (void *p, struct hip_device *hip_dev)
{
  hipDeviceptr_t pb;
  size_t ps;

  hipError_t r = HIP_CALL_NOCHECK (hipMemGetAddressRange, &pb, &ps,
				  (hipDeviceptr_t) p);
  if (r == HIP_ERROR_NOT_PERMITTED)
    {
      /* We assume that this error indicates we are in a HIP callback context,
	 where all CUDA calls are not allowed (see hipStreamAddCallback
	 documentation for description). Arrange to free this piece of device
	 memory later.  */
      struct hip_free_block *n
	= GOMP_PLUGIN_malloc (sizeof (struct hip_free_block));
      n->ptr = p;
      pthread_mutex_lock (&hip_dev->free_blocks_lock);
      n->next = hip_dev->free_blocks;
      hip_dev->free_blocks = n;
      pthread_mutex_unlock (&hip_dev->free_blocks_lock);
      return true;
    }
  else if (r != hipSuccess)
    {
      GOMP_PLUGIN_error ("hipMemGetAddressRange error: %s", hip_error (r));
      return false;
    }
  if ((hipDeviceptr_t) p != pb)
    {
      GOMP_PLUGIN_error ("invalid device address");
      return false;
    }

  HIP_CALL (hipFree, (hipDeviceptr_t) p);
  struct goacc_thread *thr = GOMP_PLUGIN_goacc_thread ();
  bool profiling_p
    = __builtin_expect (thr != NULL && thr->prof_info != NULL, false);
  if (profiling_p)
    goacc_profiling_acc_ev_free (thr, p);

  return true;
}

static void *
hip_get_current_hip_device (void)
{
  struct hip_thread *nvthd = hip_thread ();

  if (!nvthd || !nvthd->hip_dev)
    return NULL;

  return &nvthd->hip_dev->dev;
}

static void *
hip_get_current_hip_context (void)
{
  struct hip_thread *nvthd = hip_thread ();

  if (!nvthd || !nvthd->hip_dev)
    return NULL;

  return nvthd->hip_dev->ctx;
}

/* Plugin entry points.  */

const char *
GOMP_OFFLOAD_get_name (void)
{
  return "hipamd";
}

unsigned int
GOMP_OFFLOAD_get_caps (void)
{
  return GOMP_OFFLOAD_CAP_OPENACC_200 | GOMP_OFFLOAD_CAP_OPENMP_400;
}

int
GOMP_OFFLOAD_get_type (void)
{
  return OFFLOAD_TARGET_TYPE_NVIDIA_PTX;
}

int
GOMP_OFFLOAD_get_num_devices (void)
{
  return hip_get_num_devices ();
}

bool
GOMP_OFFLOAD_init_device (int n)
{
  struct hip_device *dev;

  pthread_mutex_lock (&hip_dev_lock);

  if (!hip_init () || hip_devices[n] != NULL)
    {
      pthread_mutex_unlock (&hip_dev_lock);
      return false;
    }

  dev = hip_open_device (n);
  if (dev)
    {
      hip_devices[n] = dev;
      instantiated_devices++;
    }

  pthread_mutex_unlock (&hip_dev_lock);

  return dev != NULL;
}

bool
GOMP_OFFLOAD_fini_device (int n)
{
  pthread_mutex_lock (&hip_dev_lock);

  if (hip_devices[n] != NULL)
    {
      if (!hip_attach_host_thread_to_device (n)
	  || !hip_close_device (hip_devices[n]))
	{
	  pthread_mutex_unlock (&hip_dev_lock);
	  return false;
	}
      hip_devices[n] = NULL;
      instantiated_devices--;
    }

  if (instantiated_devices == 0)
    {
      free (hip_devices);
      hip_devices = NULL;
    }

  pthread_mutex_unlock (&hip_dev_lock);
  return true;
}

/* Return the libgomp version number we're compatible with.  There is
   no requirement for cross-version compatibility.  */

unsigned
GOMP_OFFLOAD_version (void)
{
  return GOMP_VERSION;
}

/* Initialize __nvptx_clocktick, if present in MODULE.  */

//static void
//hip_set_clocktick (hipModule_t module, struct hip_device *dev)
//{
//  hipDeviceptr_t dptr;
//  hipError_t r = HIP_CALL_NOCHECK (hipModuleGetGlobal, &dptr, NULL,
//				  module, "__nvptx_clocktick");
//  if (r == hipErrorNotFound)
//    return;
//  if (r != hipSuccess)
//    GOMP_PLUGIN_fatal ("hipModuleGetGlobal error: %s", hip_error (r));
//  double __nvptx_clocktick = 1e-3 / dev->clock_khz;
//  r = HIP_CALL_NOCHECK (hipMemcpyHtoD, dptr, &__nvptx_clocktick,
//			 sizeof (__nvptx_clocktick));
//  if (r != hipSuccess)
//    GOMP_PLUGIN_fatal ("hipMemcpyHtoD error: %s", hip_error (r));
//}

/* Load the (partial) program described by TARGET_DATA to device
   number ORD.  Allocate and return TARGET_TABLE.  */

int
GOMP_OFFLOAD_load_image (int ord, unsigned version, const void *target_data,
			 struct addr_pair **target_table)
{
//  hipModule_t module;
//  const char *const *var_names;
//  const struct targ_fn_launch *fn_descs;
//  unsigned int fn_entries, var_entries, other_entries, i, j;
//  struct targ_fn_descriptor *targ_fns;
//  struct addr_pair *targ_tbl;
//  const hip_tdata_t *img_header = (const hip_tdata_t *) target_data;
//  struct hip_image_data *new_image;
//  struct hip_device *dev;
//
//  if (GOMP_VERSION_DEV (version) > GOMP_VERSION_NVIDIA_PTX)
//    {
//      GOMP_PLUGIN_error ("Offload data incompatible with HIP plugin"
//			 " (expected %u, received %u)",
//			 GOMP_VERSION_NVIDIA_PTX, GOMP_VERSION_DEV (version));
//      return -1;
//    }
//
//  if (!hip_attach_host_thread_to_device (ord)
//      || !link_ptx (&module, img_header->hip_objs, img_header->hip_num))
//    return -1;
//
//  dev = hip_devices[ord];
//
//  /* The mkoffload utility emits a struct of pointers/integers at the
//     start of each offload image.  The array of kernel names and the
//     functions addresses form a one-to-one correspondence.  */
//
//  var_entries = img_header->var_num;
//  var_names = img_header->var_names;
//  fn_entries = img_header->fn_num;
//  fn_descs = img_header->fn_descs;
//
//  /* Currently, the only other entry kind is 'device number'.  */
//  other_entries = 1;
//
//  targ_tbl = GOMP_PLUGIN_malloc (sizeof (struct addr_pair)
//				 * (fn_entries + var_entries + other_entries));
//  targ_fns = GOMP_PLUGIN_malloc (sizeof (struct targ_fn_descriptor)
//				 * fn_entries);
//
//  *target_table = targ_tbl;
//
//  new_image = GOMP_PLUGIN_malloc (sizeof (struct hip_image_data));
//  new_image->target_data = target_data;
//  new_image->module = module;
//  new_image->fns = targ_fns;
//
//  pthread_mutex_lock (&dev->image_lock);
//  new_image->next = dev->images;
//  dev->images = new_image;
//  pthread_mutex_unlock (&dev->image_lock);
//
//  for (i = 0; i < fn_entries; i++, targ_fns++, targ_tbl++)
//    {
//      hipFunction_t function;
//      int nregs, mthrs;
//
//      HIP_CALL_ERET (-1, hipModuleGetFunction, &function, module,
//		      fn_descs[i].fn);
//      HIP_CALL_ERET (-1, hipFuncGetAttribute, &nregs,
//		      HIP_FUNC_ATTRIBUTE_NUM_REGS, function);
//      HIP_CALL_ERET (-1, hipFuncGetAttribute, &mthrs,
//		      HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function);
//
//      targ_fns->fn = function;
//      targ_fns->launch = &fn_descs[i];
//      targ_fns->regs_per_thread = nregs;
//      targ_fns->max_threads_per_block = mthrs;
//
//      targ_tbl->start = (uintptr_t) targ_fns;
//      targ_tbl->end = targ_tbl->start + 1;
//    }
//
//  for (j = 0; j < var_entries; j++, targ_tbl++)
//    {
//      hipDeviceptr_t var;
//      size_t bytes;
//
//      HIP_CALL_ERET (-1, hipModuleGetGlobal,
//		      &var, &bytes, module, var_names[j]);
//
//      targ_tbl->start = (uintptr_t) var;
//      targ_tbl->end = targ_tbl->start + bytes;
//    }
//
//  hipDeviceptr_t device_num_varptr;
//  size_t device_num_varsize;
//  hipError_t r = HIP_CALL_NOCHECK (hipModuleGetGlobal, &device_num_varptr,
//				  &device_num_varsize, module,
//				  STRINGX (GOMP_DEVICE_NUM_VAR));
//  if (r == hipSuccess)
//    {
//      targ_tbl->start = (uintptr_t) device_num_varptr;
//      targ_tbl->end = (uintptr_t) (device_num_varptr + device_num_varsize);
//    }
//  else
//    /* The 'GOMP_DEVICE_NUM_VAR' variable was not in this image.  */
//    targ_tbl->start = targ_tbl->end = 0;
//  targ_tbl++;
//
//  hip_set_clocktick (module, dev);
//
//  return fn_entries + var_entries + other_entries;
}

/* Unload the program described by TARGET_DATA.  DEV_DATA is the
   function descriptors allocated by G_O_load_image.  */

bool
GOMP_OFFLOAD_unload_image (int ord, unsigned version, const void *target_data)
{
//  struct hip_image_data *image, **prev_p;
//  struct hip_device *dev = hip_devices[ord];
//
//  if (GOMP_VERSION_DEV (version) > GOMP_VERSION_NVIDIA_PTX)
//    {
//      GOMP_PLUGIN_error ("Offload data incompatible with HIP plugin"
//			 " (expected %u, received %u)",
//			 GOMP_VERSION_NVIDIA_PTX, GOMP_VERSION_DEV (version));
//      return false;
//    }
//
//  bool ret = true;
//  pthread_mutex_lock (&dev->image_lock);
//  for (prev_p = &dev->images; (image = *prev_p) != 0; prev_p = &image->next)
//    if (image->target_data == target_data)
//      {
//	*prev_p = image->next;
//	if (HIP_CALL_NOCHECK (hipModuleUnload, image->module) != hipSuccess)
//	  ret = false;
//	free (image->fns);
//	free (image);
//	break;
//      }
//  pthread_mutex_unlock (&dev->image_lock);
//  return ret;
}

void *
GOMP_OFFLOAD_alloc (int ord, size_t size)
{
  if (!hip_attach_host_thread_to_device (ord))
    return NULL;

  struct hip_device *hip_dev = hip_devices[ord];
  struct hip_free_block *blocks, *tmp;

  pthread_mutex_lock (&hip_dev->free_blocks_lock);
  blocks = hip_dev->free_blocks;
  hip_dev->free_blocks = NULL;
  pthread_mutex_unlock (&hip_dev->free_blocks_lock);

  hip_stacks_free (hip_dev, false);

  while (blocks)
    {
      tmp = blocks->next;
      hip_free (blocks->ptr, hip_dev);
      free (blocks);
      blocks = tmp;
    }

  void *d = hip_alloc (size, true);
  if (d)
    return d;
  else
    {
      /* Memory allocation failed.  Try freeing the stacks block, and
	 retrying.  */
      hip_stacks_free (hip_dev, true);
      return hip_alloc (size, false);
    }
}

bool
GOMP_OFFLOAD_free (int ord, void *ptr)
{
  return (hip_attach_host_thread_to_device (ord)
	  && hip_free (ptr, hip_devices[ord]));
}

void
GOMP_OFFLOAD_openacc_exec (void (*fn) (void *), size_t mapnum,
			   void **hostaddrs, void **devaddrs,
			   unsigned *dims, void *targ_mem_desc)
{
//  GOMP_PLUGIN_debug (0, "  %s: prepare mappings\n", __FUNCTION__);
//
//  struct goacc_thread *thr = GOMP_PLUGIN_goacc_thread ();
//  acc_prof_info *prof_info = thr->prof_info;
//  acc_event_info data_event_info;
//  acc_api_info *api_info = thr->api_info;
//  bool profiling_p = __builtin_expect (prof_info != NULL, false);
//
//  void **hp = NULL;
//  hipDeviceptr_t dp = 0;
//
//  if (mapnum > 0)
//    {
//      size_t s = mapnum * sizeof (void *);
//      hp = alloca (s);
//      for (int i = 0; i < mapnum; i++)
//	hp[i] = (devaddrs[i] ? devaddrs[i] : hostaddrs[i]);
//      HIP_CALL_ASSERT (hipMalloc, &dp, s);
//      if (profiling_p)
//	goacc_profiling_acc_ev_alloc (thr, (void *) dp, s);
//    }
//
//  /* Copy the (device) pointers to arguments to the device (dp and hp might in
//     fact have the same value on a unified-memory system).  */
//  if (mapnum > 0)
//    {
//      if (profiling_p)
//	{
//	  prof_info->event_type = acc_ev_enqueue_upload_start;
//
//	  data_event_info.data_event.event_type = prof_info->event_type;
//	  data_event_info.data_event.valid_bytes
//	    = _ACC_DATA_EVENT_INFO_VALID_BYTES;
//	  data_event_info.data_event.parent_construct
//	    = acc_construct_parallel;
//	  data_event_info.data_event.implicit = 1; /* Always implicit.  */
//	  data_event_info.data_event.tool_info = NULL;
//	  data_event_info.data_event.var_name = NULL;
//	  data_event_info.data_event.bytes = mapnum * sizeof (void *);
//	  data_event_info.data_event.host_ptr = hp;
//	  data_event_info.data_event.device_ptr = (const void *) dp;
//
//	  api_info->device_api = acc_device_api_hip;
//
//	  GOMP_PLUGIN_goacc_profiling_dispatch (prof_info, &data_event_info,
//						api_info);
//	}
//      HIP_CALL_ASSERT (hipMemcpyHtoD, dp, (void *) hp,
//			mapnum * sizeof (void *));
//      if (profiling_p)
//	{
//	  prof_info->event_type = acc_ev_enqueue_upload_end;
//	  data_event_info.data_event.event_type = prof_info->event_type;
//	  GOMP_PLUGIN_goacc_profiling_dispatch (prof_info, &data_event_info,
//						api_info);
//	}
//    }
//
//  hip_exec (fn, mapnum, hostaddrs, devaddrs, dims, targ_mem_desc,
//	      dp, NULL);
//
//  hipError_t r = HIP_CALL_NOCHECK (hipStreamSynchronize, NULL);
//  const char *maybe_abort_msg = "(perhaps abort was called)";
//  if (r == hipErrorLaunchFailure)
//    GOMP_PLUGIN_fatal ("hipStreamSynchronize error: %s %s\n", hip_error (r),
//		       maybe_abort_msg);
//  else if (r != hipSuccess)
//    GOMP_PLUGIN_fatal ("hipStreamSynchronize error: %s", hip_error (r));
//
//  HIP_CALL_ASSERT (hipFree, dp);
//  if (profiling_p)
//    goacc_profiling_acc_ev_free (thr, (void *) dp);
}

static void
hip_free_argmem (void *ptr)
{
  void **block = (void **) ptr;
  hip_free (block[0], (struct hip_device *) block[1]);
  free (block);
}

void
GOMP_OFFLOAD_openacc_async_exec (void (*fn) (void *), size_t mapnum,
				 void **hostaddrs, void **devaddrs,
				 unsigned *dims, void *targ_mem_desc,
				 struct goacc_asyncqueue *aq)
{
//  GOMP_PLUGIN_debug (0, "  %s: prepare mappings\n", __FUNCTION__);
//
//  struct goacc_thread *thr = GOMP_PLUGIN_goacc_thread ();
//  acc_prof_info *prof_info = thr->prof_info;
//  acc_event_info data_event_info;
//  acc_api_info *api_info = thr->api_info;
//  bool profiling_p = __builtin_expect (prof_info != NULL, false);
//
//  void **hp = NULL;
//  hipDeviceptr_t dp = 0;
//  void **block = NULL;
//
//  if (mapnum > 0)
//    {
//      size_t s = mapnum * sizeof (void *);
//      block = (void **) GOMP_PLUGIN_malloc (2 * sizeof (void *) + s);
//      hp = block + 2;
//      for (int i = 0; i < mapnum; i++)
//	hp[i] = (devaddrs[i] ? devaddrs[i] : hostaddrs[i]);
//      HIP_CALL_ASSERT (hipMalloc, &dp, s);
//      if (profiling_p)
//	goacc_profiling_acc_ev_alloc (thr, (void *) dp, s);
//    }
//
//  /* Copy the (device) pointers to arguments to the device (dp and hp might in
//     fact have the same value on a unified-memory system).  */
//  if (mapnum > 0)
//    {
//      if (profiling_p)
//	{
//	  prof_info->event_type = acc_ev_enqueue_upload_start;
//
//	  data_event_info.data_event.event_type = prof_info->event_type;
//	  data_event_info.data_event.valid_bytes
//	    = _ACC_DATA_EVENT_INFO_VALID_BYTES;
//	  data_event_info.data_event.parent_construct
//	    = acc_construct_parallel;
//	  data_event_info.data_event.implicit = 1; /* Always implicit.  */
//	  data_event_info.data_event.tool_info = NULL;
//	  data_event_info.data_event.var_name = NULL;
//	  data_event_info.data_event.bytes = mapnum * sizeof (void *);
//	  data_event_info.data_event.host_ptr = hp;
//	  data_event_info.data_event.device_ptr = (const void *) dp;
//
//	  api_info->device_api = acc_device_api_hip;
//
//	  GOMP_PLUGIN_goacc_profiling_dispatch (prof_info, &data_event_info,
//						api_info);
//	}
//
//      HIP_CALL_ASSERT (hipMemcpyHtoDAsync, dp, (void *) hp,
//			mapnum * sizeof (void *), aq->hip_stream);
//      block[0] = (void *) dp;
//
//      struct hip_thread *nvthd =
//	(struct hip_thread *) GOMP_PLUGIN_acc_thread ();
//      block[1] = (void *) nvthd->hip_dev;
//
//      if (profiling_p)
//	{
//	  prof_info->event_type = acc_ev_enqueue_upload_end;
//	  data_event_info.data_event.event_type = prof_info->event_type;
//	  GOMP_PLUGIN_goacc_profiling_dispatch (prof_info, &data_event_info,
//						api_info);
//	}
//    }
//
//  hip_exec (fn, mapnum, hostaddrs, devaddrs, dims, targ_mem_desc,
//	      dp, aq->hip_stream);
//
//  if (mapnum > 0)
//    GOMP_OFFLOAD_openacc_async_queue_callback (aq, hip_free_argmem, block);
}

void *
GOMP_OFFLOAD_openacc_create_thread_data (int ord)
{
  struct hip_device *hip_dev;
  struct hip_thread *hipthd
    = GOMP_PLUGIN_malloc (sizeof (struct hip_thread));
  hipCtx_t thd_ctx;

  hip_dev = hip_devices[ord];

  assert (hip_dev);

  HIP_CALL_ASSERT (hipCtxGetCurrent, &thd_ctx);

  assert (hip_dev->ctx);

  if (!thd_ctx)
    HIP_CALL_ASSERT (hipCtxPushCurrent, hip_dev->ctx);

  hipthd->hip_dev = hip_dev;

  return (void *) hipthd;
}

void
GOMP_OFFLOAD_openacc_destroy_thread_data (void *data)
{
  free (data);
}

void *
GOMP_OFFLOAD_openacc_cuda_get_current_device (void)
{
  return hip_get_current_hip_device ();
}

void *
GOMP_OFFLOAD_openacc_cuda_get_current_context (void)
{
  return hip_get_current_hip_context ();
}

/* This returns a hipStream_t.  */
void *
GOMP_OFFLOAD_openacc_cuda_get_stream (struct goacc_asyncqueue *aq)
{
  return (void *) aq->hip_stream;
}

/* This takes a hipStream_t.  */
int
GOMP_OFFLOAD_openacc_cuda_set_stream (struct goacc_asyncqueue *aq, void *stream)
{
  if (aq->hip_stream)
    {
      HIP_CALL_ASSERT (hipStreamSynchronize, aq->hip_stream);
      HIP_CALL_ASSERT (hipStreamDestroy, aq->hip_stream);
    }

  aq->hip_stream = (hipStream_t) stream;
  return 1;
}

struct goacc_asyncqueue *
GOMP_OFFLOAD_openacc_async_construct (int device __attribute__((unused)))
{
  hipStream_t stream = NULL;
  HIP_CALL_ERET (NULL, hipStreamCreateWithFlags, &stream, hipStreamDefault);

  struct goacc_asyncqueue *aq
    = GOMP_PLUGIN_malloc (sizeof (struct goacc_asyncqueue));
  aq->hip_stream = stream;
  return aq;
}

bool
GOMP_OFFLOAD_openacc_async_destruct (struct goacc_asyncqueue *aq)
{
  HIP_CALL_ERET (false, hipStreamDestroy, aq->hip_stream);
  free (aq);
  return true;
}

int
GOMP_OFFLOAD_openacc_async_test (struct goacc_asyncqueue *aq)
{
  hipError_t r = HIP_CALL_NOCHECK (hipStreamQuery, aq->hip_stream);
  if (r == hipSuccess)
    return 1;
  if (r == hipErrorNotReady)
    return 0;

  GOMP_PLUGIN_error ("hipStreamQuery error: %s", hip_error (r));
  return -1;
}

bool
GOMP_OFFLOAD_openacc_async_synchronize (struct goacc_asyncqueue *aq)
{
  HIP_CALL_ERET (false, hipStreamSynchronize, aq->hip_stream);
  return true;
}

bool
GOMP_OFFLOAD_openacc_async_serialize (struct goacc_asyncqueue *aq1,
				      struct goacc_asyncqueue *aq2)
{
  hipEvent_t e;
  HIP_CALL_ERET (false, hipEventCreateWithFlags, &e, hipEventDisableTiming);
  HIP_CALL_ERET (false, hipEventRecord, e, aq1->hip_stream);
  HIP_CALL_ERET (false, hipStreamWaitEvent, aq2->hip_stream, e, 0);
  return true;
}

static void
hip_callback_wrapper (hipStream_t stream, hipError_t res, void *ptr)
{
  if (res != hipSuccess)
    GOMP_PLUGIN_fatal ("%s error: %s", __FUNCTION__, hip_error (res));
  struct hip_callback *cb = (struct hip_callback *) ptr;
  cb->fn (cb->ptr);
  free (ptr);
}

void
GOMP_OFFLOAD_openacc_async_queue_callback (struct goacc_asyncqueue *aq,
					   void (*callback_fn)(void *),
					   void *userptr)
{
  struct hip_callback *b = GOMP_PLUGIN_malloc (sizeof (*b));
  b->fn = callback_fn;
  b->ptr = userptr;
  b->aq = aq;
  HIP_CALL_ASSERT (hipStreamAddCallback, aq->hip_stream,
		    hip_callback_wrapper, (void *) b, 0);
}

static bool
hip_memcpy_sanity_check (const void *h, const void *d, size_t s)
{
  hipDeviceptr_t pb;
  size_t ps;
  if (!s)
    return true;
  if (!d)
    {
      GOMP_PLUGIN_error ("invalid device address");
      return false;
    }
  HIP_CALL (hipMemGetAddressRange, &pb, &ps, (hipDeviceptr_t) d);
  if (!pb)
    {
      GOMP_PLUGIN_error ("invalid device address");
      return false;
    }
  if (!h)
    {
      GOMP_PLUGIN_error ("invalid host address");
      return false;
    }
  if (d == h)
    {
      GOMP_PLUGIN_error ("invalid host or device address");
      return false;
    }
  if ((void *)(d + s) > (void *)(pb + ps))
    {
      GOMP_PLUGIN_error ("invalid size");
      return false;
    }
  return true;
}

bool
GOMP_OFFLOAD_host2dev (int ord, void *dst, const void *src, size_t n)
{
  if (!hip_attach_host_thread_to_device (ord)
      || !hip_memcpy_sanity_check (src, dst, n))
    return false;
  HIP_CALL (hipMemcpyHtoD, (hipDeviceptr_t) dst, src, n);
  return true;
}

bool
GOMP_OFFLOAD_dev2host (int ord, void *dst, const void *src, size_t n)
{
  if (!hip_attach_host_thread_to_device (ord)
      || !hip_memcpy_sanity_check (dst, src, n))
    return false;
  HIP_CALL (hipMemcpyDtoH, dst, (hipDeviceptr_t) src, n);
  return true;
}

bool
GOMP_OFFLOAD_dev2dev (int ord, void *dst, const void *src, size_t n)
{
  HIP_CALL (hipMemcpyDtoDAsync, (hipDeviceptr_t) dst, (hipDeviceptr_t) src, n, NULL);
  return true;
}

bool
GOMP_OFFLOAD_openacc_async_host2dev (int ord, void *dst, const void *src,
				     size_t n, struct goacc_asyncqueue *aq)
{
  if (!hip_attach_host_thread_to_device (ord)
      || !hip_memcpy_sanity_check (src, dst, n))
    return false;
  HIP_CALL (hipMemcpyHtoDAsync, (hipDeviceptr_t) dst, src, n, aq->hip_stream);
  return true;
}

bool
GOMP_OFFLOAD_openacc_async_dev2host (int ord, void *dst, const void *src,
				     size_t n, struct goacc_asyncqueue *aq)
{
  if (!hip_attach_host_thread_to_device (ord)
      || !hip_memcpy_sanity_check (dst, src, n))
    return false;
  HIP_CALL (hipMemcpyDtoHAsync, dst, (hipDeviceptr_t) src, n, aq->hip_stream);
  return true;
}

union goacc_property_value
GOMP_OFFLOAD_openacc_get_property (int n, enum goacc_property prop)
{
  union goacc_property_value propval = { .val = 0 };

  pthread_mutex_lock (&hip_dev_lock);

  if (n >= hip_get_num_devices () || n < 0 || hip_devices[n] == NULL)
    {
      pthread_mutex_unlock (&hip_dev_lock);
      return propval;
    }

  struct hip_device *hip_dev = hip_devices[n];
  switch (prop)
    {
    case GOACC_PROPERTY_MEMORY:
      {
	size_t total_mem;

	HIP_CALL_ERET (propval, hipDeviceTotalMem, &total_mem, hip_dev->dev);
	propval.val = total_mem;
      }
      break;
    case GOACC_PROPERTY_FREE_MEMORY:
      {
	size_t total_mem;
	size_t free_mem;
	hipDevice_t ctxdev;

	HIP_CALL_ERET (propval, hipCtxGetDevice, &ctxdev);
	if (hip_dev->dev == ctxdev)
	  HIP_CALL_ERET (propval, hipMemGetInfo, &free_mem, &total_mem);
	else if (hip_dev->ctx)
	  {
	    hipCtx_t old_ctx;

	    HIP_CALL_ERET (propval, hipCtxPushCurrent, hip_dev->ctx);
	    HIP_CALL_ERET (propval, hipMemGetInfo, &free_mem, &total_mem);
	    HIP_CALL_ASSERT (hipCtxPopCurrent, &old_ctx);
	  }
	else
	  {
	    hipCtx_t new_ctx;

	    HIP_CALL_ERET (propval, hipCtxCreate, &new_ctx, hipDeviceScheduleAuto,
			    hip_dev->dev);
	    HIP_CALL_ERET (propval, hipMemGetInfo, &free_mem, &total_mem);
	    HIP_CALL_ASSERT (hipCtxDestroy, new_ctx);
	  }
	propval.val = free_mem;
      }
      break;
    case GOACC_PROPERTY_NAME:
      propval.ptr = hip_dev->name;
      break;
    case GOACC_PROPERTY_VENDOR:
      propval.ptr = "AMD";
      break;
    case GOACC_PROPERTY_DRIVER:
      propval.ptr = rocm_version_s;
      break;
    default:
      break;
    }

  pthread_mutex_unlock (&hip_dev_lock);
  return propval;
}

/* Adjust launch dimensions: pick good values for number of blocks and warps
   and ensure that number of warps does not exceed CUDA limits as well as GCC's
   own limits.  */

static void
hip_adjust_launch_bounds (struct targ_fn_descriptor *fn,
			    struct hip_device *hip_dev,
			    int *teams_p, int *threads_p)
{
//  int max_warps_block = fn->max_threads_per_block / 32;
//  /* Maximum 16 warps per block is an implementation limit in HIP backend (AMD devices)
//     and libgcc, which matches documented limit of all GPUs as of 2015.  */
//  
//  if (max_warps_block > 32)
//    max_warps_block = 32;
//  if (*threads_p <= 0)
//    *threads_p = 8;
//  if (*threads_p > max_warps_block)
//    *threads_p = max_warps_block;
//
//  int regs_per_block = fn->regs_per_thread * 32 * *threads_p;
//  /* This is an estimate of how many blocks the device can host simultaneously.
//     Actual limit, which may be lower, can be queried with "occupancy control"
//     runtime interface.  */
//  int max_blocks = hip_dev->regs_per_sm / regs_per_block * hip_dev->num_sms;
//  if (*teams_p <= 0 || *teams_p > max_blocks)
//    *teams_p = max_blocks;
//  
}

/* Return the size of per-warp stacks (see gcc -msoft-stack) to use for OpenMP
   target regions.  */

static size_t
hip_stacks_size ()
{
  return 128 * 1024;
}

/* Return contiguous storage for NUM stacks, each SIZE bytes.  The lock for
   the storage should be held on entry, and remains held on exit.  */

static void *
hip_stacks_acquire (struct hip_device *hip_dev, size_t size, int num)
{
  if (hip_dev->omp_stacks.ptr && hip_dev->omp_stacks.size >= size * num)
    return (void *) hip_dev->omp_stacks.ptr;

  /* Free the old, too-small stacks.  */
  if (hip_dev->omp_stacks.ptr)
    {
      hipError_t r = HIP_CALL_NOCHECK (hipCtxSynchronize, );
      if (r != hipSuccess)
	GOMP_PLUGIN_fatal ("hipCtxSynchronize error: %s\n", hip_error (r));
      r = HIP_CALL_NOCHECK (hipFree, hip_dev->omp_stacks.ptr);
      if (r != hipSuccess)
	GOMP_PLUGIN_fatal ("hipFree error: %s", hip_error (r));
    }

  /* Make new and bigger stacks, and remember where we put them and how big
     they are.  */
  hipError_t r = HIP_CALL_NOCHECK (hipMalloc, &hip_dev->omp_stacks.ptr,
				  size * num);
  if (r != hipSuccess)
    GOMP_PLUGIN_fatal ("hipMalloc error: %s", hip_error (r));

  hip_dev->omp_stacks.size = size * num;

  return (void *) hip_dev->omp_stacks.ptr;
}

void
GOMP_OFFLOAD_run (int ord, void *tgt_fn, void *tgt_vars, void **args)
{
//  struct targ_fn_descriptor *tgt_fn_desc
//    = (struct targ_fn_descriptor *) tgt_fn;
//  hipFunction_t function = tgt_fn_desc->fn;
//  const struct targ_fn_launch *launch = tgt_fn_desc->launch;
//  const char *fn_name = launch->fn;
//  hipError_t r;
//  struct hip_device *hip_dev = hip_devices[ord];
//  const char *maybe_abort_msg = "(perhaps abort was called)";
//  int teams = 0, threads = 0;
//
//  if (!args)
//    GOMP_PLUGIN_fatal ("No target arguments provided");
//  while (*args)
//    {
//      intptr_t id = (intptr_t) *args++, val;
//      if (id & GOMP_TARGET_ARG_SUBSEQUENT_PARAM)
//	val = (intptr_t) *args++;
//      else
//        val = id >> GOMP_TARGET_ARG_VALUE_SHIFT;
//      if ((id & GOMP_TARGET_ARG_DEVICE_MASK) != GOMP_TARGET_ARG_DEVICE_ALL)
//	continue;
//      val = val > INT_MAX ? INT_MAX : val;
//      id &= GOMP_TARGET_ARG_ID_MASK;
//      if (id == GOMP_TARGET_ARG_NUM_TEAMS)
//	teams = val;
//      else if (id == GOMP_TARGET_ARG_THREAD_LIMIT)
//	threads = val;
//    }
//  hip_adjust_launch_bounds (tgt_fn, hip_dev, &teams, &threads);
//
//  size_t stack_size = hip_stacks_size ();
//
//  pthread_mutex_lock (&hip_dev->omp_stacks.lock);
//  void *stacks = hip_stacks_acquire (hip_dev, stack_size, teams * threads);
//  void *fn_args[] = {tgt_vars, stacks, (void *) stack_size};
//  size_t fn_args_size = sizeof fn_args;
//  void *config[] = {
//    HIP_LAUNCH_PARAM_BUFFER_POINTER, fn_args,
//    HIP_LAUNCH_PARAM_BUFFER_SIZE, &fn_args_size,
//    HIP_LAUNCH_PARAM_END
//  };
//  GOMP_PLUGIN_debug (0, "  %s: kernel %s: launch"
//		     " [(teams: %u), 1, 1] [(lanes: 32), (threads: %u), 1]\n",
//		     __FUNCTION__, fn_name, teams, threads);
//  r = HIP_CALL_NOCHECK (hipModuleLaunchKernel, function, teams, 1, 1,
//			 32, threads, 1, 0, NULL, NULL, config);
//  if (r != hipSuccess)
//    GOMP_PLUGIN_fatal ("hipModuleLaunchKernel error: %s", hip_error (r));
//
//  r = HIP_CALL_NOCHECK (hipCtxSynchronize, );
//  if (r == hipErrorLaunchFailure)
//    GOMP_PLUGIN_fatal ("hipCtxSynchronize error: %s %s\n", hip_error (r),
//		       maybe_abort_msg);
//  else if (r != hipSuccess)
//    GOMP_PLUGIN_fatal ("hipCtxSynchronize error: %s", hip_error (r));
//
//  pthread_mutex_unlock (&hip_dev->omp_stacks.lock);
}

/* TODO: Implement GOMP_OFFLOAD_async_run. */
