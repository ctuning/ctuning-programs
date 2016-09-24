/**
 * syrk.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Updated by Grigori Fursin (http://cTuning.org/lab/people/gfursin)
 * to work with Collective Mind, OpenME plugin interface and 
 * Collective Knowledge Frameworks for automatic, machine-learning based
 * and collective tuning and data mining: http://cTuning.org
 *
 */

#ifndef WINDOWS
 #include <unistd.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <cuda.h>

#include "polybench.h"

#ifdef OPENME
#include <openme.h>
#endif
#ifdef XOPENME
#include <xopenme.h>
#endif

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
#ifndef NI
#define NI 256 // 1024
#endif
#ifndef NJ
#define NJ 256 // 1024
#endif

/* Thread block dimensions */
#ifndef DIM_THREAD_BLOCK_Y
#define DIM_THREAD_BLOCK_X 32
#endif
#ifndef DIM_THREAD_BLOCK_Y
#define DIM_THREAD_BLOCK_Y 8
#endif

/* Declared constant values for alpha and beta (same as values in PolyBench 2.0) */
#define alpha 12435
#define beta 4546

/* Can switch DATA_TYPE between float and double */
# ifndef DATA_TYPE
#  define DATA_TYPE float
# endif

void init_arrays(DATA_TYPE* A, DATA_TYPE* C)
{
	int i, j;
	
	for (i = 0; i < NJ; i++)
    	{
		for (j = 0; j < NI; j++)
		{
			A[i*NI + j] = ((DATA_TYPE) i*j) / NJ;
		}
		
		for (j = 0; j < NJ; j++)
		{
			C[i*NI + j] = ((DATA_TYPE) i*j + 2) / NJ;
		}
	}
}


void syrk(DATA_TYPE* A, DATA_TYPE* C)
{
	int i, j, k;
	
	/*  C := alpha*A*A' + beta*C */
	for (i = 0; i < NJ; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			C[i*NI + j] *= beta;
		}
	}
	
	for (i = 0; i < NJ; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			for (k = 0; k < NI; k++)
			{
				C[i*NJ + j] += alpha * A[i*NI + k] * A[j*NI + k];
			}
		}
	}
}


void compareResults(DATA_TYPE* C, DATA_TYPE* C_outputFromGpu)
{
	int i,j,fail;
	fail = 0;

	// Compare C with D
	for (i=0; i<NJ; i++)
	{
		for (j=0; j<NI; j++)
		{
			if (percentDiff(C[i*NI + j], C_outputFromGpu[i*NI + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
  /* Grigori Fursin added support for CK widgets */
  int gpgpu_device_id=GPU_DEVICE;

  int devID = 0;
  cudaError_t error;
  cudaDeviceProp deviceProp;
  error = cudaGetDevice(&devID);

  if (getenv("CK_COMPUTE_DEVICE_ID")!=NULL) gpgpu_device_id=atol(getenv("CK_COMPUTE_DEVICE_ID"));

  cudaGetDeviceProperties(&deviceProp, gpgpu_device_id);

  if (deviceProp.computeMode == cudaComputeModeProhibited)
  {
    printf("Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
    exit(EXIT_SUCCESS);
  }

  if (error != cudaSuccess)
    printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
  else
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

  cudaSetDevice( gpgpu_device_id );
}


__global__ void syrk_kernel(DATA_TYPE ALPHA, DATA_TYPE BETA, DATA_TYPE *a, DATA_TYPE *c)
{
	/*  C := alpha*A*A' + beta*C */
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NJ) && (j < NJ))
	{
		c[i * NJ + j] *= beta;
		int k;		
		for(k=0; k< NI; k++)
		{
			c[i * NJ + j] += alpha * a[i * NI + k] * a[j * NI + k];
		}
	}
}


void syrkCuda(DATA_TYPE* A, DATA_TYPE* C, DATA_TYPE* C_outputFromGpu)
{
        cudaError_t error;
	double t_start, t_end;

	DATA_TYPE* A_gpu;
	DATA_TYPE* C_gpu;

	error=cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NJ * NI);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NJ * NJ);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NJ * NI, cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NJ * NJ, cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil(((float)NJ) / ((float)DIM_THREAD_BLOCK_X))), (size_t)ceil(((float)NJ) / ((float)DIM_THREAD_BLOCK_Y)));
//	t_start = rtclock();
	syrk_kernel<<<grid,block>>>(alpha, beta, A_gpu,C_gpu);
	cudaThreadSynchronize();
//	t_end = rtclock();
//	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

	error=cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * NJ * NJ, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	cudaFree(A_gpu);
	cudaFree(C_gpu);
}


int main()
{
  /* Prepare ctuning vars */
  long ct_repeat=0;
  long ct_repeat_max=1;

  double t_start, t_end;

  DATA_TYPE* A;
  DATA_TYPE* C;
  DATA_TYPE* C_outputFromGpu;

#ifdef XOPENME
  xopenme_init(2,0);
#endif

#ifdef OPENME
  openme_init(NULL,NULL,NULL,0);
  openme_callback("PROGRAM_START", NULL);
#endif

  /* Run kernel. */
  if (getenv("CT_REPEAT_MAIN")!=NULL) ct_repeat_max=atol(getenv("CT_REPEAT_MAIN"));

  A = (DATA_TYPE*)malloc(NJ*NI*sizeof(DATA_TYPE));
  C = (DATA_TYPE*)malloc(NJ*NI*sizeof(DATA_TYPE));
  C_outputFromGpu = (DATA_TYPE*)malloc(NJ*NI*sizeof(DATA_TYPE));

  srand(1);
  init_arrays(A, C);
  GPU_argv_init();	

#ifdef OPENME
  openme_callback("ACC_KERNEL_START", NULL);
#endif
#ifdef XOPENME
  xopenme_clock_start(0);
#endif
  for (ct_repeat=0; ct_repeat<ct_repeat_max; ct_repeat++)
  {
    syrkCuda(A, C, C_outputFromGpu);
  }
#ifdef XOPENME
  xopenme_clock_end(0);
#endif
#ifdef OPENME
  openme_callback("ACC_KERNEL_END", NULL);
#endif

/*
  srand(1);
  init_arrays(A, C);

#ifdef OPENME
  openme_callback("KERNEL_START", NULL);
#endif
#ifdef XOPENME
  xopenme_clock_start(1);
#endif
  for (ct_repeat=0; ct_repeat<ct_repeat_max; ct_repeat++)
  {
    syrk(A, C);
  }
#ifdef XOPENME
  xopenme_clock_end(1);
#endif
#ifdef OPENME
  openme_callback("KERNEL_END", NULL);
#endif
*/

  compareResults(C, C_outputFromGpu);

  free(A);
  free(C);
  free(C_outputFromGpu);

#ifdef XOPENME
  xopenme_dump_state();
  xopenme_finish();
#endif

#ifdef OPENME
  openme_callback("PROGRAM_END", NULL);
#endif

  return 0;
}

