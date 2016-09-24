/**
 * atax.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

/* Problem size. */
#ifndef NX
#define NX 4096
#endif
#ifndef NY
#define NY 4096
#endif

/* Thread block dimensions */
#ifndef DIM_THREAD_BLOCK_X
#define DIM_THREAD_BLOCK_X 256
#endif
#ifndef DIM_THREAD_BLOCK_Y
#define DIM_THREAD_BLOCK_Y 1
#endif

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
# ifndef DATA_TYPE
#  define DATA_TYPE float
# endif

void init_array(DATA_TYPE *x, DATA_TYPE *A)
{
	int i, j;

	for (i = 0; i < NX; i++)
	{
		x[i] = i * M_PI;
		for (j = 0; j < NY; j++)
		{
			A[i*NY + j] = ((DATA_TYPE) i*(j)) / NX;
		}
	}
}


void compareResults(DATA_TYPE *z, DATA_TYPE *z_outputFromGpu)
{
	int i, fail;
	fail = 0;

	for (i=0; i<NY; i++)
	{
		if (percentDiff(z[i], z_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
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


__global__ void atax_kernel1(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < NX)
	{
		int j;
		for(j=0; j < NY; j++)
		{
			tmp[i] += A[i * NY + j] * x[j];
		}
	}
}

__global__ void atax_kernel2(DATA_TYPE *A, DATA_TYPE *y, DATA_TYPE *tmp)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < NY)
	{
		int i;
		for(i=0; i < NX; i++)
		{
			y[j] += A[i * NY + j] * tmp[i];
		}
	}
}


void atax_cpu(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp)
{
	int i,j;
	
	for (i= 0; i < NY; i++)
	{
    	y[i] = 0;
	}
  
	for (i = 0; i < NX; i++)
 	{
      	tmp[i] = 0;

      	for (j = 0; j < NY; j++)
		{
			tmp[i] = tmp[i] + A[i*NY + j] * x[j];
		}
		
      	for (j = 0; j < NY; j++)
		{
			y[j] = y[j] + A[i*NY + j] * tmp[i];
		}
    }
}


void ataxGpu(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp, DATA_TYPE* y_outputFromGpu)
{
        cudaError_t error;
	double t_start, t_end;

	DATA_TYPE *A_gpu;
	DATA_TYPE *x_gpu;
	DATA_TYPE *y_gpu;
	DATA_TYPE *tmp_gpu;

	error=cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NX * NY);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * NY);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * NY);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * NX);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }
	
	error=cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMemcpy(y_gpu, y, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMemcpy(tmp_gpu, tmp, sizeof(DATA_TYPE) * NX, cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NX) / ((float)block.x) )), 1);
	dim3 grid2((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);

//	t_start = rtclock();
	atax_kernel1<<< grid1, block >>>(A_gpu,x_gpu,tmp_gpu);
	cudaThreadSynchronize();
	atax_kernel2<<< grid2, block >>>(A_gpu,y_gpu,tmp_gpu);
	cudaThreadSynchronize();
//	t_end = rtclock();
//	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	
	error=cudaMemcpy(y_outputFromGpu, y_gpu, sizeof(DATA_TYPE) * NX, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	cudaFree(A_gpu);
	cudaFree(x_gpu);
	cudaFree(y_gpu);
	cudaFree(tmp_gpu);
}


int main(int argc, char** argv)
{
  /* Prepare ctuning vars */
  long ct_repeat=0;
  long ct_repeat_max=1;

  DATA_TYPE* A;
  DATA_TYPE* x;
  DATA_TYPE* y;
  DATA_TYPE* y_outputFromGpu;
  DATA_TYPE* tmp;

#ifdef XOPENME
  xopenme_init(2,0);
#endif

#ifdef OPENME
  openme_init(NULL,NULL,NULL,0);
  openme_callback("PROGRAM_START", NULL);
#endif

  /* Run kernel. */
  if (getenv("CT_REPEAT_MAIN")!=NULL) ct_repeat_max=atol(getenv("CT_REPEAT_MAIN"));

  A = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
  x = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
  y = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
  y_outputFromGpu = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
  tmp = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));

  srand(1);
  init_array(x, A);
  GPU_argv_init();

#ifdef OPENME
  openme_callback("ACC_KERNEL_START", NULL);
#endif
#ifdef XOPENME
  xopenme_clock_start(0);
#endif
  for (ct_repeat=0; ct_repeat<ct_repeat_max; ct_repeat++)
  {
    ataxGpu(A, x, y, tmp, y_outputFromGpu);
  }
#ifdef XOPENME
  xopenme_clock_end(0);
#endif
#ifdef OPENME
  openme_callback("ACC_KERNEL_END", NULL);
#endif

/*
  srand(1);
  init_array(x, A);

#ifdef OPENME
  openme_callback("KERNEL_START", NULL);
#endif
#ifdef XOPENME
  xopenme_clock_start(1);
#endif
  for (ct_repeat=0; ct_repeat<ct_repeat_max; ct_repeat++)
  {
    atax_cpu(A, x, y, tmp);
  }
#ifdef XOPENME
  xopenme_clock_end(1);
#endif
#ifdef OPENME
  openme_callback("KERNEL_END", NULL);
#endif
*/

  compareResults(y, y_outputFromGpu);

  free(A);
  free(x);
  free(y);
  free(y_outputFromGpu);
  free(tmp);

#ifdef XOPENME
  xopenme_dump_state();
  xopenme_finish();
#endif

#ifdef OPENME
  openme_callback("PROGRAM_END", NULL);
#endif

  return 0;
}

