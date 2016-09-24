/**
 * covariance.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define GPU_DEVICE 0

/* Problem size */
#ifndef NI
#define NI 512 //2048
#endif
#ifndef NJ
#define NJ 512 //2048
#endif

/* Thread block dimensions for kernel 1*/
#ifndef DIM_THREAD_BLOCK_KERNEL_1_X
#define DIM_THREAD_BLOCK_KERNEL_1_X 256
#endif
#ifndef DIM_THREAD_BLOCK_KERNEL_1_Y
#define DIM_THREAD_BLOCK_KERNEL_1_Y 1
#endif

/* Thread block dimensions for kernel 2*/
#ifndef DIM_THREAD_BLOCK_KERNEL_2_X
#define DIM_THREAD_BLOCK_KERNEL_2_X 32
#endif
#ifndef DIM_THREAD_BLOCK_KERNEL_2_Y
#define DIM_THREAD_BLOCK_KERNEL_2_Y 8
#endif

/* Thread block dimensions for kernel 3*/
#ifndef DIM_THREAD_BLOCK_KERNEL_3_X
#define DIM_THREAD_BLOCK_KERNEL_3_X 256
#endif
#ifndef DIM_THREAD_BLOCK_KERNEL_3_Y
#define DIM_THREAD_BLOCK_KERNEL_3_Y 1
#endif

#define sqrt_of_array_cell(x,j) sqrt(x[j])

#define FLOAT_N 3214212.01
#define EPS 0.005

/* Can switch DATA_TYPE between float and double */
# ifndef DATA_TYPE
#  define DATA_TYPE float
# endif

void init_arrays(DATA_TYPE* data)
{
	int i, j;

	for (i = 1; i < (NI+1); i++)
	{
		for (j = 1; j < (NJ+1); j++)
		{
			data[i*(NJ+1) + j] = ((DATA_TYPE) i*j) / NI;
		}
	}
}


void covariance(DATA_TYPE* data, DATA_TYPE* symmat, DATA_TYPE* mean)
{
	int i, j, j1,j2;

  	/* Determine mean of column vectors of input data matrix */
	for (j = 1; j < (NI+1); j++)
	{
		mean[j] = 0.0;
		for (i = 1; i < (NJ+1); i++)
		{
        		mean[j] += data[i*(NI+1) + j];
		}
		mean[j] /= FLOAT_N;
	}

  	/* Center the column vectors. */
	for (i = 1; i < (NJ+1); i++)
	{
		for (j = 1; j < (NI+1); j++)
		{
			data[i*(NI+1) + j] -= mean[j];
		}
	}

  	/* Calculate the m * m covariance matrix. */
	for (j1 = 1; j1 < (NI+1); j1++)
	{
		for (j2 = j1; j2 < (NI+1); j2++)
     		{
       		symmat[j1*(NI+1) + j2] = 0.0;
			for (i = 1; i < NJ+1; i++)
			{
				symmat[j1*(NI+1) + j2] += data[i*(NI+1) + j1] * data[i*(NI+1) + j2];
			}
        		symmat[j2*(NI+1) + j1] = symmat[j1*(NI+1) + j2];
      		}
	}
}


void compareResults(DATA_TYPE* symmat, DATA_TYPE* symmat_outputFromGpu)
{
	int i,j,fail;
	fail = 0;

	for (i=1; i < (NI+1); i++)
	{
		for (j=1; j < (NJ+1); j++)
		{
			if (percentDiff(symmat[i*(NJ+1) + j], symmat_outputFromGpu[i*(NJ+1) + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}			
		}
	}
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


__global__ void mean_kernel(DATA_TYPE *mean, DATA_TYPE *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

	if ((j >= 1) && (j < (NI+1)))
	{
		mean[j] = 0.0;

		int i;
		for(i = 1; i < (NJ+1); i++)
		{
			mean[j] += data[i * (NI+1) + j];
		}
		mean[j] /= (DATA_TYPE)FLOAT_N;
	}
}


__global__ void reduce_kernel(DATA_TYPE *mean, DATA_TYPE *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
		
	if ((i >= 1) && (i < (NJ+1)) && (j >= 1) && (j < (NI+1)))
	{
		data[i * (NI+1) + j] -= mean[j];	
	}
}


__global__ void covar_kernel(DATA_TYPE *symmat, DATA_TYPE *data)
{
	int j1 = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int i, j2;

	if ((j1 >= 1) && (j1 < (NI+1)))
	{
		for (j2 = j1; j2 < (NI+1); j2++)
		{		
			symmat[j1*(NI+1) + j2] = 0.0;
			for(i = 1; i < (NJ+1); i++)
			{
				symmat[j1 * (NI+1) + j2] += data[i *(NI+1) + j1] * data[i *(NI+1) + j2];
			}
			symmat[j2 * (NI+1) + j1] = symmat[j1 * (NI+1) + j2];
		}
	}
}


void covarianceCuda(DATA_TYPE* data, DATA_TYPE* symmat, DATA_TYPE* mean, DATA_TYPE* symmat_outputFromGpu)
{
        cudaError_t error;
	double t_start, t_end;

	DATA_TYPE *data_gpu;
	DATA_TYPE *mean_gpu;
	DATA_TYPE *symmat_gpu;

	error=cudaMalloc((void **)&data_gpu, sizeof(DATA_TYPE) * (NI+1) * (NJ+1));
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMalloc((void **)&symmat_gpu, sizeof(DATA_TYPE) * (NI+1) * (NI+1));
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMalloc((void **)&mean_gpu, sizeof(DATA_TYPE) * (NI+1));
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMemcpy(data_gpu, data, sizeof(DATA_TYPE) * (NI+1) * (NJ+1), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMemcpy(symmat_gpu, symmat, sizeof(DATA_TYPE) * (NI+1) * (NI+1), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMemcpy(mean_gpu, mean, sizeof(DATA_TYPE) * (NI+1), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }
	
	dim3 block1(DIM_THREAD_BLOCK_KERNEL_1_X, DIM_THREAD_BLOCK_KERNEL_1_Y);
	dim3 grid1((size_t)(ceil((float)NI) / ((float)DIM_THREAD_BLOCK_KERNEL_1_X)), 1);
	
	dim3 block2(DIM_THREAD_BLOCK_KERNEL_2_X, DIM_THREAD_BLOCK_KERNEL_2_Y);
	dim3 grid2((size_t)(ceil((float)NI) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X)), (size_t)(ceil((float)NJ) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X)));
	
	dim3 block3(DIM_THREAD_BLOCK_KERNEL_3_X, DIM_THREAD_BLOCK_KERNEL_3_Y);
	dim3 grid3((size_t)(ceil((float)NI) / ((float)DIM_THREAD_BLOCK_KERNEL_3_X)), 1);
	
//	t_start = rtclock();

	mean_kernel<<<grid1, block1>>>(mean_gpu,data_gpu);
	cudaThreadSynchronize();
	reduce_kernel<<<grid2, block2>>>(mean_gpu,data_gpu);
	cudaThreadSynchronize();
	covar_kernel<<<grid3, block3>>>(symmat_gpu,data_gpu);
	cudaThreadSynchronize();
//	t_end = rtclock();
//	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

	error=cudaMemcpy(symmat_outputFromGpu, symmat_gpu, sizeof(DATA_TYPE) * (NI+1) * (NJ+1), cudaMemcpyDeviceToHost);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }
	
	cudaFree(data_gpu);
	cudaFree(symmat_gpu);
	cudaFree(mean_gpu);
}


int main()
{
  /* Prepare ctuning vars */
  long ct_repeat=0;
  long ct_repeat_max=1;

  double t_start, t_end;

  DATA_TYPE* data;
  DATA_TYPE* symmat;
  DATA_TYPE* mean;
  DATA_TYPE* symmat_outputFromGpu;	

#ifdef XOPENME
  xopenme_init(2,0);
#endif

#ifdef OPENME
  openme_init(NULL,NULL,NULL,0);
  openme_callback("PROGRAM_START", NULL);
#endif

  /* Run kernel. */
  if (getenv("CT_REPEAT_MAIN")!=NULL) ct_repeat_max=atol(getenv("CT_REPEAT_MAIN"));

  data = (DATA_TYPE*)malloc((NI+1)*(NJ+1)*sizeof(DATA_TYPE));
  symmat = (DATA_TYPE*)malloc((NI+1)*(NI+1)*sizeof(DATA_TYPE));
  mean = (DATA_TYPE*)malloc((NI+1)*sizeof(DATA_TYPE));
  symmat_outputFromGpu = (DATA_TYPE*)malloc((NI+1)*(NI+1)*sizeof(DATA_TYPE));	

  srand(1);
  init_arrays(data);
  GPU_argv_init();

#ifdef OPENME
  openme_callback("ACC_KERNEL_START", NULL);
#endif
#ifdef XOPENME
  xopenme_clock_start(0);
#endif
  for (ct_repeat=0; ct_repeat<ct_repeat_max; ct_repeat++)
  {
    covarianceCuda(data, symmat, mean, symmat_outputFromGpu);
  }
#ifdef XOPENME
  xopenme_clock_end(0);
#endif
#ifdef OPENME
  openme_callback("ACC_KERNEL_END", NULL);
#endif

/*
  srand(1);
  init_arrays(data);

#ifdef OPENME
  openme_callback("KERNEL_START", NULL);
#endif
#ifdef XOPENME
  xopenme_clock_start(1);
#endif
  for (ct_repeat=0; ct_repeat<ct_repeat_max; ct_repeat++)
  {
    covariance(data, symmat, mean);
  }
#ifdef XOPENME
  xopenme_clock_end(1);
#endif
#ifdef OPENME
  openme_callback("KERNEL_END", NULL);
#endif
*/

  compareResults(symmat, symmat_outputFromGpu);

  free(data);
  free(symmat);
  free(mean);
  free(symmat_outputFromGpu);	

#ifdef XOPENME
  xopenme_dump_state();
  xopenme_finish();
#endif

#ifdef OPENME
  openme_callback("PROGRAM_END", NULL);
#endif

  return 0;
}

