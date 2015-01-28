/**
 * correlation.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Updated by Grigori Fursin (http://cTuning.org/lab/people/gfursin)
 * to work with Collective Mind Framework and OpenME interfqce for automatic 
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

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define GPU_DEVICE 0

/* Problem size */
#define M 512 //2048
#define N 512 //2048

/* Thread block dimensions for kernel 1*/
#define DIM_THREAD_BLOCK_KERNEL_1_X 256
#define DIM_THREAD_BLOCK_KERNEL_1_Y 1

/* Thread block dimensions for kernel 2*/
#define DIM_THREAD_BLOCK_KERNEL_2_X 256
#define DIM_THREAD_BLOCK_KERNEL_2_Y 1

/* Thread block dimensions for kernel 3*/
#define DIM_THREAD_BLOCK_KERNEL_3_X 32
#define DIM_THREAD_BLOCK_KERNEL_3_Y 8

/* Thread block dimensions for kernel 4*/
#define DIM_THREAD_BLOCK_KERNEL_4_X 256
#define DIM_THREAD_BLOCK_KERNEL_4_Y 1

#define sqrt_of_array_cell(x,j) sqrt(x[j])

#define FLOAT_N 3214212.01f
#define EPS 0.005f

/* Can switch DATA_TYPE between float and double */
# ifndef DATA_TYPE
#  define DATA_TYPE float
# endif

void init_arrays(DATA_TYPE* data)
{
	int i, j;
	
	for (i=0; i < (M+1); i++) 
	{
    		for (j=0; j< (N+1); j++) 
		{
       			data[i*(N+1) + j] = ((DATA_TYPE) i*j)/ (M+1);	
       		}
    	}
}


void correlation(DATA_TYPE* data, DATA_TYPE* mean, DATA_TYPE* stddev, DATA_TYPE* symmat)
{
	int i, j, j1, j2;	
	
	// Determine mean of column vectors of input data matrix 
  	for (j = 1; j < (M+1); j++)
   	{
  		mean[j] = 0.0;

   		for (i = 1; i < (N+1); i++)
		{
			mean[j] += data[i*(M+1) + j];
   		}
		
		mean[j] /= (DATA_TYPE)FLOAT_N;
   	}

	// Determine standard deviations of column vectors of data matrix. 
  	for (j = 1; j < (M+1); j++)
   	{
   		stddev[j] = 0.0;
      
		for (i = 1; i < (N+1); i++)
		{
			stddev[j] += (data[i*(M+1) + j] - mean[j]) * (data[i*(M+1) + j] - mean[j]);
		}
		
		stddev[j] /= FLOAT_N;
		stddev[j] = sqrt_of_array_cell(stddev, j);
		stddev[j] = stddev[j] <= EPS ? 1.0 : stddev[j];
	}

 	// Center and reduce the column vectors. 
  	for (i = 1; i < (N+1); i++)
	{
		for (j = 1; j < (M+1); j++)
		{
			data[i*(M+1) + j] -= mean[j];
			data[i*(M+1) + j] /= (sqrt(FLOAT_N)*stddev[j]) ;
		}
	}

	// Calculate the m * m correlation matrix. 
  	for (j1 = 1; j1 < M; j1++)
	{	
		symmat[j1*(M+1) + j1] = 1.0;
    
		for (j2 = j1+1; j2 < (M+1); j2++)
		{
	  		symmat[j1*(M+1) + j2] = 0.0;

	  		for (i = 1; i < (N+1); i++)
			{
	   			symmat[j1*(M+1) + j2] += (data[i*(M+1) + j1] * data[i*(M+1) + j2]);
			}

	  		symmat[j2*(M+1) + j1] = symmat[j1*(M+1) + j2];
		}
	}
 
	symmat[M*(M+1) + M] = 1.0;
}


void compareResults(DATA_TYPE* symmat, DATA_TYPE* symmat_outputFromGpu)
{
	int i,j,fail;
	fail = 0;

	for (i=1; i < (M+1); i++)
	{
		for (j=1; j < (N+1); j++)
		{
			if (percentDiff(symmat[i*(N+1) + j], symmat_outputFromGpu[i*(N+1) + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
//				printf("i: %d j: %d\n1: %f 2: %f\n", i, j, symmat[i*N + j], symmat_outputFromGpu[i*N + j]);
		
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
  int devID = 0;
  cudaError_t error;
  cudaDeviceProp deviceProp;
  error = cudaGetDevice(&devID);

  cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
  
  if (deviceProp.computeMode == cudaComputeModeProhibited)
  {
    printf("Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
    exit(EXIT_SUCCESS);
  }

  if (error != cudaSuccess)
    printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
  else
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

  cudaSetDevice( GPU_DEVICE );
}

	
__global__ void mean_kernel(DATA_TYPE *mean, DATA_TYPE *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

	if ((j >= 1) && (j < (M+1)))
	{
		mean[j] = 0.0;

		int i;
		for(i=1; i < (N+1); i++)
		{
			mean[j] += data[i*(M+1) + j];
		}
		
		mean[j] /= (DATA_TYPE)FLOAT_N;
	}
}


__global__ void std_kernel(DATA_TYPE *mean, DATA_TYPE *std, DATA_TYPE *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
	
	if ((j >= 1) && (j < (M+1)))
	{
		std[j] = 0.0;

		int i;
		for(i = 1; i < (N+1); i++)
		{
			std[j] += (data[i*(M+1) + j] - mean[j]) * (data[i*(M+1) + j] - mean[j]);
		}
		std[j] /= (FLOAT_N);
		std[j] = sqrt(std[j]);
		if(std[j] <= EPS) 
		{
			std[j] = 1.0;
		}
	}
}


__global__ void reduce_kernel(DATA_TYPE *mean, DATA_TYPE *std, DATA_TYPE *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
	
	if ((i >= 1) && (i < (N+1)) && (j >= 1) && (j < (M+1)))
	{
		data[i*(M+1) + j] -= mean[j];
		data[i*(M+1) + j] /= (sqrt(FLOAT_N) * std[j]);
	}
}


__global__ void corr_kernel(DATA_TYPE *symmat, DATA_TYPE *data)
{
	int j1 = blockIdx.x * blockDim.x + threadIdx.x + 1;

	int i, j2;
	if ((j1 >= 1) && (j1 < M))
	{
		symmat[j1*(M+1) + j1] = 1.0;

		for (j2 = (j1 + 1); j2 < (M+1); j2++)
		{
			symmat[j1*(M+1) + j2] = 0.0;

			for(i = 1; i < (N+1); i++)
			{
				symmat[j1*(M+1) + j2] += data[i*(M+1) + j1] * data[i*(M+1) + j2];
			}
			symmat[j2*(M+1) + j1] = symmat[j1*(M+1) + j2];
		}
	}
}


void correlationCuda(DATA_TYPE* data, DATA_TYPE* mean, DATA_TYPE* stddev, DATA_TYPE* symmat,
			DATA_TYPE* symmat_outputFromGpu)
{
        cudaError_t error;
	double t_start, t_end;

	DATA_TYPE *data_gpu;
	DATA_TYPE *stddev_gpu;
	DATA_TYPE *mean_gpu;
	DATA_TYPE *symmat_gpu;

	error=cudaMalloc((void **)&data_gpu, sizeof(DATA_TYPE) * (M+1) * (N+1));
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMalloc((void **)&symmat_gpu, sizeof(DATA_TYPE) * (M+1) * (N+1));
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMalloc((void **)&stddev_gpu, sizeof(DATA_TYPE) * (M+1));
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMalloc((void **)&mean_gpu, sizeof(DATA_TYPE) * (M+1));
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMemcpy(data_gpu, data, sizeof(DATA_TYPE) * (M+1) * (N+1), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMemcpy(symmat_gpu, symmat, sizeof(DATA_TYPE) * (M+1) * (N+1), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMemcpy(stddev_gpu, stddev, sizeof(DATA_TYPE) * (M+1), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMemcpy(mean_gpu, mean, sizeof(DATA_TYPE) * (M+1), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }
		
	dim3 block1(DIM_THREAD_BLOCK_KERNEL_1_X, DIM_THREAD_BLOCK_KERNEL_1_Y);
	dim3 grid1((size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_1_X)), 1);
	
	dim3 block2(DIM_THREAD_BLOCK_KERNEL_2_X, DIM_THREAD_BLOCK_KERNEL_2_Y);
	dim3 grid2((size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X)), 1);
	
	dim3 block3(DIM_THREAD_BLOCK_KERNEL_3_X, DIM_THREAD_BLOCK_KERNEL_3_Y);
	dim3 grid3((size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_3_X)), (size_t)(ceil((float)(N)) / ((float)DIM_THREAD_BLOCK_KERNEL_3_Y)));
	
	dim3 block4(DIM_THREAD_BLOCK_KERNEL_4_X, DIM_THREAD_BLOCK_KERNEL_4_Y);
	dim3 grid4((size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_4_X)), 1);

//	t_start = rtclock();
	mean_kernel<<< grid1, block1 >>>(mean_gpu,data_gpu);
	cudaThreadSynchronize();
	std_kernel<<< grid2, block2 >>>(mean_gpu,stddev_gpu,data_gpu);
	cudaThreadSynchronize();
	reduce_kernel<<< grid3, block3 >>>(mean_gpu,stddev_gpu,data_gpu);
	cudaThreadSynchronize();
	corr_kernel<<< grid4, block4 >>>(symmat_gpu,data_gpu);
	cudaThreadSynchronize();
//	t_end = rtclock();
//	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);	

	DATA_TYPE valueAtSymmatIndexMTimesMPlus1PlusMPoint = 1.0;
	error=cudaMemcpy(&(symmat_gpu[(M)*(M+1) + (M)]), &valueAtSymmatIndexMTimesMPlus1PlusMPoint, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMemcpy(symmat_outputFromGpu, symmat_gpu, sizeof(DATA_TYPE) * (M+1) * (N+1), cudaMemcpyDeviceToHost);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }
	
	cudaFree(data_gpu);
	cudaFree(symmat_gpu);
	cudaFree(stddev_gpu);
	cudaFree(mean_gpu);
}


int main()
{
  /* Prepare ctuning vars */
  long ct_repeat=0;
  long ct_repeat_max=1;

  double t_start, t_end;

  DATA_TYPE* data;
  DATA_TYPE* mean;
  DATA_TYPE* stddev;
  DATA_TYPE* symmat;
  DATA_TYPE* symmat_outputFromGpu;

#ifdef OPENME
  openme_init(NULL,NULL,NULL,0);
  openme_callback("PROGRAM_START", NULL);
#endif

  /* Run kernel. */
  if (getenv("CT_REPEAT_MAIN")!=NULL) ct_repeat_max=atol(getenv("CT_REPEAT_MAIN"));

  data = (DATA_TYPE*)malloc((M+1)*(N+1)*sizeof(DATA_TYPE));
  mean = (DATA_TYPE*)malloc((M+1)*sizeof(DATA_TYPE));
  stddev = (DATA_TYPE*)malloc((M+1)*sizeof(DATA_TYPE));
  symmat = (DATA_TYPE*)malloc((M+1)*(N+1)*sizeof(DATA_TYPE));
  symmat_outputFromGpu = (DATA_TYPE*)malloc((M+1)*(N+1)*sizeof(DATA_TYPE));

  srand(1);
  init_arrays(data);
  GPU_argv_init();

#ifdef OPENME
  openme_callback("ACC_KERNEL_START", NULL);
#endif
  for (ct_repeat=0; ct_repeat<ct_repeat_max; ct_repeat++)
  {
    correlationCuda(data, mean, stddev, symmat, symmat_outputFromGpu);
  }
#ifdef OPENME
  openme_callback("ACC_KERNEL_END", NULL);
#endif

  srand(1);
  init_arrays(data);

#ifdef OPENME
  openme_callback("KERNEL_START", NULL);
#endif
  for (ct_repeat=0; ct_repeat<ct_repeat_max; ct_repeat++)
  {
    correlation(data, mean, stddev, symmat);
  }
#ifdef OPENME
  openme_callback("KERNEL_END", NULL);
#endif

  compareResults(symmat, symmat_outputFromGpu);

  free(data);
  free(mean);
  free(stddev);
  free(symmat);
  free(symmat_outputFromGpu);

#ifdef OPENME
  openme_callback("PROGRAM_END", NULL);
#endif

  return 0;
}

