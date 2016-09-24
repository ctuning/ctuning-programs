/**
 * covariance.c: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "polybench.h"

#ifdef OPENME
#include <openme.h>
#endif
#ifdef XOPENME
#include <xopenme.h>
#endif

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define MAX_SOURCE_SIZE (0x100000)

/* Problem size */
#ifndef M
#define M 256 //2048
#endif
#ifndef N
#define N 256 //2048
#endif

/* Thread block dimensions for kernel 1*/
#ifndef LWS_KERNEL_1_X
#define LWS_KERNEL_1_X 256
#endif
#ifndef LWS_KERNEL_1_Y
#define LWS_KERNEL_1_Y 1
#endif

/* Thread block dimensions for kernel 2*/
#ifndef LWS_KERNEL_2_X
#define LWS_KERNEL_2_X 32
#endif
#ifndef LWS_KERNEL_2_Y
#define LWS_KERNEL_2_Y 8
#endif

/* Thread block dimensions for kernel 3*/
#ifndef LWS_KERNEL_3_X
#define LWS_KERNEL_3_X 256
#endif
#ifndef LWS_KERNEL_3_Y
#define LWS_KERNEL_3_Y 1
#endif

#define sqrt_of_array_cell(x,j) sqrt(x[j])

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Can switch DATA_TYPE between float and double */
# ifndef DATA_TYPE
#  define DATA_TYPE float
# endif

char str_temp[1024];

DATA_TYPE float_n= 3214212.01;
DATA_TYPE eps=  0.005;

cl_platform_id platform_id;
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int err_code;
cl_context clGPUContext;
cl_kernel clKernel_mean;
cl_kernel clKernel_std;
cl_kernel clKernel_reduce;
cl_kernel clKernel_covar;
cl_command_queue clCommandQue;
cl_program clProgram;
cl_mem data_mem_obj;
cl_mem stddev_mem_obj;
cl_mem mean_mem_obj;
cl_mem symmat_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;



void compareResults(DATA_TYPE* symmat, DATA_TYPE* symmat_outputFromGpu)
{
	int i,j,fail;
	fail = 0;

	for (i=0; i<=M; i++)
	{
		for (j=0; j<=N; j++)
		{
			if (percentDiff(symmat[i*(N+1) + j], symmat_outputFromGpu[i*(N+1) + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;
			}			
		}
	}
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


/* Grigori changed next function to call kernel from CMD for CK */
void read_cl_file(char* fn)
{
	// Load the kernel source code into the array source_str
	fp = fopen(fn, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init_arrays(DATA_TYPE* data)
{
	int i, j;

	for (i = 0; i < M; i++)
	{
		for (j = 0; j < N; j++)
		{
			data[i*(N+1) + j] = ((DATA_TYPE) i*j) / M;
		}
	}
}


void cl_initialization()
{
  /* Grigori Fursin added support for CK widgets */
  int gpgpu_platform_id=0;
  int gpgpu_device_id=0;

  cl_platform_id* platforms;
  cl_device_id* devices;

  cl_device_id device;   

  if (getenv("CK_COMPUTE_PLATFORM_ID")!=NULL) gpgpu_platform_id=atol(getenv("CK_COMPUTE_PLATFORM_ID"));
  if (getenv("CK_COMPUTE_DEVICE_ID")!=NULL) gpgpu_device_id=atol(getenv("CK_COMPUTE_DEVICE_ID"));

  platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * gpgpu_platform_id+1);
  devices = (cl_device_id*) malloc(sizeof(cl_device_id) * gpgpu_device_id+1);

	// Get platform and device information
	err_code = clGetPlatformIDs(15, platforms, &num_platforms);
	if(err_code == CL_SUCCESS) printf("number of platforms is %d\n",num_platforms);
	else printf("Error getting platform IDs\n");

	platform_id=platforms[gpgpu_platform_id];

	err_code = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, sizeof(str_temp), str_temp,NULL);
	if(err_code == CL_SUCCESS) printf("platform name is %s\n",str_temp);
	else printf("Error getting platform name\n");

	err_code = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(str_temp), str_temp,NULL);
	if(err_code == CL_SUCCESS) printf("platform version is %s\n",str_temp);
	else printf("Error getting platform version\n");

	err_code = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 15, devices, &num_devices);
	if(err_code == CL_SUCCESS) printf("number of devices is %d\n", num_devices);
	else printf("Error getting device IDs\n");

	device_id=devices[gpgpu_device_id];

	err_code = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
	if(err_code == CL_SUCCESS) printf("device name is %s\n",str_temp);
	else printf("Error getting device name\n");
	
	// Create an OpenCL context
	clGPUContext = clCreateContext( NULL, 1, &device_id, NULL, NULL, &err_code);
	if(err_code != CL_SUCCESS) printf("Error in creating context\n");
 
	//Create a command-queue
	clCommandQue = clCreateCommandQueue(clGPUContext, device_id, 0, &err_code);
	if(err_code != CL_SUCCESS) printf("Error in creating command queue\n");
}


void cl_mem_init(DATA_TYPE* data, DATA_TYPE* symmat, DATA_TYPE* mean)
{
	data_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * (M+1) * (N+1), NULL, &err_code);
	symmat_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * (M+1) * (N+1), NULL, &err_code);
	mean_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * (M+1), NULL, &err_code);
		
	if(err_code != CL_SUCCESS) printf("Error in creating buffers\n");

	err_code = clEnqueueWriteBuffer(clCommandQue, data_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * (M+1) * (N+1), data, 0, NULL, NULL);
	err_code = clEnqueueWriteBuffer(clCommandQue, symmat_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * (M+1) * (N+1), symmat, 0, NULL, NULL);
	err_code = clEnqueueWriteBuffer(clCommandQue, mean_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * (M+1), mean, 0, NULL, NULL);
	if(err_code != CL_SUCCESS)printf("Error in writing buffers\n");
}

 
void cl_load_prog()
{
#ifdef MAC
	char *flags = "-DMAC";
#else
	char *flags = "";
#endif
        char buffer[16384];
        size_t length;

	// Create a program from the kernel source
	clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &err_code);
	if(err_code != CL_SUCCESS)
        {
          printf("Error in creating program\n");
          exit(1);
        }

	// Build the program
	err_code = clBuildProgram(clProgram, 1, &device_id, flags, NULL, NULL);
	if(err_code != CL_SUCCESS) 
        {
          printf("Error in building program (%d)\n", err_code);
          clGetProgramBuildInfo(clProgram, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
          printf("Error output:\n%s\n", buffer);
          exit(1);
        }
		
	// Create the OpenCL kernel
	clKernel_mean = clCreateKernel(clProgram, "mean_kernel", &err_code);
	if(err_code != CL_SUCCESS) 
        {
          printf("Error in creating kernel1\n");
          exit(1);
        }

	clKernel_reduce = clCreateKernel(clProgram, "reduce_kernel", &err_code);
	if(err_code != CL_SUCCESS) 
        {
          printf("Error in creating kernel2\n");
          exit(1);
        }

	clKernel_covar = clCreateKernel(clProgram, "covar_kernel", &err_code);
	if(err_code != CL_SUCCESS) 
        {
          printf("Error in creating kernel3\n");
          exit(1);
        }

	clFinish(clCommandQue);
}

void cl_launch_kernel()
{
	double t_start, t_end;

	int m = M;
	int n = N;

	size_t localWorkSize_Kernel1[2], globalWorkSize_Kernel1[2];
	size_t localWorkSize_Kernel2[2], globalWorkSize_Kernel2[2];
	size_t localWorkSize_Kernel3[2], globalWorkSize_Kernel3[2];

	localWorkSize_Kernel1[0] = LWS_KERNEL_1_X;
	localWorkSize_Kernel1[1] = LWS_KERNEL_1_Y;
	globalWorkSize_Kernel1[0] = (size_t)ceil(((float)M) / ((float)LWS_KERNEL_1_X)) * LWS_KERNEL_1_X;
	globalWorkSize_Kernel1[1] = 1;

	localWorkSize_Kernel2[0] = LWS_KERNEL_2_X;
	localWorkSize_Kernel2[1] = LWS_KERNEL_2_Y;
	globalWorkSize_Kernel2[0] = (size_t)ceil(((float)M) / ((float)LWS_KERNEL_2_X)) * LWS_KERNEL_2_X;
	globalWorkSize_Kernel2[1] = (size_t)ceil(((float)N) / ((float)LWS_KERNEL_2_Y)) * LWS_KERNEL_2_Y;

	localWorkSize_Kernel3[0] = LWS_KERNEL_3_X;
	localWorkSize_Kernel3[1] = LWS_KERNEL_3_Y;
	globalWorkSize_Kernel3[0] = (size_t)ceil(((float)M) / ((float)LWS_KERNEL_3_X)) * LWS_KERNEL_3_X;
	globalWorkSize_Kernel3[1] = 1;

//	t_start = rtclock();
	
	// Set the arguments of the kernel
	err_code =  clSetKernelArg(clKernel_mean, 0, sizeof(cl_mem), (void *)&mean_mem_obj);
	err_code |= clSetKernelArg(clKernel_mean, 1, sizeof(cl_mem), (void *)&data_mem_obj);
	err_code |= clSetKernelArg(clKernel_mean, 2, sizeof(DATA_TYPE), (void *)&float_n);
	err_code |= clSetKernelArg(clKernel_mean, 3, sizeof(int), (void *)&m);
	err_code |= clSetKernelArg(clKernel_mean, 4, sizeof(int), (void *)&n);
	if(err_code != CL_SUCCESS)
        {
          printf("Error in seting arguments1\n");
          exit(1);
        }
	
	// Execute the OpenCL kernel
	err_code = clEnqueueNDRangeKernel(clCommandQue, clKernel_mean, 1, NULL, globalWorkSize_Kernel1, localWorkSize_Kernel1, 0, NULL, NULL);
	if(err_code != CL_SUCCESS)
        {
          printf("Error in launching kernel1\n");
          exit(1);
        }

	clEnqueueBarrier(clCommandQue);

	// Set the arguments of the kernel
	err_code =  clSetKernelArg(clKernel_reduce, 0, sizeof(cl_mem), (void *)&mean_mem_obj);
	err_code |= clSetKernelArg(clKernel_reduce, 1, sizeof(cl_mem), (void *)&data_mem_obj);
	err_code |= clSetKernelArg(clKernel_reduce, 2, sizeof(int), (void *)&m);
	err_code |= clSetKernelArg(clKernel_reduce, 3, sizeof(int), (void *)&n);

	if(err_code != CL_SUCCESS)
        {
          printf("Error in seting arguments2\n");
          exit(1);
        }

	// Execute the OpenCL kernel
	err_code = clEnqueueNDRangeKernel(clCommandQue, clKernel_reduce, 2, NULL, globalWorkSize_Kernel2, localWorkSize_Kernel2, 0, NULL, NULL);
	if(err_code != CL_SUCCESS)
        {
          printf("Error in launching kernel2\n");
          exit(1);
        }

	clEnqueueBarrier(clCommandQue);
	
	// Set the arguments of the kernel
	
	err_code =  clSetKernelArg(clKernel_covar, 0, sizeof(cl_mem), (void *)&symmat_mem_obj);
	err_code |= clSetKernelArg(clKernel_covar, 1, sizeof(cl_mem), (void *)&data_mem_obj);
	err_code |= clSetKernelArg(clKernel_covar, 2, sizeof(int), (void *)&m);
	err_code |= clSetKernelArg(clKernel_covar, 3, sizeof(int), (void *)&n);

	if(err_code != CL_SUCCESS)
        {
          printf("Error in seting arguments4\n");
          exit(1);
        }

	// Execute the OpenCL kernel
	err_code = clEnqueueNDRangeKernel(clCommandQue, clKernel_covar, 1, NULL, globalWorkSize_Kernel3, localWorkSize_Kernel3, 0, NULL, NULL);
	if(err_code != CL_SUCCESS)
        {
          printf("Error in launching kernel4\n");
          exit(1);
        }

	clFinish(clCommandQue);

//	t_end = rtclock();
//	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
}


void cl_clean_up()
{
	// Clean up
	err_code = clFlush(clCommandQue);
	err_code = clFinish(clCommandQue);
	err_code = clReleaseKernel(clKernel_reduce);
	err_code = clReleaseKernel(clKernel_mean);
	err_code = clReleaseKernel(clKernel_std);
	err_code = clReleaseKernel(clKernel_covar);
	err_code = clReleaseProgram(clProgram);
	err_code = clReleaseMemObject(symmat_mem_obj);
	err_code = clReleaseMemObject(data_mem_obj);
	err_code = clReleaseMemObject(mean_mem_obj);
	err_code = clReleaseMemObject(stddev_mem_obj);
	err_code = clReleaseCommandQueue(clCommandQue);
	err_code = clReleaseContext(clGPUContext);
	if(err_code != CL_SUCCESS) printf("Error in cleanup\n");
}


void covariance(DATA_TYPE* data, DATA_TYPE* symmat, DATA_TYPE* mean)
{
	int i, j, j1,j2;

  	/* Determine mean of column vectors of input data matrix */
	for (j = 1; j <= M; j++)
	{
		mean[j] = 0.0;
		for (i = 1; i <= N; i++)
		{
        		mean[j] += data[i*(M+1) + j];
		}
		mean[j] /= float_n;
	}

  	/* Center the column vectors. */
	for (i = 1; i <= N; i++)
	{
		for (j = 1; j <= M; j++)
		{
			data[i*(M+1) + j] -= mean[j];
		}
	}

  	/* Calculate the m * m covariance matrix. */
	for (j1 = 1; j1 <= M; j1++)
	{
		for (j2 = j1; j2 <= M; j2++)
   		{
       		symmat[j1*(M+1) + j2] = 0.0;
			for (i = 1; i <= N; i++)
			{
				symmat[j1*(M+1) + j2] += data[i*(M+1) + j1] * data[i*(M+1) + j2];
			}
       		symmat[j2*(M+1) + j1] = symmat[j1*(M+1) + j2];
      	}
	}
}


int main(int argc, char *argv[])
{
  /* Prepare ctuning vars */
  long ct_repeat=0;
  long ct_repeat_max=1;

  DATA_TYPE* data;
  DATA_TYPE* symmat;
  DATA_TYPE* mean;
  DATA_TYPE* symmat_outputFromGpu;	

  DATA_TYPE* stddev_deb;
  DATA_TYPE* mean_deb;
  DATA_TYPE* data_deb;

#ifdef OPENME
  openme_init(NULL,NULL,NULL,0);
  openme_callback("PROGRAM_START", NULL);
#endif
#ifdef XOPENME
  xopenme_init(2,0);
#endif

  /* Run kernel. */
  if (getenv("CT_REPEAT_MAIN")!=NULL) ct_repeat_max=atol(getenv("CT_REPEAT_MAIN"));

  data = (DATA_TYPE*)malloc((M + 1)*(N + 1)*sizeof(DATA_TYPE));
  symmat = (DATA_TYPE*)malloc((M + 1)*(M + 1)*sizeof(DATA_TYPE));
  mean = (DATA_TYPE*)malloc((M + 1)*sizeof(DATA_TYPE));
  symmat_outputFromGpu = (DATA_TYPE*)malloc((M + 1)*(M + 1)*sizeof(DATA_TYPE));	

  srand(1);
  init_arrays(data);
  read_cl_file(argv[1]);
  cl_initialization();
  cl_mem_init(data, symmat, mean);
  cl_load_prog();

#ifdef OPENME
  openme_callback("ACC_KERNEL_START", NULL);
#endif
#ifdef XOPENME
  xopenme_clock_start(0);
#endif
  for (ct_repeat=0; ct_repeat<ct_repeat_max; ct_repeat++)
  {
    cl_launch_kernel();

    err_code = clEnqueueReadBuffer(clCommandQue, symmat_mem_obj, CL_TRUE, 0, (M+1) * (N+1) * sizeof(DATA_TYPE), symmat_outputFromGpu, 0, NULL, NULL);
    if(err_code != CL_SUCCESS)
    {
      printf("Error in reading GPU mem\n");
      exit(1);
    }
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
  cl_clean_up();

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

