/**
 * gramschmidt.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define M 128 // 2048
#endif
#ifndef N
#define N 128 // 2048
#endif

/* Thread block dimensions */
#ifndef LWS_X
#define LWS_X 256
#endif
#ifndef LWS_Y
#define LWS_Y 1
#endif

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
 

cl_platform_id platform_id;
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int err_code;
cl_context clGPUContext;
cl_kernel clKernel1;
cl_kernel clKernel2;
cl_kernel clKernel3;
cl_command_queue clCommandQue;
cl_program clProgram;

cl_mem a_mem_obj;
cl_mem r_mem_obj;
cl_mem q_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;



void compareResults(DATA_TYPE* A, DATA_TYPE* A_outputFromGpu)
{
	int i, j, fail;
	fail = 0;

	for (i=0; i < M; i++) 
	{
		for (j=0; j < N; j++) 
		{
			if (percentDiff(A[i*N + j], A_outputFromGpu[i*N + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{				
				fail++;
				printf("i: %d j: %d \n1: %f\n 2: %f\n", i, j, A[i*N + j], A_outputFromGpu[i*N + j]);
			}
		}
	}
	
	// Print results
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


void init_array(DATA_TYPE* A)
{
	int i, j;

	for (i = 0; i < M; i++)
	{
		for (j = 0; j < N; j++)
		{
			A[i*N + j] = ((DATA_TYPE) (i+1)*(j+1)) / (M+1);
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


void cl_mem_init(DATA_TYPE* A)
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M * N, NULL, &err_code);
	r_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M * N, NULL, &err_code);
	q_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * M * N, NULL, &err_code);
	
	if(err_code != CL_SUCCESS) printf("Error in creating buffers\n");

	err_code = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * M * N, A, 0, NULL, NULL);
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
	clKernel1 = clCreateKernel(clProgram, "gramschmidt_kernel1", &err_code);
	if(err_code != CL_SUCCESS) 
        {
          printf("Error in creating kernel1\n");
          exit(1);
        }

	clKernel2 = clCreateKernel(clProgram, "gramschmidt_kernel2", &err_code);
	if(err_code != CL_SUCCESS) 
        {
          printf("Error in creating kernel2\n");
          exit(1);
        }

	clKernel3 = clCreateKernel(clProgram, "gramschmidt_kernel3", &err_code);
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

	int k;

	size_t localWorkSize[2], globalWorkSizeKernel1[2], globalWorkSizeKernel2[2], globalWorkSizeKernel3[2];

	localWorkSize[0] = DIM_THREAD_BLOCK_X;
	localWorkSize[1] = DIM_THREAD_BLOCK_Y;
	globalWorkSizeKernel1[0] = DIM_THREAD_BLOCK_X;
	globalWorkSizeKernel1[1] = DIM_THREAD_BLOCK_Y;
	globalWorkSizeKernel2[0] = (size_t)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_X)) * DIM_THREAD_BLOCK_X;
	globalWorkSizeKernel2[1] = 1;
	globalWorkSizeKernel3[0] = (size_t)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_X)) * DIM_THREAD_BLOCK_X;
	globalWorkSizeKernel3[1] = 1;

//	t_start = rtclock();
	
	for (k = 0; k < N; k++)
	{
		// Set the arguments of the kernel
		err_code =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
		err_code =  clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&r_mem_obj);
		err_code |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&q_mem_obj);
		err_code |= clSetKernelArg(clKernel1, 3, sizeof(int), (void *)&k);
		err_code |= clSetKernelArg(clKernel1, 4, sizeof(int), (void *)&m);
		err_code |= clSetKernelArg(clKernel1, 5, sizeof(int), (void *)&n);
	
		if(err_code != CL_SUCCESS) 
                {
                  printf("Error in seting arguments1\n");
                  exit(1);
                }
	
		// Execute the OpenCL kernel
		err_code = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 1, NULL, globalWorkSizeKernel1, localWorkSize, 0, NULL, NULL);
		if(err_code != CL_SUCCESS)
                {
                  printf("Error in launching kernel1\n");
                  exit(1);
                }

		clEnqueueBarrier(clCommandQue);

		err_code =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&a_mem_obj);
		err_code =  clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&r_mem_obj);
		err_code |= clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void *)&q_mem_obj);
		err_code |= clSetKernelArg(clKernel2, 3, sizeof(int), (void *)&k);
		err_code |= clSetKernelArg(clKernel2, 4, sizeof(int), (void *)&m);
		err_code |= clSetKernelArg(clKernel2, 5, sizeof(int), (void *)&n);
	
		if(err_code != CL_SUCCESS)
                {
                  printf("Error in seting arguments1\n");
                  exit(1);
                }
	
		// Execute the OpenCL kernel
		err_code = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 1, NULL, globalWorkSizeKernel2, localWorkSize, 0, NULL, NULL);
		if(err_code != CL_SUCCESS)
                {
                  printf("Error in launching kernel2\n");
                  exit(1);
                }

		clEnqueueBarrier(clCommandQue);

		err_code =  clSetKernelArg(clKernel3, 0, sizeof(cl_mem), (void *)&a_mem_obj);
		err_code =  clSetKernelArg(clKernel3, 1, sizeof(cl_mem), (void *)&r_mem_obj);
		err_code |= clSetKernelArg(clKernel3, 2, sizeof(cl_mem), (void *)&q_mem_obj);
		err_code |= clSetKernelArg(clKernel3, 3, sizeof(int), (void *)&k);
		err_code |= clSetKernelArg(clKernel3, 4, sizeof(int), (void *)&m);
		err_code |= clSetKernelArg(clKernel3, 5, sizeof(int), (void *)&n);
	
		if(err_code != CL_SUCCESS)
                {
                  printf("Error in seting arguments1\n");
                  exit(1);
                }
	
		// Execute the OpenCL kernel
		err_code = clEnqueueNDRangeKernel(clCommandQue, clKernel3, 1, NULL, globalWorkSizeKernel3, localWorkSize, 0, NULL, NULL);
		if(err_code != CL_SUCCESS)
                {
                  printf("Error in launching kernel3\n");
                  exit(1);
                }

		clEnqueueBarrier(clCommandQue);

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
	err_code = clReleaseKernel(clKernel1);
	err_code = clReleaseKernel(clKernel2);
	err_code = clReleaseKernel(clKernel3);
	err_code = clReleaseProgram(clProgram);
	err_code = clReleaseMemObject(a_mem_obj);
	err_code = clReleaseMemObject(r_mem_obj);
	err_code = clReleaseMemObject(q_mem_obj);
	err_code = clReleaseCommandQueue(clCommandQue);
	err_code = clReleaseContext(clGPUContext);
	if(err_code != CL_SUCCESS) printf("Error in cleanup\n");
}


void gramschmidt(DATA_TYPE* A, DATA_TYPE* R, DATA_TYPE* Q)
{
	int i,j,k;
	DATA_TYPE nrm;
	for (k = 0; k < N; k++)
	{
		nrm = 0;
		for (i = 0; i < M; i++)
		{
			nrm += A[i*N + k] * A[i*N + k];
		}
		
		R[k*N + k] = sqrt(nrm);
		for (i = 0; i < M; i++)
		{
			Q[i*N + k] = A[i*N + k] / R[k*N + k];
		}
		
		for (j = k + 1; j < N; j++)
		{
			R[k*N + j] = 0;
			for (i = 0; i < M; i++)
			{
				R[k*N + j] += Q[i*N + k] * A[i*N + j];
			}
			for (i = 0; i < M; i++)
			{
				A[i*N + j] = A[i*N + j] - Q[i*N + k] * R[k*N + j];
			}
		}
	}
}


int main(int argc, char *argv[])
{
  /* Prepare ctuning vars */
  long ct_repeat=0;
  long ct_repeat_max=1;

  int i;

  DATA_TYPE* A;
  DATA_TYPE* A_outputFromGpu;
  DATA_TYPE* R;
  DATA_TYPE* Q;

#ifdef OPENME
  openme_init(NULL,NULL,NULL,0);
  openme_callback("PROGRAM_START", NULL);
#endif
#ifdef XOPENME
  xopenme_init(2,0);
#endif

  /* Run kernel. */
  if (getenv("CT_REPEAT_MAIN")!=NULL) ct_repeat_max=atol(getenv("CT_REPEAT_MAIN"));

  A = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));
  A_outputFromGpu = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));
  R = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));  
  Q = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));  

  srand(1);
  init_array(A);
  read_cl_file(argv[1]);
  cl_initialization();
  cl_mem_init(A);
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

    err_code = clEnqueueReadBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, M*N*sizeof(DATA_TYPE), A_outputFromGpu, 0, NULL, NULL);
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
  init_array(A);

#ifdef OPENME
  openme_callback("KERNEL_START", NULL);
#endif
#ifdef XOPENME
  xopenme_clock_start(1);
#endif
  for (ct_repeat=0; ct_repeat<ct_repeat_max; ct_repeat++)
  {
    gramschmidt(A, R, Q);
  }
#ifdef XOPENME
  xopenme_clock_end(1);
#endif
#ifdef OPENME
  openme_callback("KERNEL_END", NULL);
#endif
*/

  compareResults(A, A_outputFromGpu);
  cl_clean_up();

  free(A);
  free(A_outputFromGpu);
  free(R);
  free(Q);  

#ifdef XOPENME
  xopenme_dump_state();
  xopenme_finish();
#endif
#ifdef OPENME
  openme_callback("PROGRAM_END", NULL);
#endif

  return 0;
}
