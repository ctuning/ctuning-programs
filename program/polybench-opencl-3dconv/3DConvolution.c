/**
 * 3DConvolution.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define MAX_SOURCE_SIZE (0x100000)

/* Problem size */
#ifndef NI
#define NI 64 //256
#endif
#ifndef NJ
#define NJ 64 //256
#endif
#ifndef NK
#define NK 64 //256
#endif

/* Thread block dimensions */
#ifndef LWS_X
#define LWS_X 8
#endif
#ifndef LWS_Y
#define LWS_Y 8
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
cl_kernel clKernel;
cl_command_queue clCommandQue;
cl_program clProgram;
cl_mem a_mem_obj;
cl_mem b_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;



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


void init(DATA_TYPE* A)
{
	int i, j, k;

	for (i = 0; i < NI; ++i)
    	{
		for (j = 0; j < NJ; ++j)
		{
			for (k = 0; k < NK; ++k)
			{
				A[i*(NK * NJ) + j*NK + k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
			}
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


void cl_mem_init(DATA_TYPE* A, DATA_TYPE* B)
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY, sizeof(DATA_TYPE) * NI * NJ * NK, NULL, &err_code);
	b_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NI * NJ * NK, NULL, &err_code);
	
	if(err_code != CL_SUCCESS) printf("Error in creating buffers\n");

	err_code = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NJ * NK, A, 0, NULL, NULL);
	err_code = clEnqueueWriteBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NJ * NK, B, 0, NULL, NULL);
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
	clKernel = clCreateKernel(clProgram, "Convolution3D_kernel", &err_code);
	if(err_code != CL_SUCCESS) 
        {
          printf("Error in creating kernel\n");
          exit(1);
        }

	clFinish(clCommandQue);
}

void cl_launch_kernel()
{
	double t_start, t_end;

	int ni = NI;
	int nj = NJ;
	int nk = NK;

	int i;

	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = LWS_X;
	localWorkSize[1] = LWS_Y;
	globalWorkSize[0] = (size_t)ceil(((float)NK) / ((float)LWS_X)) * LWS_X;
	globalWorkSize[1] = (size_t)ceil(((float)NJ) / ((float)LWS_Y)) * LWS_Y;

//	t_start = rtclock();
	
	// Set the arguments of the kernel
	err_code =  clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	err_code |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	err_code |= clSetKernelArg(clKernel, 2, sizeof(int), &ni);
	err_code |= clSetKernelArg(clKernel, 3, sizeof(int), &nj);
	err_code |= clSetKernelArg(clKernel, 4, sizeof(int), &nk);
	if(err_code != CL_SUCCESS)
        {
          printf("Error in seting arguments\n");
          exit(1);
        }
	
	for (i = 1; i < NI - 1; ++i) // 0
	{
		// set the current value of 'i' for the argument in the kernel
		err_code |= clSetKernelArg(clKernel, 5, sizeof(int), &i);

		// Execute the OpenCL kernel
		err_code = clEnqueueNDRangeKernel(clCommandQue, clKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	}

	if(err_code != CL_SUCCESS)
        {
          printf("Error in launching kernel\n");
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
	err_code = clReleaseKernel(clKernel);
	err_code = clReleaseProgram(clProgram);
	err_code = clReleaseMemObject(a_mem_obj);
	err_code = clReleaseMemObject(b_mem_obj);
	err_code = clReleaseCommandQueue(clCommandQue);
	err_code = clReleaseContext(clGPUContext);
	if(err_code != CL_SUCCESS) printf("Error in cleanup\n");
}


void compareResults(DATA_TYPE* B, DATA_TYPE* B_outputFromGpu)
{
	int i, j, k, fail;
	fail = 0;
	
	// Compare result from cpu and gpu...
	for (i = 1; i < NI - 1; ++i) // 0
	{
		for (j = 1; j < NJ - 1; ++j) // 1
		{
			for (k = 1; k < NK - 1; ++k) // 2
			{
				if (percentDiff(B[i*(NK * NJ) + j*NK + k], B_outputFromGpu[i*(NK * NJ) + j*NK + k]) > PERCENT_DIFF_ERROR_THRESHOLD)
				{
					fail++;
				}
			}	
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


void conv3D(DATA_TYPE* A, DATA_TYPE* B)
{
	int i, j, k;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +2;  c21 = +5;  c31 = -8;
	c12 = -3;  c22 = +6;  c32 = -9;
	c13 = +4;  c23 = +7;  c33 = +10;

	for (i = 1; i < NI - 1; ++i) // 0
	{
		for (j = 1; j < NJ - 1; ++j) // 1
		{
			for (k = 1; k < NK -1; ++k) // 2
			{
				//printf("i:%d\nj:%d\nk:%d\n", i, j, k);
				B[i*(NK * NJ) + j*NK + k] = c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c21 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c23 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c31 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c33 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c12 * A[(i + 0)*(NK * NJ) + (j - 1)*NK + (k + 0)]  +  c22 * A[(i + 0)*(NK * NJ) + (j + 0)*NK + (k + 0)]   
					     +   c32 * A[(i + 0)*(NK * NJ) + (j + 1)*NK + (k + 0)]  +  c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]  
					     +   c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]  +  c21 * A[(i - 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]  
					     +   c23 * A[(i + 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]  +  c31 * A[(i - 1)*(NK * NJ) + (j + 1)*NK + (k + 1)]  
					     +   c33 * A[(i + 1)*(NK * NJ) + (j + 1)*NK + (k + 1)];
			}
		}
	}
}


int main(int argc, char *argv[])
{
  /* Prepare ctuning vars */
  long ct_repeat=0;
  long ct_repeat_max=1;

  DATA_TYPE* A;
  DATA_TYPE* B;
  DATA_TYPE* B_outputFromGpu;

  int i;

#ifdef OPENME
  openme_init(NULL,NULL,NULL,0);
  openme_callback("PROGRAM_START", NULL);
#endif
#ifdef XOPENME
  xopenme_init(2,0);
#endif

  /* Run kernel. */
  if (getenv("CT_REPEAT_MAIN")!=NULL) ct_repeat_max=atol(getenv("CT_REPEAT_MAIN"));

  A = (DATA_TYPE*)malloc(NI*NJ*NK*sizeof(DATA_TYPE));
  B = (DATA_TYPE*)malloc(NI*NJ*NK*sizeof(DATA_TYPE));
  B_outputFromGpu = (DATA_TYPE*)malloc(NI*NJ*NK*sizeof(DATA_TYPE));

  srand(1);
  init(A);
  read_cl_file(argv[1]);
  cl_initialization();
  cl_mem_init(A, B);
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

    err_code = clEnqueueReadBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, NI * NJ * NK * sizeof(DATA_TYPE), B_outputFromGpu, 0, NULL, NULL); 
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
  init(A);

#ifdef OPENME
  openme_callback("KERNEL_START", NULL);
#endif
#ifdef XOPENME
  xopenme_clock_start(1);
#endif
  for (ct_repeat=0; ct_repeat<ct_repeat_max; ct_repeat++)
  {
    conv3D(A, B);
  }
#ifdef XOPENME
  xopenme_clock_end(1);
#endif
#ifdef OPENME
  openme_callback("KERNEL_END", NULL);
#endif
*/

  compareResults(B, B_outputFromGpu);
  cl_clean_up();

  free(A);
  free(B);
  free(B_outputFromGpu);

#ifdef XOPENME
  xopenme_dump_state();
  xopenme_finish();
#endif
#ifdef OPENME
  openme_callback("PROGRAM_END", NULL);
#endif

  return 0;
}

