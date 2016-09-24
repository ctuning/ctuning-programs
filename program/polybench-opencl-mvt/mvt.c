/**
 * mvt.c: This file is part of the PolyBench/GPU 1.0 test suite.
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
#ifndef N
#define N 4096
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

cl_platform_id platform_id;
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int err_code;
cl_context clGPUContext;
cl_kernel clKernel1;
cl_kernel clKernel2;
cl_command_queue clCommandQue;
cl_program clProgram;

cl_mem a_mem_obj;
cl_mem x1_mem_obj;
cl_mem x2_mem_obj;
cl_mem y1_mem_obj;
cl_mem y2_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;
const int LIST_SIZE = N;
char str_temp[1024];



void compareResults(DATA_TYPE* x1, DATA_TYPE* x1_outputFromGpu, DATA_TYPE* x2, DATA_TYPE* x2_outputFromGpu)
{
	int i, fail;
	fail = 0;
	
	for (i=0; i<N; i++) 
	{
		if (percentDiff(x1[i], x1_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}

		if (percentDiff(x2[i], x2_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
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
		fprintf(stdout, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


void init_arrays(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y_1, DATA_TYPE* y_2)
{
    	int i, j;
	
	for (i=0; i<N; i++) 
   	{
        	x1[i] = 0.0;
        	x2[i] = 0.0;
		y_1[i] = 0.0;
        	y_2[i] = 0.0;
		
		for (j=0; j<N; j++)
		{
			a[i*N + j] = (DATA_TYPE)(i+j+1.0)/N;
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


void cl_mem_init(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y_1, DATA_TYPE* y_2)
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N * N, NULL, &err_code);
	x1_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &err_code);
	x2_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &err_code);
	y1_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &err_code);
	y2_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &err_code);
		
	if(err_code != CL_SUCCESS) printf("Error in creating buffers\n");

	err_code = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N * N, a, 0, NULL, NULL);
	err_code = clEnqueueWriteBuffer(clCommandQue, x1_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, x1, 0, NULL, NULL);
	err_code = clEnqueueWriteBuffer(clCommandQue, x2_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, x2, 0, NULL, NULL);
	err_code = clEnqueueWriteBuffer(clCommandQue, y1_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, y_1, 0, NULL, NULL);
	err_code = clEnqueueWriteBuffer(clCommandQue, y2_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, y_2, 0, NULL, NULL);

	if(err_code != CL_SUCCESS) printf("Error in writing buffers\n");
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
	clKernel1 = clCreateKernel(clProgram, "mvt_kernel1", &err_code);
	if(err_code != CL_SUCCESS) 
        {
          printf("Error in creating kernel1\n");
          exit(1);
        }

	clKernel2 = clCreateKernel(clProgram, "mvt_kernel2", &err_code);
	if(err_code != CL_SUCCESS) 
        {
          printf("Error in creating kernel2\n");
          exit(1);
        }

	clFinish(clCommandQue);
}


void cl_launch_kernel()
{
	double t_start, t_end;

	int n = N;

	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = LWS_X;
	localWorkSize[1] = LWS_Y;
	globalWorkSize[0] = (size_t)ceil(((float)N) / ((float)LWS_X)) * LWS_X;
	globalWorkSize[1] = 1;

//	t_start = rtclock();
	
	// Set the arguments of the kernel
	err_code =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	err_code |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&x1_mem_obj);
	err_code |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&y1_mem_obj);
	err_code |= clSetKernelArg(clKernel1, 3, sizeof(int), (void *)&n);
	if(err_code != CL_SUCCESS) 
        {
          printf("Error in seting arguments\n");
          exit(1);
        }
 
	// Execute the OpenCL kernel
	err_code = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(err_code != CL_SUCCESS)
        {
          printf("Error in launching kernel\n");
          exit(1);
        }

	// Set the arguments of the kernel
	err_code =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	err_code |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&x2_mem_obj);
	err_code |= clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void *)&y2_mem_obj);
	err_code |= clSetKernelArg(clKernel2, 3, sizeof(int), (void *)&n);
	if(err_code != CL_SUCCESS)
        {
          printf("Error in seting arguments\n");
          exit(1);
        }

	// Execute the OpenCL kernel
	err_code = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
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
	err_code = clReleaseKernel(clKernel1);
	err_code = clReleaseKernel(clKernel2);
	err_code = clReleaseProgram(clProgram);
	err_code = clReleaseMemObject(a_mem_obj);
	err_code = clReleaseMemObject(x1_mem_obj);
	err_code = clReleaseMemObject(x2_mem_obj);
	err_code = clReleaseMemObject(y1_mem_obj);
	err_code = clReleaseMemObject(y2_mem_obj);
	err_code = clReleaseCommandQueue(clCommandQue);
	err_code = clReleaseContext(clGPUContext);
	if(err_code != CL_SUCCESS) printf("Error in cleanup\n");
}


void runMvt(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y1, DATA_TYPE* y2)
{
	int i, j, k, l;
	
	for (i=0; i<N; i++) 
	{
		for (j=0; j<N; j++) 
		{
       			x1[i] = x1[i] + a[i*N + j] * y1[j];
        	}
    	}
	
	for (k=0; k<N; k++) 
	{
		for (l=0; l<N; l++) 
		{
 		       	x2[k] = x2[k] + a[k*N + l] * y2[l];
      		}
    	}
}


int main(int argc, char *argv[])
{
  /* Prepare ctuning vars */
  long ct_repeat=0;
  long ct_repeat_max=1;

  DATA_TYPE* a;
  DATA_TYPE* x1;
  DATA_TYPE* x2;
  DATA_TYPE* x1_outputFromGpu;
  DATA_TYPE* x2_outputFromGpu;
  DATA_TYPE* y_1;
  DATA_TYPE* y_2;

#ifdef OPENME
  openme_init(NULL,NULL,NULL,0);
  openme_callback("PROGRAM_START", NULL);
#endif
#ifdef XOPENME
  xopenme_init(2,0);
#endif

  /* Run kernel. */
  if (getenv("CT_REPEAT_MAIN")!=NULL) ct_repeat_max=atol(getenv("CT_REPEAT_MAIN"));

  a = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));
  x1 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
  x2 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
  x1_outputFromGpu = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
  x2_outputFromGpu = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
  y_1 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
  y_2 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));

  srand(1);
  init_arrays(a, x1, x2, y_1, y_2);
  read_cl_file(argv[1]);
  cl_initialization();
  cl_mem_init(a, x1, x2, y_1, y_2);
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

    err_code = clEnqueueReadBuffer(clCommandQue, x2_mem_obj, CL_TRUE, 0, N*sizeof(DATA_TYPE), x2_outputFromGpu, 0, NULL, NULL);
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
  init_arrays(a, x1, x2, y_1, y_2);

#ifdef OPENME
  openme_callback("KERNEL_START", NULL);
#endif
#ifdef XOPENME
  xopenme_clock_start(1);
#endif
  for (ct_repeat=0; ct_repeat<ct_repeat_max; ct_repeat++)
  {
    runMvt(a, x1, x2, y_1, y_2);
  }
#ifdef XOPENME
  xopenme_clock_end(1);
#endif
#ifdef OPENME
  openme_callback("KERNEL_END", NULL);
#endif
*/

  compareResults(x1, x1_outputFromGpu, x2, x2_outputFromGpu);
  cl_clean_up();

  free(a);
  free(x1);
  free(x2);
  free(x1_outputFromGpu);
  free(x2_outputFromGpu);
  free(y_1);
  free(y_2);

#ifdef XOPENME
  xopenme_dump_state();
  xopenme_finish();
#endif
#ifdef OPENME
  openme_callback("PROGRAM_END", NULL);
#endif

  return 0;
}

