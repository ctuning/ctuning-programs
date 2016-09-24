/**
 * fdtd2d.c: This file is part of the PolyBench/GPU 1.0 test suite.
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

/* Problem size */
#ifndef TMAX
#define TMAX 500
#endif
#ifndef NX
#define NX 512 // 2048
#endif
#ifndef NY
#define NY 512 // 2048
#endif

/* Thread block dimensions */
#ifndef LWS_X
#define LWS_X 32
#endif
#ifndef LWS_Y
#define LWS_Y 8
#endif

#define MAX_SOURCE_SIZE (0x100000)

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

DATA_TYPE alpha = 23;
DATA_TYPE beta = 15;

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

cl_mem fict_mem_obj;
cl_mem ex_mem_obj;
cl_mem ey_mem_obj;
cl_mem hz_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;



void compareResults(DATA_TYPE* hz1, DATA_TYPE* hz2)
{
	int i, j, fail;
	fail = 0;
	
	for (i=0; i < NX; i++) 
	{
		for (j=0; j < NY; j++) 
		{
			if (percentDiff(hz1[i*NY + j], hz2[i*NY + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
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


void init_arrays(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
	int i, j;

  	for (i = 0; i < TMAX; i++)
	{
		_fict_[i] = (DATA_TYPE) i;
	}
	
	for (i = 0; i < NX; i++)
	{
		for (j = 0; j < NY; j++)
		{
			ex[i*NY + j] = ((DATA_TYPE) i*(j+1) + 1) / NX;
			ey[i*NY + j] = ((DATA_TYPE) (i-1)*(j+2) + 2) / NX;
			hz[i*NY + j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / NX;
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


void cl_mem_init(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
	fict_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * TMAX, NULL, &err_code);
	ex_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NX * (NY + 1), NULL, &err_code);
	ey_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * (NX + 1) * NY, NULL, &err_code);
	hz_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NX * NY, NULL, &err_code);
	
	if(err_code != CL_SUCCESS) printf("Error in creating buffers\n");

	err_code = clEnqueueWriteBuffer(clCommandQue, fict_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * TMAX, _fict_, 0, NULL, NULL);
	err_code = clEnqueueWriteBuffer(clCommandQue, ex_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX * (NY + 1), ex, 0, NULL, NULL);
	err_code = clEnqueueWriteBuffer(clCommandQue, ey_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * (NX + 1) * NY, ey, 0, NULL, NULL);
	err_code = clEnqueueWriteBuffer(clCommandQue, hz_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NX * NY, hz, 0, NULL, NULL);
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
	clKernel1 = clCreateKernel(clProgram, "fdtd_kernel1", &err_code);
	if(err_code != CL_SUCCESS) 
        {
          printf("Error in creating kernel1\n");
          exit(1);
        }

	clKernel2 = clCreateKernel(clProgram, "fdtd_kernel2", &err_code);
	if(err_code != CL_SUCCESS) 
        {
          printf("Error in creating kernel2\n");
          exit(1);
        }

	clKernel3 = clCreateKernel(clProgram, "fdtd_kernel3", &err_code);
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

	int nx = NX;
	int ny = NY;

	int t;

	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = LWS_X;
	localWorkSize[1] = LWS_Y;
	globalWorkSize[0] = (size_t)ceil(((float)NY) / ((float)LWS_X)) * LWS_X;
	globalWorkSize[1] = (size_t)ceil(((float)NX) / ((float)LWS_Y)) * LWS_Y;

//	t_start = rtclock();
	for(t=0;t<TMAX;t++)
	{
		// Set the arguments of the kernel
		err_code =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&fict_mem_obj);
		err_code =  clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&ex_mem_obj);
		err_code |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&ey_mem_obj);
		err_code |= clSetKernelArg(clKernel1, 3, sizeof(cl_mem), (void *)&hz_mem_obj);
		err_code |= clSetKernelArg(clKernel1, 4, sizeof(int), (void *)&t);
		err_code |= clSetKernelArg(clKernel1, 5, sizeof(int), (void *)&nx);
		err_code |= clSetKernelArg(clKernel1, 6, sizeof(int), (void *)&ny);
		
		if(err_code != CL_SUCCESS)
                {
                  printf("Error in seting arguments1\n");
                  exit(1);
                }
		// Execute the OpenCL kernel
		err_code = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		if(err_code != CL_SUCCESS)
                {
                  printf("Error in launching kernel1\n");
                  exit(1);
                }

		clEnqueueBarrier(clCommandQue);

		// Set the arguments of the kernel
		err_code =  clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *)&ex_mem_obj);
		err_code |= clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *)&ey_mem_obj);
		err_code |= clSetKernelArg(clKernel2, 2, sizeof(cl_mem), (void *)&hz_mem_obj);
		err_code |= clSetKernelArg(clKernel2, 3, sizeof(int), (void *)&nx);
		err_code |= clSetKernelArg(clKernel2, 4, sizeof(int), (void *)&ny);
		
		if(err_code != CL_SUCCESS)
                {
                  printf("Error in seting arguments1\n");
                  exit(1);
                }

		// Execute the OpenCL kernel
		err_code = clEnqueueNDRangeKernel(clCommandQue, clKernel2, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

		if(err_code != CL_SUCCESS)
                {
                  printf("Error in launching kernel1\n");
                  exit(1);
                }
		clEnqueueBarrier(clCommandQue);

		// Set the arguments of the kernel
		err_code =  clSetKernelArg(clKernel3, 0, sizeof(cl_mem), (void *)&ex_mem_obj);
		err_code |= clSetKernelArg(clKernel3, 1, sizeof(cl_mem), (void *)&ey_mem_obj);
		err_code |= clSetKernelArg(clKernel3, 2, sizeof(cl_mem), (void *)&hz_mem_obj);
		err_code |= clSetKernelArg(clKernel3, 3, sizeof(int), (void *)&nx);
		err_code |= clSetKernelArg(clKernel3, 4, sizeof(int), (void *)&ny);
		
		if(err_code != CL_SUCCESS)
                {
                  printf("Error in seting arguments1\n");
                  exit(1);
                }

		// Execute the OpenCL kernel
		err_code = clEnqueueNDRangeKernel(clCommandQue, clKernel3, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		if(err_code != CL_SUCCESS)
                {
                  printf("Error in launching kernel1\n");
                  exit(1);
                }

		clFinish(clCommandQue);
	}

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
	err_code = clReleaseMemObject(fict_mem_obj);
	err_code = clReleaseMemObject(ex_mem_obj);
	err_code = clReleaseMemObject(ey_mem_obj);
	err_code = clReleaseMemObject(hz_mem_obj);
	err_code = clReleaseCommandQueue(clCommandQue);
	err_code = clReleaseContext(clGPUContext);
	if(err_code != CL_SUCCESS) printf("Error in cleanup\n");
}


void runFdtd(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
	int t, i, j;
	
	for(t=0; t < TMAX; t++)  
	{
		for (j=0; j < NY; j++)
		{
			ey[0*NY + j] = _fict_[t];
		}
	
		for (i = 1; i < NX; i++)
		{
       			for (j = 0; j < NY; j++)
				{
       				ey[i*NY + j] = ey[i*NY + j] - 0.5*(hz[i*NY + j] - hz[(i-1)*NY + j]);
        		}
		}

		for (i = 0; i < NX; i++)
		{
       		for (j = 1; j < NY; j++)
			{
				ex[i*(NY+1) + j] = ex[i*(NY+1) + j] - 0.5*(hz[i*NY + j] - hz[i*NY + (j-1)]);
			}
		}

		for (i = 0; i < NX; i++)
		{
			for (j = 0; j < NY; j++)
			{
				hz[i*NY + j] = hz[i*NY + j] - 0.7*(ex[i*(NY+1) + (j+1)] - ex[i*(NY+1) + j] + ey[(i+1)*NY + j] - ey[i*NY + j]);
			}
		}
	}
}


int main(int argc, char *argv[])
{
  /* Prepare ctuning vars */
  long ct_repeat=0;
  long ct_repeat_max=1;

  DATA_TYPE* _fict_;
  DATA_TYPE* ex;
  DATA_TYPE* ey;
  DATA_TYPE* hz;
  DATA_TYPE* hz_outputFromGpu;

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

  _fict_ = (DATA_TYPE*)malloc(TMAX*sizeof(DATA_TYPE));
  ex = (DATA_TYPE*)malloc(NX*(NY+1)*sizeof(DATA_TYPE));
  ey = (DATA_TYPE*)malloc((NX+1)*NY*sizeof(DATA_TYPE));
  hz = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
  hz_outputFromGpu = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));

  srand(1);
  init_arrays(_fict_, ex, ey, hz);
  read_cl_file(argv[1]);
  cl_initialization();
  cl_mem_init(_fict_, ex, ey, hz);
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

    err_code = clEnqueueReadBuffer(clCommandQue, hz_mem_obj, CL_TRUE, 0, NX * NY * sizeof(DATA_TYPE), hz_outputFromGpu, 0, NULL, NULL);
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
  init_arrays(_fict_, ex, ey, hz);

#ifdef OPENME
  openme_callback("KERNEL_START", NULL);
#endif
#ifdef XOPENME
  xopenme_clock_start(1);
#endif
  for (ct_repeat=0; ct_repeat<ct_repeat_max; ct_repeat++)
  {
    runFdtd(_fict_, ex, ey, hz);
  }
#ifdef XOPENME
  xopenme_clock_end(1);
#endif
#ifdef OPENME
  openme_callback("KERNEL_END", NULL);
#endif
*/

  compareResults(hz, hz_outputFromGpu);
  cl_clean_up();

  free(_fict_);
  free(ex);
  free(ey);
  free(hz);
  free(hz_outputFromGpu);

#ifdef XOPENME
  xopenme_dump_state();
  xopenme_finish();
#endif
#ifdef OPENME
  openme_callback("PROGRAM_END", NULL);
#endif

  return 0;
}

