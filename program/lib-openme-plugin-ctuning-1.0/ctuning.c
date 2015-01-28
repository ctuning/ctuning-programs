/*

 Collective Mind OpenME cTuning plugin

 cTuning plugin is used for fine-grain online application timing and tuning

 OpenME - Event-driven, plugin-based interactive interface to "open up" 
          any software (C/C++/Fortran/Java/PHP) and possibly connect it to cM

 cM - Collective Mind infrastructure to discover, collect,
      share and reuse knowledge

 Developer(s): (C) Grigori Fursin
 http://cTuning.org/lab/people/gfursin

 This library is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 2.1 of the License, or (at your option) any later version.

 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 Lesser General Public License for more details.

 You should have received a copy of the GNU Lesser General Public
 License along with this library; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

*/

#include <stdio.h>
#include <stdlib.h>
#ifdef __MINGW32__
#include <sys/time.h>
#else
#include <time.h>
#endif

#include <openme.h>
#include <cJSON.h>

#ifndef WINDOWS
#include <dlfcn.h>
#endif

static char buf[1024];
static int ibuf;
static char *bufx;
static char *bufy;
static cJSON *dummy=NULL;
static cJSON *json=NULL;
static cJSON *json1=NULL;
static cJSON *json_str=NULL;

static void clock_start(void);
static void clock_end(void);
static void acc_clock_start(void);
static void acc_clock_end(void);
static void program_start(void);
static void program_end(void);

#ifdef WINDOWS
static clock_t start=0.0, stop=0.0, acc_start=0.0, acc_stop=0.0;
#else 
static double start=0.0, stop=0.0, acc_start=0.0, acc_stop=0.0;
static struct timeval  before, after, acc_before, acc_after;
#endif
static double secs, acc_secs;

static char *env;

extern
#ifdef WINDOWS
__declspec(dllexport) 
#endif
int openme_plugin_init(struct openme_info *oi)
{
  /* FGG: We need next few lines to initialize malloc and free
     for both OpenME and cJSON from user space to be able
     to allocate memory - this is mainly needed for statically
     compiled programs. */
  struct openme_hooks oh;
  cJSON_Hooks jh;

  oh.malloc=oi->hooks->malloc;
  oh.free=oi->hooks->free;
  oh.fopen=oi->hooks->fopen;
  oh.fprintf=oi->hooks->fprintf;
  oh.fseek=oi->hooks->fseek;
  oh.ftell=oi->hooks->ftell;
  oh.fread=oi->hooks->fread;
  oh.fclose=oi->hooks->fclose;
  openme_init_hooks(&oh);

  jh.malloc_fn=oi->hooks->malloc;
  jh.free_fn=oi->hooks->free;
  cJSON_InitHooks(&jh);

  /* Register callbacks */
  openme_register_callback(oi, "KERNEL_START", clock_start);
  openme_register_callback(oi, "KERNEL_END", clock_end);
  openme_register_callback(oi, "ACC_KERNEL_START", acc_clock_start);
  openme_register_callback(oi, "ACC_KERNEL_END", acc_clock_end);
  openme_register_callback(oi, "PROGRAM_START", program_start);
  openme_register_callback(oi, "PROGRAM_END", program_end);

  /* FGG: Dummy call to cJSON to be able to use cJSON functions
     in OpenME library. I do not deallocate it here since compiler
     may perform dead code elimination. However, we should find
     a better way to tell compiler that we will use cJSON
     functions in OpenME ... */
  dummy=cJSON_CreateObject();

  return 0;
}

extern void clock_start(void)
{
#ifdef WINDOWS
  start = clock();
#else
  #ifdef __INTEL_COMPILERX
    start = (double)_rdtsc();
  #else
    gettimeofday(&before, NULL);
  #endif
#endif
  if ( ((env = getenv(OPENME_DEBUG)) != NULL) && (atoi(env)==1) )
    printf("OpenME event: start clock\n");

  if ( ((env = getenv(OPENME_DEBUG)) != NULL) && (atoi(env)==1) )
  {
    /* Test cM */

/*
    printf("********************************************************\n");
    printf("TESTING cM access ...\n");

    sprintf(buf, "{\"cm_run_module_uoa\":\"module\","
                  "\"cm_action\":\"list\","
                  "\"cm_console\":\"json\"}");
    sprintf(buf, "{\"cm_run_module_uoa\":\"module\","
                  "\"cm_action\":\"list\","
                  "\"cm_console\":\"json\"}");
    json=cJSON_Parse(buf);

    json1=cm_action(json);

    if (json1==NULL) {openme_print_error(); exit(1);}

    json=openme_get_obj(json1, "cm_return");
    if (json==NULL)
    {
      openme_set_error("OpenME error - can't find cm_return in cm_access output ...\n", NULL);
      return;
    }
    ibuf=json->valueint;

    if (ibuf!=0)
    {
      json=openme_get_obj(json1, "cm_error");
      if (json==NULL)
      {
        openme_set_error("OpenME error - can't find cm_error in cm_access output ...\n", NULL);
        return;
      }
      bufy=json->valuestring;

      if (bufy!=NULL)
        sprintf(buf, "OpenME error - cm_return (%u) >0; cm_error=%s ...\n", ibuf, bufy);
      else
        sprintf(buf, "OpenME error - cm_return (%u) >0\n", ibuf);
      printf(buf);

    }
   
    openme_print_obj(json1);
    printf("********************************************************\n");
*/
  }
}

extern void acc_clock_start(void)
{
#ifdef WINDOWS
  acc_start = clock();
#else
  #ifdef __INTEL_COMPILERX
    acc_start = (double)_rdtsc();
  #else
    gettimeofday(&acc_before, NULL);
  #endif
#endif
  if ( ((env = getenv(OPENME_DEBUG)) != NULL) && (atoi(env)==1) )
    printf("OpenME event: start accelerator clock\n");
}

extern void clock_end(void)
{
#ifdef WINDOWS
  stop = clock();
  secs = ((double)(stop - start)) / CLOCKS_PER_SEC;
#else
  #ifdef __INTEL_COMPILERX
  stop = (double)_rdtsc();
  secs = ((double)(stop - start)) / (double) getCPUFreq();
  #else
  gettimeofday(&after, NULL);
  secs = (after.tv_sec - before.tv_sec) + (after.tv_usec - before.tv_usec)/1000000.0;
  #endif
#endif
  if ( ((env = getenv(OPENME_DEBUG)) != NULL) && (atoi(env)==1) )
    printf("OpenME event: stop clock: %f\n", secs);
}

extern void acc_clock_end(void)
{
#ifdef WINDOWS
  acc_stop = clock();
  acc_secs = ((double)(acc_stop - acc_start)) / CLOCKS_PER_SEC;
#else
  #ifdef __INTEL_COMPILERX
  acc_stop = (double)_rdtsc();
  acc_secs = ((double)(acc_stop - acc_start)) / (double) getCPUFreq();
  #else
  gettimeofday(&acc_after, NULL);
  acc_secs = (acc_after.tv_sec - acc_before.tv_sec) + (acc_after.tv_usec - acc_before.tv_usec)/1000000.0;
  #endif
#endif
  if ( ((env = getenv(OPENME_DEBUG)) != NULL) && (atoi(env)==1) )
    printf("OpenME event: stop clock: %f\n", secs);
}

extern void program_start(void)
{
  if ( ((env = getenv(OPENME_DEBUG)) != NULL) && (atoi(env)==1) )
    printf("OpenME event: start program\n");
}

extern void program_end(void)
{
  int r=0;

  if ( ((env = getenv(OPENME_DEBUG)) != NULL) && (atoi(env)==1) )
    printf("OpenME event: ending program\n");

  if ((env = getenv(OPENME_OUTPUT_FILE)) != NULL)
  {
    /* FGG: During static compilation, I had to add print to buf 2 times
       otherwise always 0 - why I don't know - any suggestions are welcome */
//    r=sprintf(buf, "run_time_kernel=%f", secs);
//    r=sprintf(buf, "run_time_kernel=%f", secs);
//    json=openme_create_obj(buf);

    sprintf(buf, "{\"run_time_kernel\":\"%f\", \"run_time_acc_kernel\":\"%f\"}", secs, acc_secs);
    sprintf(buf, "{\"run_time_kernel\":\"%f\", \"run_time_acc_kernel\":\"%f\"}", secs, acc_secs);
    json=cJSON_Parse(buf);
    if (json==NULL) {openme_print_error(); exit(1);}

    if (openme_store_json_file(json, env)>0) {openme_print_error(); exit(1);}
  }

  if ( ((env = getenv(OPENME_DEBUG)) != NULL) && (atoi(env)==1) )
    printf("OpenME event: end program; kernel time in seconds = %f\n", secs);
}
