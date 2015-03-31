/*

 CK OpenME cTuning plugin

 cTuning plugin is used for fine-grain online application timing and tuning

 OpenME - Event-driven, plugin-based interactive interface to "open up" 
          any software (C/C++/Fortran/Java/PHP) and possibly connect it to cM

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

#define OPENME_DEBUG "OPENME_DEBUG"

char* ck_time_file="tmp-ck-timer.json";

#ifdef WINDOWS
static clock_t start=0.0, stop=0.0, acc_start=0.0, acc_stop=0.0;
#else 
static double start=0.0, stop=0.0, acc_start=0.0, acc_stop=0.0;
static struct timeval  before, after, acc_before, acc_after;
#endif
static double secs, acc_secs;

static char *env;

void clock_start(void)
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

}

void acc_clock_start(void)
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

void clock_end(void)
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

void acc_clock_end(void)
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

void program_start(void)
{
  if ( ((env = getenv(OPENME_DEBUG)) != NULL) && (atoi(env)==1) )
    printf("OpenME event: start program\n");

  printf("Start program\n");
}

void program_end(void)
{
  FILE* f;

  if ( ((env = getenv(OPENME_DEBUG)) != NULL) && (atoi(env)==1) )
    printf("OpenME event: ending program\n");

  printf("Stop program\n");

  f=fopen(ck_time_file, "w");
  if (f==NULL)
  {
    printf("Error: can't open timer file %s for writing\n", ck_time_file);
    exit(1);
  }

  fprintf(f,"{\n");
  fprintf(f,"  \"execution_time\":\"%.6lf\",\n", secs);
  fprintf(f,"  \"execution_time_extra1\":\"%.6lf\"\n", acc_secs);
  fprintf(f,"}\n");

  fclose(f);
}
