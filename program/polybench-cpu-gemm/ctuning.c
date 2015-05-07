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

#define NTIMERS 1

#ifdef WINDOWS
static clock_t start[NTIMERS];
#else 
static double start[NTIMERS];
static struct timeval  before[NTIMERS], after;
#endif
static double secs[NTIMERS];

static char *env;

void clock_start(int timer)
{
#ifdef WINDOWS
  start[timer] = clock();
#else
  #ifdef __INTEL_COMPILERX
    start[timer] = (double)_rdtsc();
  #else
    gettimeofday(&before[timer], NULL);
  #endif
#endif
  if ( ((env = getenv(OPENME_DEBUG)) != NULL) && (atoi(env)==1) )
    printf("OpenME event: start clock\n");

}

void clock_end(int timer)
{
#ifdef WINDOWS
  secs[timer] = ((double)(clock() - start[timer])) / CLOCKS_PER_SEC;
#else
  #ifdef __INTEL_COMPILERX
  secs[timer] = ((double)((double)_rdtsc() - start[timer])) / (double) getCPUFreq();
  #else
  gettimeofday(&after, NULL);
  secs[timer] = (after.tv_sec - before[timer].tv_sec) + (after.tv_usec - before[timer].tv_usec)/1000000.0;
  #endif
#endif
  if ( ((env = getenv(OPENME_DEBUG)) != NULL) && (atoi(env)==1) )
    printf("OpenME event: stop clock: %f\n", secs[timer]);
}

void program_start(void)
{
  int timer;

  if ( ((env = getenv(OPENME_DEBUG)) != NULL) && (atoi(env)==1) )
    printf("OpenME event: start program\n");

  printf("Start program\n");

  for (timer=0; timer<NTIMERS; timer++)
  {
    secs[timer] = 0.0;
    start[timer] = 0.0;
  }
}

void program_end(void)
{
  FILE* f;
  int timer;

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
  fprintf(f," \"execution_time\":%.6lf,\n", secs[0]);
  fprintf(f," \"kernel_execution_time\":[\n");
  for (timer=0; timer<NTIMERS; timer++) 
  {
    fprintf(f,"    %.6lf", secs[timer]);
    if (timer!=(NTIMERS-1)) fprintf(f, ",");
    fprintf(f, "\n");
  }
  fprintf(f," ]\n");
  fprintf(f,"}\n");

  fclose(f);
}
