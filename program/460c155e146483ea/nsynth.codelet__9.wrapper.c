/* 
 Codelet from MILEPOST project: http://cTuning.org/project-milepost
 Updated by Grigori Fursin to work with Collective Mind Framework

 3 "./nsynth.codelet__9.wrapper.c" 3 4
*/

#include <stdio.h>

int __astex_write_message(const char * format, ...);
int __astex_write_output(const char * format, ...);
void __astex_exit_on_error(const char * msg, int code, const char * additional_msg);
void * __astex_fopen(const char * name, const char * mode);
void * __astex_memalloc(long bytes);
void __astex_close_file(void * file);
void __astex_read_from_file(void * dest, long bytes, void * file);
int __astex_getenv_int(const char * var);
void * __astex_start_measure();
double __astex_stop_measure(void * _before);
void  astex_codelet__9(float amp_voice, float amp_aspir, float noise, float __astex_addr__glotout[1], float __astex_addr__par_glotout[1], float voice, float __astex_addr__aspiration[1]);
int main(int argc, const char **argv)
{
  float  amp_voice = 0.000000f;
  float  amp_aspir = 0.071900f;
  float  noise = 31.000000f;
  float  *__astex_addr__glotout;
  float  *__astex_addr__par_glotout;
  float  voice = 207.982208f;
  float  *__astex_addr__aspiration;
  void * codelet_data_file_descriptor = (void *) 0;

#ifdef OPENME
  openme_init(NULL,NULL,NULL,0);
  openme_callback("PROGRAM_START", NULL);
#endif

  if (argc < 2)
    __astex_exit_on_error("Please specify data file in command-line.", 1, argv[0]);
  codelet_data_file_descriptor = __astex_fopen(argv[1], "rb");
  
  char * __astex_addr__glotout__region_buffer = (char *) __astex_memalloc(4);
  __astex_write_message("Reading __astex_addr__glotout value from %s\n", argv[1]);
  __astex_read_from_file(__astex_addr__glotout__region_buffer, 4, codelet_data_file_descriptor);
  __astex_addr__glotout = (float *) (__astex_addr__glotout__region_buffer + 0l);
  char * __astex_addr__par_glotout__region_buffer = (char *) __astex_memalloc(4);
  __astex_write_message("Reading __astex_addr__par_glotout value from %s\n", argv[1]);
  __astex_read_from_file(__astex_addr__par_glotout__region_buffer, 4, codelet_data_file_descriptor);
  __astex_addr__par_glotout = (float *) (__astex_addr__par_glotout__region_buffer + 0l);
  char * __astex_addr__aspiration__region_buffer = (char *) __astex_memalloc(4);
  __astex_write_message("Reading __astex_addr__aspiration value from %s\n", argv[1]);
  __astex_read_from_file(__astex_addr__aspiration__region_buffer, 4, codelet_data_file_descriptor);
  __astex_addr__aspiration = (float *) (__astex_addr__aspiration__region_buffer + 0l);
  void * _astex_timeval_before = __astex_start_measure();
  int _astex_wrap_loop = __astex_getenv_int("CT_REPEAT_MAIN");
  if (! _astex_wrap_loop)
    _astex_wrap_loop = 1;

#ifdef OPENME
  openme_callback("KERNEL_START", NULL);
#endif

  while (_astex_wrap_loop > 0)
  {
    --_astex_wrap_loop;
  astex_codelet__9(amp_voice, amp_aspir, noise, __astex_addr__glotout, __astex_addr__par_glotout, voice, __astex_addr__aspiration);

  }

#ifdef OPENME
  openme_callback("KERNEL_END", NULL);
#endif

  __astex_write_output("Total time: %lf\n", __astex_stop_measure(_astex_timeval_before));


#ifdef OPENME
  openme_callback("PROGRAM_END", NULL);
#endif

  return 0;
}

