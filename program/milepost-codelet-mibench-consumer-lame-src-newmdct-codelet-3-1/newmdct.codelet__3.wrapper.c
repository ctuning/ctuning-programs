/* 
 Codelet from MILEPOST project: http://cTuning.org/project-milepost
 Updated by Grigori Fursin to work with Collective Mind Framework

 3 "./newmdct.codelet__3.wrapper.c" 3 4
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
typedef double  FLOAT8;
typedef FLOAT8  D576[576];
typedef FLOAT8  D192_3[192][3];
void  astex_codelet__3(FLOAT8 d[32], FLOAT8 *in, FLOAT8 s, FLOAT8 t, FLOAT8 *wp);
int main(int argc, const char **argv)
{
  FLOAT8  *d;
  FLOAT8  *in;
  FLOAT8  s = 0.000000 ;
  FLOAT8  t = 0.000000 ;
  FLOAT8  *wp;
  void * codelet_data_file_descriptor = (void *) 0;

#ifdef OPENME
  openme_init(NULL,NULL,NULL,0);
  openme_callback("PROGRAM_START", NULL);
#endif

  if (argc < 2)
    __astex_exit_on_error("Please specify data file in command-line.", 1, argv[0]);
  codelet_data_file_descriptor = __astex_fopen(argv[1], "rb");
  
  char * d__region_buffer = (char *) __astex_memalloc(18432);
  __astex_write_message("Reading d value from %s\n", argv[1]);
  __astex_read_from_file(d__region_buffer, 18432, codelet_data_file_descriptor);
  d = (FLOAT8 *) (d__region_buffer + 4608l);
  char * in__region_buffer = (char *) __astex_memalloc(1152);
  __astex_write_message("Reading in value from %s\n", argv[1]);
  __astex_read_from_file(in__region_buffer, 1152, codelet_data_file_descriptor);
  in = (FLOAT8 *) (in__region_buffer + 616l);
  char * wp__region_buffer = (char *) __astex_memalloc(3968);
  __astex_write_message("Reading wp value from %s\n", argv[1]);
  __astex_read_from_file(wp__region_buffer, 3968, codelet_data_file_descriptor);
  wp = (FLOAT8 *) (wp__region_buffer + 0l);
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
  astex_codelet__3(d, in, s, t, wp);

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

