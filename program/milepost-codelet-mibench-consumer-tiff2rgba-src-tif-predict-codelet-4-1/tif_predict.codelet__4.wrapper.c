/* 
 Codelet from MILEPOST project: http://cTuning.org/project-milepost
 Updated by Grigori Fursin to work with Collective Mind Framework

 3 "./tif_predict.codelet__4.wrapper.c" 3 4
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
typedef unsigned int  __u_int;
typedef __u_int  u_int;
typedef int  int32;
typedef int32  tsize_t;
typedef void  *tdata_t;
typedef int32  toff_t;
typedef void  *thandle_t;
typedef tsize_t  (*TIFFReadWriteProc)(thandle_t , tdata_t , tsize_t );
void  astex_codelet__4(tsize_t cc, tsize_t stride, char *cp);
int main(int argc, const char **argv)
{
  tsize_t  cc = 486;
  tsize_t  stride = 3;
  char  *cp;
  void * codelet_data_file_descriptor = (void *) 0;

#ifdef OPENME
  openme_init(NULL,NULL,NULL,0);
  openme_callback("PROGRAM_START", NULL);
#endif

  if (argc < 2)
    __astex_exit_on_error("Please specify data file in command-line.", 1, argv[0]);
  codelet_data_file_descriptor = __astex_fopen(argv[1], "rb");
  
  char * cp__region_buffer = (char *) __astex_memalloc(7776);
  __astex_write_message("Reading cp value from %s\n", argv[1]);
  __astex_read_from_file(cp__region_buffer, 7776, codelet_data_file_descriptor);
  cp = (char *) (cp__region_buffer + 0l);
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
  astex_codelet__4(cc, stride, cp);

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

