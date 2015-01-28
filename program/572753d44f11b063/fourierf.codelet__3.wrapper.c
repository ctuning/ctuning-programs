/* 
 Codelet from MILEPOST project: http://cTuning.org/project-milepost
 Updated by Grigori Fursin to work with Collective Mind Framework

 3 "./fourierf.codelet__3.wrapper.c" 3 4
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
void  astex_codelet__3(unsigned NumSamples, int InverseTransform, float *RealOut, float *ImagOut, unsigned BlockEnd, double angle_numerator);
int main(int argc, const char **argv)
{
  unsigned  NumSamples = 32768u;
  int  InverseTransform = 0;
  float  *RealOut;
  float  *ImagOut;
  unsigned  BlockEnd = 1u;
  double  angle_numerator = 6.283185 ;
  void * codelet_data_file_descriptor = (void *) 0;

#ifdef OPENME
  openme_init(NULL,NULL,NULL,0);
  openme_callback("PROGRAM_START", NULL);
#endif

  if (argc < 2)
    __astex_exit_on_error("Please specify data file in command-line.", 1, argv[0]);
  codelet_data_file_descriptor = __astex_fopen(argv[1], "rb");
  
  char * RealOut__region_buffer = (char *) __astex_memalloc(131072);
  __astex_write_message("Reading RealOut value from %s\n", argv[1]);
  __astex_read_from_file(RealOut__region_buffer, 131072, codelet_data_file_descriptor);
  RealOut = (float *) (RealOut__region_buffer + 0l);
  char * ImagOut__region_buffer = (char *) __astex_memalloc(131072);
  __astex_write_message("Reading ImagOut value from %s\n", argv[1]);
  __astex_read_from_file(ImagOut__region_buffer, 131072, codelet_data_file_descriptor);
  ImagOut = (float *) (ImagOut__region_buffer + 0l);
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
  astex_codelet__3(NumSamples, InverseTransform, RealOut, ImagOut, BlockEnd, angle_numerator);

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

