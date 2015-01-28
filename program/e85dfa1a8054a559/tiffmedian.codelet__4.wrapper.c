/* 
 Codelet from MILEPOST project: http://cTuning.org/project-milepost
 Updated by Grigori Fursin to work with Collective Mind Framework

 3 "./tiffmedian.codelet__4.wrapper.c" 3 4
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
typedef unsigned int  uint32;
typedef uint32  ttag_t;
typedef uint32  tstrip_t;
typedef uint32  ttile_t;
typedef struct colorbox  {
  struct colorbox  *next, *prev;
  int  rmin, rmax;
  int  gmin, gmax;
  int  bmin, bmax;
  int  total;
} Colorbox;
void  astex_codelet__4(Colorbox *box, int histogram[(1L << 5)][(1L << 5)][(1L << 5)], uint32 imagewidth, unsigned char *inptr);
int main(int argc, const char **argv)
{
  Colorbox  *box;
  int  *histogram;
  uint32  imagewidth = 162u;
  unsigned char  *inptr;
  void * codelet_data_file_descriptor = (void *) 0;

#ifdef OPENME
  openme_init(NULL,NULL,NULL,0);
  openme_callback("PROGRAM_START", NULL);
#endif

  if (argc < 2)
    __astex_exit_on_error("Please specify data file in command-line.", 1, argv[0]);
  codelet_data_file_descriptor = __astex_fopen(argv[1], "rb");
  
  char * box__region_buffer = (char *) __astex_memalloc(12288);
  __astex_write_message("Reading box value from %s\n", argv[1]);
  __astex_read_from_file(box__region_buffer, 12288, codelet_data_file_descriptor);
  box = (Colorbox *) (box__region_buffer + 0l);
  char * histogram__region_buffer = (char *) __astex_memalloc(131072);
  __astex_write_message("Reading histogram value from %s\n", argv[1]);
  __astex_read_from_file(histogram__region_buffer, 131072, codelet_data_file_descriptor);
  histogram = (int *) (histogram__region_buffer + 0l);
  char * inptr__region_buffer = (char *) __astex_memalloc(486);
  __astex_write_message("Reading inptr value from %s\n", argv[1]);
  __astex_read_from_file(inptr__region_buffer, 486, codelet_data_file_descriptor);
  inptr = (unsigned char *) (inptr__region_buffer + 0l);
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
  astex_codelet__4(box, histogram, imagewidth, inptr);

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

