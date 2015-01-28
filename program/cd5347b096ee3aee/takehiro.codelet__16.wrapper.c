/* 
 Codelet from MILEPOST project: http://cTuning.org/project-milepost
 Updated by Grigori Fursin to work with Collective Mind Framework

 3 "./takehiro.codelet__16.wrapper.c" 3 4
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
typedef struct   {
  unsigned  part2_3_length;
  unsigned  big_values;
  unsigned  count1;
  unsigned  global_gain;
  unsigned  scalefac_compress;
  unsigned  window_switching_flag;
  unsigned  block_type;
  unsigned  mixed_block_flag;
  unsigned  table_select[3];
  int  subblock_gain[3];
  unsigned  region0_count;
  unsigned  region1_count;
  unsigned  preflag;
  unsigned  scalefac_scale;
  unsigned  count1table_select;
  unsigned  part2_length;
  unsigned  sfb_lmax;
  unsigned  sfb_smax;
  unsigned  count1bits;
  unsigned  *sfb_partition_table;
  unsigned  slen[4];
} gr_info;
struct huffcodetab  {
  unsigned int  xlen;
  unsigned int  linmax;
  unsigned long int  *table;
  unsigned char  *hlen;
} ;
void  astex_codelet__16(int ix[576], gr_info *gi, struct huffcodetab ht[34], int __astex_addr__i[1], int __astex_addr__bits[1], int __astex_addr__astex_what_return[1], int __astex_addr__astex_do_return[1]);
int main(int argc, const char **argv)
{
  int  *ix;
  gr_info  *gi;
  struct huffcodetab  *ht;
  int  *__astex_addr__i;
  int  *__astex_addr__bits;
  int  *__astex_addr__astex_what_return;
  int  *__astex_addr__astex_do_return;
  void * codelet_data_file_descriptor = (void *) 0;

#ifdef OPENME
  openme_init(NULL,NULL,NULL,0);
  openme_callback("PROGRAM_START", NULL);
#endif

  if (argc < 2)
    __astex_exit_on_error("Please specify data file in command-line.", 1, argv[0]);
  codelet_data_file_descriptor = __astex_fopen(argv[1], "rb");
  
  char * ix__region_buffer = (char *) __astex_memalloc(2304);
  __astex_write_message("Reading ix value from %s\n", argv[1]);
  __astex_read_from_file(ix__region_buffer, 2304, codelet_data_file_descriptor);
  ix = (int *) (ix__region_buffer + 0l);
  char * gi__region_buffer = (char *) __astex_memalloc(0);
  gi = (gr_info *) (gi__region_buffer + 0l);
  char * ht__region_buffer = (char *) __astex_memalloc(816);
  __astex_write_message("Reading ht value from %s\n", argv[1]);
  __astex_read_from_file(ht__region_buffer, 816, codelet_data_file_descriptor);
  ht = (struct huffcodetab *) (ht__region_buffer + 0l);
  char * __astex_addr__i__region_buffer = (char *) __astex_memalloc(4);
  __astex_write_message("Reading __astex_addr__i value from %s\n", argv[1]);
  __astex_read_from_file(__astex_addr__i__region_buffer, 4, codelet_data_file_descriptor);
  __astex_addr__i = (int *) (__astex_addr__i__region_buffer + 0l);
  char * __astex_addr__bits__region_buffer = (char *) __astex_memalloc(4);
  __astex_write_message("Reading __astex_addr__bits value from %s\n", argv[1]);
  __astex_read_from_file(__astex_addr__bits__region_buffer, 4, codelet_data_file_descriptor);
  __astex_addr__bits = (int *) (__astex_addr__bits__region_buffer + 0l);
  char * __astex_addr__astex_what_return__region_buffer = (char *) __astex_memalloc(4);
  __astex_write_message("Reading __astex_addr__astex_what_return value from %s\n", argv[1]);
  __astex_read_from_file(__astex_addr__astex_what_return__region_buffer, 4, codelet_data_file_descriptor);
  __astex_addr__astex_what_return = (int *) (__astex_addr__astex_what_return__region_buffer + 0l);
  char * __astex_addr__astex_do_return__region_buffer = (char *) __astex_memalloc(4);
  __astex_write_message("Reading __astex_addr__astex_do_return value from %s\n", argv[1]);
  __astex_read_from_file(__astex_addr__astex_do_return__region_buffer, 4, codelet_data_file_descriptor);
  __astex_addr__astex_do_return = (int *) (__astex_addr__astex_do_return__region_buffer + 0l);
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
  astex_codelet__16(ix, gi, ht, __astex_addr__i, __astex_addr__bits, __astex_addr__astex_what_return, __astex_addr__astex_do_return);

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

