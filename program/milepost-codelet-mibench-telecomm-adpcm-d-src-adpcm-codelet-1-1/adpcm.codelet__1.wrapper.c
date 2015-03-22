/* 
 Codelet from MILEPOST project: http://cTuning.org/project-milepost
 Updated by Grigori Fursin to work with Collective Mind Framework

 3 "./adpcm.codelet__1.wrapper.c" 3 4
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
struct adpcm_state  {
  short  valprev;
  char  index;
} ;
void  astex_codelet__1(int len, struct adpcm_state *state, int indexTable[16], int stepsizeTable[89], signed char *inp, short *outp, int step, int valpred, int index, int inputbuffer, int bufferstep);
int main(int argc, const char **argv)
{
  int  len = 1000;
  struct adpcm_state  *state;
  int  *indexTable;
  int  *stepsizeTable;
  signed char  *inp;
  short  *outp;
  int  step = 7;
  int  valpred = 0;
  int  index = 0;
  int  inputbuffer = 500;
  int  bufferstep = 0;
  void * codelet_data_file_descriptor = (void *) 0;

#ifdef OPENME
  openme_init(NULL,NULL,NULL,0);
  openme_callback("PROGRAM_START", NULL);
#endif

  if (argc < 2)
    __astex_exit_on_error("Please specify data file in command-line.", 1, argv[0]);
  codelet_data_file_descriptor = __astex_fopen(argv[1], "rb");
  
  char * state__region_buffer = (char *) __astex_memalloc(0);
  state = (struct adpcm_state *) (state__region_buffer + 0l);
  char * indexTable__region_buffer = (char *) __astex_memalloc(64);
  __astex_write_message("Reading indexTable value from %s\n", argv[1]);
  __astex_read_from_file(indexTable__region_buffer, 64, codelet_data_file_descriptor);
  indexTable = (int *) (indexTable__region_buffer + 0l);
  char * stepsizeTable__region_buffer = (char *) __astex_memalloc(356);
  __astex_write_message("Reading stepsizeTable value from %s\n", argv[1]);
  __astex_read_from_file(stepsizeTable__region_buffer, 356, codelet_data_file_descriptor);
  stepsizeTable = (int *) (stepsizeTable__region_buffer + 0l);
  char * inp__region_buffer = (char *) __astex_memalloc(500);
  __astex_write_message("Reading inp value from %s\n", argv[1]);
  __astex_read_from_file(inp__region_buffer, 500, codelet_data_file_descriptor);
  inp = (signed char *) (inp__region_buffer + 0l);
  char * outp__region_buffer = (char *) __astex_memalloc(2000);
  __astex_write_message("Reading outp value from %s\n", argv[1]);
  __astex_read_from_file(outp__region_buffer, 2000, codelet_data_file_descriptor);
  outp = (short *) (outp__region_buffer + 0l);
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
  astex_codelet__1(len, state, indexTable, stepsizeTable, inp, outp, step, valpred, index, inputbuffer, bufferstep);

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

