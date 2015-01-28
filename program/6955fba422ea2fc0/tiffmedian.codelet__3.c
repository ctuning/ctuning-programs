typedef unsigned int  uint32;

typedef uint32  ttag_t;

typedef uint32  tstrip_t;

typedef uint32  ttile_t;

#pragma hmpp astex_codelet__3 codelet &
#pragma hmpp astex_codelet__3 , args[inptr].io=in &
#pragma hmpp astex_codelet__3 , args[outptr].io=inout &
#pragma hmpp astex_codelet__3 , args[histogram].io=in &
#pragma hmpp astex_codelet__3 , target=C &
#pragma hmpp astex_codelet__3 , version=1.4.0

void astex_codelet__3(int histogram[(1L << 5)][(1L << 5)][(1L << 5)], uint32 imagewidth, unsigned char *outptr, unsigned char *inptr)
{
  int  blue;
  int  green;
  int  red;
  uint32  j;
astex_thread_begin:  {
    for (j = 0 ; j < imagewidth ; j++)
      {
        red = *inptr++ >> (8 - 5);
        green = *inptr++ >> (8 - 5);
        blue = *inptr++ >> (8 - 5);
        *outptr++ = histogram[red][green][blue];
      }
  }
astex_thread_end:;
}

