typedef unsigned int  uint32;

typedef uint32  ttag_t;

typedef uint32  tstrip_t;

typedef uint32  ttile_t;

#pragma hmpp astex_codelet__1 codelet &
#pragma hmpp astex_codelet__1 , args[nextptr].io=inout &
#pragma hmpp astex_codelet__1 , args[thisptr].io=inout &
#pragma hmpp astex_codelet__1 , args[outptr].io=inout &
#pragma hmpp astex_codelet__1 , target=C &
#pragma hmpp astex_codelet__1 , version=1.4.0

void astex_codelet__1(uint32 imagewidth, int threshold, unsigned char *outptr, short *thisptr, short *nextptr, uint32 jmax, int lastline, int bit)
{
  int  lastpixel;
  uint32  j;
astex_thread_begin:  {
    for (j = 0 ; j < imagewidth ; ++j)
      {
        register int  v;
        lastpixel = (j == jmax);
        v = *thisptr++;
        if (v < 0)
          v = 0;
        else         if (v > 255)
          v = 255;
        if (v > threshold)
          {
            *outptr |= bit;
            v -= 255;
          }
        bit >>= 1;
        if (bit == 0)
          {
            outptr++;
            bit = 0x80;
          }
        if ( !lastpixel)
          thisptr[0] += v * 7 / 16;
        if ( !lastline)
          {
            if (j != 0)
              nextptr[-1] += v * 3 / 16;
            *nextptr++ += v * 5 / 16;
            if ( !lastpixel)
              nextptr[0] += v / 16;
          }
      }
  }
astex_thread_end:;
}

