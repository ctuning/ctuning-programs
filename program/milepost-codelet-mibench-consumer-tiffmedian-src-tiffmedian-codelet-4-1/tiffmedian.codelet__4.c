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

#pragma hmpp astex_codelet__4 codelet &
#pragma hmpp astex_codelet__4 , args[inptr].io=in &
#pragma hmpp astex_codelet__4 , args[histogram].io=inout &
#pragma hmpp astex_codelet__4 , args[box].io=inout &
#pragma hmpp astex_codelet__4 , target=C &
#pragma hmpp astex_codelet__4 , version=1.4.0

void astex_codelet__4(Colorbox *box, int histogram[(1L << 5)][(1L << 5)][(1L << 5)], uint32 imagewidth, unsigned char *inptr)
{
  uint32  j;
  int  blue;
  int  green;
  int  red;
astex_thread_begin:  {
    for (j = imagewidth ; j-- > 0 ; )
      {
        red = *inptr++ >> (8 - 5);
        green = *inptr++ >> (8 - 5);
        blue = *inptr++ >> (8 - 5);
        if (red < box->rmin)
          box->rmin = red;
        if (red > box->rmax)
          box->rmax = red;
        if (green < box->gmin)
          box->gmin = green;
        if (green > box->gmax)
          box->gmax = green;
        if (blue < box->bmin)
          box->bmin = blue;
        if (blue > box->bmax)
          box->bmax = blue;
        histogram[red][green][blue]++;
      }
  }
astex_thread_end:;
}

