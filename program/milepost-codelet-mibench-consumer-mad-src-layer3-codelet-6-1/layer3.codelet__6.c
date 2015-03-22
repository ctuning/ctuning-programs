typedef signed int  mad_fixed_t;

typedef signed int  mad_fixed64hi_t;

typedef unsigned int  mad_fixed64lo_t;

typedef mad_fixed_t  mad_sample_t;

#pragma hmpp astex_codelet__6 codelet &
#pragma hmpp astex_codelet__6 , args[ca].io=in &
#pragma hmpp astex_codelet__6 , args[cs].io=in &
#pragma hmpp astex_codelet__6 , args[xr].io=inout &
#pragma hmpp astex_codelet__6 , target=C &
#pragma hmpp astex_codelet__6 , version=1.4.0

void astex_codelet__6(mad_fixed_t xr[576], const mad_fixed_t cs[8], const mad_fixed_t ca[8], const mad_fixed_t *bound)
{
  int  i;
astex_thread_begin:  {
    for (xr += 18 ; xr < bound ; xr += 18)
      {
        for (i = 0 ; i < 8 ; ++i)
          {
            register mad_fixed_t  a, b;
            register mad_fixed64hi_t  hi;
            register mad_fixed64lo_t  lo;
            a = xr[-1 - i];
            b = xr[i];
            ((lo) = (((((a)) + (1L << 11)) >> 12) * ((((cs[i])) + (1L << 15)) >> 16)));
            ((lo) += ((((( -b)) + (1L << 11)) >> 12) * ((((ca[i])) + (1L << 15)) >> 16)));
            xr[-1 - i] = ((void ) (hi), (mad_fixed_t ) (lo));
            ((lo) = (((((b)) + (1L << 11)) >> 12) * ((((cs[i])) + (1L << 15)) >> 16)));
            ((lo) += (((((a)) + (1L << 11)) >> 12) * ((((ca[i])) + (1L << 15)) >> 16)));
            xr[i] = ((void ) (hi), (mad_fixed_t ) (lo));
          }
      }
  }
astex_thread_end:;
}

