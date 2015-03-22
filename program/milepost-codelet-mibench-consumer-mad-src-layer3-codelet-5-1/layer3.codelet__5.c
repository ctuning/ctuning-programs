typedef signed int  mad_fixed_t;

typedef mad_fixed_t  mad_sample_t;

#pragma hmpp astex_codelet__5 codelet &
#pragma hmpp astex_codelet__5 , args[window_s].io=in &
#pragma hmpp astex_codelet__5 , args[window_l].io=in &
#pragma hmpp astex_codelet__5 , args[z].io=inout &
#pragma hmpp astex_codelet__5 , target=C &
#pragma hmpp astex_codelet__5 , version=1.4.0

void astex_codelet__5(mad_fixed_t z[36], unsigned int block_type, const mad_fixed_t window_l[36], const mad_fixed_t window_s[12])
{
  unsigned int  i;
astex_thread_begin:  {
    switch (block_type)      
        {
          case 0:
          for (i = 0 ; i < 36 ; i += 4)
            {
              z[i + 0] = ((((z[i + 0]) + (1L << 11)) >> 12) * (((window_l[i + 0]) + (1L << 15)) >> 16));
              z[i + 1] = ((((z[i + 1]) + (1L << 11)) >> 12) * (((window_l[i + 1]) + (1L << 15)) >> 16));
              z[i + 2] = ((((z[i + 2]) + (1L << 11)) >> 12) * (((window_l[i + 2]) + (1L << 15)) >> 16));
              z[i + 3] = ((((z[i + 3]) + (1L << 11)) >> 12) * (((window_l[i + 3]) + (1L << 15)) >> 16));
            }
          break;
          case 1:
          for (i = 0 ; i < 18 ; ++i)
            z[i] = ((((z[i]) + (1L << 11)) >> 12) * (((window_l[i]) + (1L << 15)) >> 16));
          for (i = 24 ; i < 30 ; ++i)
            z[i] = ((((z[i]) + (1L << 11)) >> 12) * (((window_s[i - 18]) + (1L << 15)) >> 16));
          for (i = 30 ; i < 36 ; ++i)
            z[i] = 0;
          break;
          case 3:
          for (i = 0 ; i < 6 ; ++i)
            z[i] = 0;
          for (i = 6 ; i < 12 ; ++i)
            z[i] = ((((z[i]) + (1L << 11)) >> 12) * (((window_s[i - 6]) + (1L << 15)) >> 16));
          for (i = 18 ; i < 36 ; ++i)
            z[i] = ((((z[i]) + (1L << 11)) >> 12) * (((window_l[i]) + (1L << 15)) >> 16));
          break;
        }
  }
astex_thread_end:;
}

