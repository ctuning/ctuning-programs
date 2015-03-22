#pragma hmpp astex_codelet__5 codelet &
#pragma hmpp astex_codelet__5 , args[__astex_addr__nrand].io=out &
#pragma hmpp astex_codelet__5 , args[__astex_addr__frics].io=inout &
#pragma hmpp astex_codelet__5 , args[__astex_addr__sourc].io=inout &
#pragma hmpp astex_codelet__5 , args[__astex_addr__noise].io=out &
#pragma hmpp astex_codelet__5 , args[__astex_addr__nlast].io=inout &
#pragma hmpp astex_codelet__5 , target=C &
#pragma hmpp astex_codelet__5 , version=1.4.0

void astex_codelet__5(long nper, long nmod, float amp_frica, float __astex_addr__nlast[1], unsigned long seed, float __astex_addr__noise[1], float __astex_addr__sourc[1], float __astex_addr__frics[1], long __astex_addr__nrand[1])
{
  long  nrand;
  float  frics = __astex_addr__frics[0];
  float  sourc = __astex_addr__sourc[0];
  float  noise;
  float  nlast = __astex_addr__nlast[0];
astex_thread_begin:  {
    nrand = (((long ) seed) << (8 * sizeof (long ) - 32)) >> (8 * sizeof (long ) - 14);
    noise = nrand + (0.75 * nlast);
    nlast = noise;
    if (nper > nmod)
      {
        noise *= 0.5;
      }
    sourc = frics = amp_frica * noise;
  }
astex_thread_end:;
  __astex_addr__nlast[0] = nlast;
  __astex_addr__noise[0] = noise;
  __astex_addr__sourc[0] = sourc;
  __astex_addr__frics[0] = frics;
  __astex_addr__nrand[0] = nrand;
}

