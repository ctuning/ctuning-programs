#pragma hmpp astex_codelet__9 codelet &
#pragma hmpp astex_codelet__9 , args[__astex_addr__aspiration].io=out &
#pragma hmpp astex_codelet__9 , args[__astex_addr__par_glotout].io=inout &
#pragma hmpp astex_codelet__9 , args[__astex_addr__glotout].io=out &
#pragma hmpp astex_codelet__9 , target=C &
#pragma hmpp astex_codelet__9 , version=1.4.0

void astex_codelet__9(float amp_voice, float amp_aspir, float noise, float __astex_addr__glotout[1], float __astex_addr__par_glotout[1], float voice, float __astex_addr__aspiration[1])
{
  float  aspiration;
  float  par_glotout = __astex_addr__par_glotout[0];
  float  glotout;
astex_thread_begin:  {
    glotout = amp_voice * voice;
    aspiration = amp_aspir * noise;
    glotout += aspiration;
    par_glotout = glotout;
  }
astex_thread_end:;
  __astex_addr__glotout[0] = glotout;
  __astex_addr__par_glotout[0] = par_glotout;
  __astex_addr__aspiration[0] = aspiration;
}

