struct adpcm_state  {
  short  valprev;
  char  index;
} ;

#pragma hmpp astex_codelet__1 codelet &
#pragma hmpp astex_codelet__1 , args[outp].io=inout &
#pragma hmpp astex_codelet__1 , args[inp].io=in &
#pragma hmpp astex_codelet__1 , args[stepsizeTable].io=in &
#pragma hmpp astex_codelet__1 , args[indexTable].io=in &
#pragma hmpp astex_codelet__1 , args[state].io=inout &
#pragma hmpp astex_codelet__1 , target=C &
#pragma hmpp astex_codelet__1 , version=1.4.0

void astex_codelet__1(int len, struct adpcm_state *state, int indexTable[16], int stepsizeTable[89], short *inp, signed char *outp, int step, int valpred, int index, int outputbuffer, int bufferstep)
{
  int  vpdiff;
  int  diff;
  int  delta;
  int  sign;
  int  val;
astex_thread_begin:  {
    for ( ; len > 0 ; len--)
      {
        val = *inp++;
        diff = val - valpred;
        sign = (diff < 0)?8:0;
        if (sign)
          diff = ( -diff);
        delta = 0;
        vpdiff = (step >> 3);
        if (diff >= step)
          {
            delta = 4;
            diff -= step;
            vpdiff += step;
          }
        step >>= 1;
        if (diff >= step)
          {
            delta |= 2;
            diff -= step;
            vpdiff += step;
          }
        step >>= 1;
        if (diff >= step)
          {
            delta |= 1;
            vpdiff += step;
          }
        if (sign)
          valpred -= vpdiff;
        else         valpred += vpdiff;
        if (valpred > 32767)
          valpred = 32767;
        else         if (valpred <  -32768)
          valpred =  -32768;
        delta |= sign;
        index += indexTable[delta];
        if (index < 0)
          index = 0;
        if (index > 88)
          index = 88;
        step = stepsizeTable[index];
        if (bufferstep)
          {
            outputbuffer = (delta << 4) & 0xf0;
          }
        else         {
          *outp++ = (delta & 0x0f) | outputbuffer;
        }
        bufferstep =  !bufferstep;
      }
    if ( !bufferstep)
      *outp++ = outputbuffer;
    state->valprev = valpred;
    state->index = index;
  }
astex_thread_end:;
}

