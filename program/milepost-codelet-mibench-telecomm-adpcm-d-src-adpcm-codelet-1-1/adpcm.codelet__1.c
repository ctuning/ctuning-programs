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

void astex_codelet__1(int len, struct adpcm_state *state, int indexTable[16], int stepsizeTable[89], signed char *inp, short *outp, int step, int valpred, int index, int inputbuffer, int bufferstep)
{
  int  vpdiff;
  int  delta;
  int  sign;
astex_thread_begin:  {
    for ( ; len > 0 ; len--)
      {
        if (bufferstep)
          {
            delta = inputbuffer & 0xf;
          }
        else         {
          inputbuffer = *inp++;
          delta = (inputbuffer >> 4) & 0xf;
        }
        bufferstep =  !bufferstep;
        index += indexTable[delta];
        if (index < 0)
          index = 0;
        if (index > 88)
          index = 88;
        sign = delta & 8;
        delta = delta & 7;
        vpdiff = step >> 3;
        if (delta & 4)
          vpdiff += step;
        if (delta & 2)
          vpdiff += step >> 1;
        if (delta & 1)
          vpdiff += step >> 2;
        if (sign)
          valpred -= vpdiff;
        else         valpred += vpdiff;
        if (valpred > 32767)
          valpred = 32767;
        else         if (valpred <  -32768)
          valpred =  -32768;
        step = stepsizeTable[index];
        *outp++ = valpred;
      }
    state->valprev = valpred;
    state->index = index;
  }
astex_thread_end:;
}

