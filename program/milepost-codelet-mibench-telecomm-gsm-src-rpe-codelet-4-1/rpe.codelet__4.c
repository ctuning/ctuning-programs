typedef short  word;

typedef long  longword;

#pragma hmpp astex_codelet__4 codelet &
#pragma hmpp astex_codelet__4 , args[x].io=inout &
#pragma hmpp astex_codelet__4 , args[e].io=in &
#pragma hmpp astex_codelet__4 , target=C &
#pragma hmpp astex_codelet__4 , version=1.4.0

void astex_codelet__4(word *e, word *x)
{
  int  k;
  longword  L_result;
astex_thread_begin:  {
    for (k = 0 ; k <= 39 ; k++)
      {
        L_result = 8192 >> 1;
        L_result += (e[k + 0] * (longword ) -134);
        L_result += (e[k + 1] * (longword ) -374);
        L_result += (e[k + 3] * (longword ) 2054);
        L_result += (e[k + 4] * (longword ) 5741);
        L_result += (e[k + 5] * (longword ) 8192);
        L_result += (e[k + 6] * (longword ) 5741);
        L_result += (e[k + 7] * (longword ) 2054);
        L_result += (e[k + 9] * (longword ) -374);
        L_result += (e[k + 10] * (longword ) -134);
        L_result = ((L_result) >> (13));
        x[k] = (L_result < ((-32767) - 1)?((-32767) - 1):(L_result > (32767)?(32767):L_result));
      }
  }
astex_thread_end:;
}

