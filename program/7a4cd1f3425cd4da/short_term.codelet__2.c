typedef short  word;

typedef long  longword;

typedef unsigned long  ulongword;

#pragma hmpp astex_codelet__2 codelet &
#pragma hmpp astex_codelet__2 , args[u].io=inout &
#pragma hmpp astex_codelet__2 , args[s].io=inout &
#pragma hmpp astex_codelet__2 , args[rp].io=in &
#pragma hmpp astex_codelet__2 , target=C &
#pragma hmpp astex_codelet__2 , version=1.4.0

void astex_codelet__2(word *rp, int k_n, word *s, word *u, longword ltmp)
{
  word  rpi;
  word  sav;
  word  ui;
  word  zzz;
  word  di;
  int  i;
astex_thread_begin:  {
    for ( ; k_n-- ; s++)
      {
        di = sav = *s;
        for (i = 0 ; i < 8 ; i++)
          {
            ui = u[i];
            rpi = rp[i];
            u[i] = sav;
            zzz = (((((longword ) (rpi) * (longword ) (di) + 16384)) >> (15)));
            sav = ((ulongword ) ((ltmp = (longword ) (ui) + (longword ) (zzz)) - ((-32767) - 1)) > (32767) - ((-32767) - 1)?(ltmp > 0?(32767):((-32767) - 1)):ltmp);
            zzz = (((((longword ) (rpi) * (longword ) (ui) + 16384)) >> (15)));
            di = ((ulongword ) ((ltmp = (longword ) (di) + (longword ) (zzz)) - ((-32767) - 1)) > (32767) - ((-32767) - 1)?(ltmp > 0?(32767):((-32767) - 1)):ltmp);
          }
        *s = di;
      }
  }
astex_thread_end:;
}

