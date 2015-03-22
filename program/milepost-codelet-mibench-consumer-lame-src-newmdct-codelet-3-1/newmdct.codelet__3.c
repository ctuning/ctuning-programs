typedef double  FLOAT8;

typedef FLOAT8  D576[576];

typedef FLOAT8  D192_3[192][3];

#pragma hmpp astex_codelet__3 codelet &
#pragma hmpp astex_codelet__3 , args[wp].io=in &
#pragma hmpp astex_codelet__3 , args[in].io=in &
#pragma hmpp astex_codelet__3 , args[d].io=inout &
#pragma hmpp astex_codelet__3 , target=C &
#pragma hmpp astex_codelet__3 , version=1.4.0

void astex_codelet__3(FLOAT8 d[32], FLOAT8 *in, FLOAT8 s, FLOAT8 t, FLOAT8 *wp)
{
  int  i;
astex_thread_begin:  {
    for (i = 15 ; i >= 0 ; --i)
      {
        int  j;
        FLOAT8  s0 = s;
        FLOAT8  s1 = t * *wp++;
        for (j = 14 ; j >= 0 ; j--)
          {
            s0 += *wp++ * *in++;
            s1 += *wp++ * *in++;
          }
        in -= 30;
        d[i] = s0 + s1;
        d[31 - i] = s0 - s1;
      }
  }
astex_thread_end:;
}

