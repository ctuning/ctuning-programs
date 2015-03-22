typedef double  FLOAT8;

typedef FLOAT8  D576[576];

typedef FLOAT8  D192_3[192][3];

#pragma hmpp astex_codelet__5 codelet &
#pragma hmpp astex_codelet__5 , args[__astex_addr__astex_do_return].io=out &
#pragma hmpp astex_codelet__5 , args[__astex_addr__astex_what_return].io=out &
#pragma hmpp astex_codelet__5 , args[__astex_addr__i].io=out &
#pragma hmpp astex_codelet__5 , args[xr].io=in &
#pragma hmpp astex_codelet__5 , target=C &
#pragma hmpp astex_codelet__5 , version=1.4.0

void astex_codelet__5(FLOAT8 *xr, int __astex_addr__i[1], FLOAT8 w, int __astex_addr__astex_what_return[1], int __astex_addr__astex_do_return[1])
{
  int  astex_do_return;
  astex_do_return = 0;
  int  astex_what_return;
  int  i;
astex_thread_begin:  {
    for (i = 0 ; i < 576 ; i++)
      {
        if (xr[i] > w)
          {
            astex_what_return = 100000;
            astex_do_return = 1;
goto astex_thread_end;
          }
      }
  }
astex_thread_end:;
  __astex_addr__i[0] = i;
  __astex_addr__astex_what_return[0] = astex_what_return;
  __astex_addr__astex_do_return[0] = astex_do_return;
}

