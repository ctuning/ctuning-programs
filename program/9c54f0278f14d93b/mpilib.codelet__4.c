typedef unsigned short  word16;

typedef word16  unit;

typedef unit  *unitptr;

#pragma hmpp astex_codelet__4 codelet &
#pragma hmpp astex_codelet__4 , args[__astex_addr__astex_do_return].io=out &
#pragma hmpp astex_codelet__4 , args[__astex_addr__astex_what_return].io=out &
#pragma hmpp astex_codelet__4 , args[r2].io=in &
#pragma hmpp astex_codelet__4 , args[r1].io=in &
#pragma hmpp astex_codelet__4 , target=C &
#pragma hmpp astex_codelet__4 , version=1.4.0

void astex_codelet__4(unitptr r1, unitptr r2, short precision, short __astex_addr__astex_what_return[1], int __astex_addr__astex_do_return[1])
{
  int  astex_do_return;
  astex_do_return = 0;
  short  astex_what_return;
astex_thread_begin:  {
    do
      {
        if (*r1 < *r2)
          {
            astex_what_return =  -1;
            astex_do_return = 1;
goto astex_thread_end;
          }
        if (*((r1)--) > *((r2)--))
          {
            astex_what_return = 1;
            astex_do_return = 1;
goto astex_thread_end;
          }
      }
    while (--precision);
    {
      astex_what_return = 0;
      astex_do_return = 1;
goto astex_thread_end;
    }
  }
astex_thread_end:;
  __astex_addr__astex_what_return[0] = astex_what_return;
  __astex_addr__astex_do_return[0] = astex_do_return;
}

