#pragma hmpp astex_codelet__2 codelet &
#pragma hmpp astex_codelet__2 , args[__astex_addr__astex_do_return].io=out &
#pragma hmpp astex_codelet__2 , args[__astex_addr__astex_what_return].io=out &
#pragma hmpp astex_codelet__2 , target=C &
#pragma hmpp astex_codelet__2 , version=1.4.0

void astex_codelet__2(long x, int n, int __astex_addr__astex_what_return[1], int __astex_addr__astex_do_return[1])
{
  int  astex_do_return;
  astex_do_return = 0;
  int  astex_what_return;
astex_thread_begin:  {
    if (x)
      do
        n++;
      while (0 != (x = x & (x - 1)));
    {
      astex_what_return = (n);
      astex_do_return = 1;
goto astex_thread_end;
    }
  }
astex_thread_end:;
  __astex_addr__astex_what_return[0] = astex_what_return;
  __astex_addr__astex_do_return[0] = astex_do_return;
}

