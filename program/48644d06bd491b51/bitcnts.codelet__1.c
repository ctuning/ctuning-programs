#pragma hmpp astex_codelet__1 codelet &
#pragma hmpp astex_codelet__1 , args[__astex_addr__astex_do_return].io=out &
#pragma hmpp astex_codelet__1 , args[__astex_addr__astex_what_return].io=out &
#pragma hmpp astex_codelet__1 , target=C &
#pragma hmpp astex_codelet__1 , version=1.4.0

void astex_codelet__1(long int x, int __astex_addr__astex_what_return[1], int __astex_addr__astex_do_return[1])
{
  int  astex_do_return;
  astex_do_return = 0;
  int  astex_what_return;
  int  n;
  int  i;
astex_thread_begin:  {
    for (i = n = 0 ; x && (i < (sizeof (long ) * 8)) ; ++i, x >>= 1)
      n += (int ) (x & 1L);
    {
      astex_what_return = n;
      astex_do_return = 1;
goto astex_thread_end;
    }
  }
astex_thread_end:;
  __astex_addr__astex_what_return[0] = astex_what_return;
  __astex_addr__astex_do_return[0] = astex_do_return;
}

