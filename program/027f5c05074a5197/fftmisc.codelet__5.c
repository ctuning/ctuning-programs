#pragma hmpp astex_codelet__5 codelet &
#pragma hmpp astex_codelet__5 , args[__astex_addr__astex_do_return].io=out &
#pragma hmpp astex_codelet__5 , args[__astex_addr__astex_what_return].io=out &
#pragma hmpp astex_codelet__5 , target=C &
#pragma hmpp astex_codelet__5 , version=1.4.0

void astex_codelet__5(unsigned index, unsigned NumBits, unsigned __astex_addr__astex_what_return[1], int __astex_addr__astex_do_return[1])
{
  int  astex_do_return;
  astex_do_return = 0;
  unsigned  astex_what_return;
  unsigned  rev;
  unsigned  i;
astex_thread_begin:  {
    for (i = rev = 0 ; i < NumBits ; i++)
      {
        rev = (rev << 1) | (index & 1);
        index >>= 1;
      }
    {
      astex_what_return = rev;
      astex_do_return = 1;
goto astex_thread_end;
    }
  }
astex_thread_end:;
  __astex_addr__astex_what_return[0] = astex_what_return;
  __astex_addr__astex_do_return[0] = astex_do_return;
}

