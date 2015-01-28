#pragma hmpp astex_codelet__1 codelet &
#pragma hmpp astex_codelet__1 , args[b].io=inout &
#pragma hmpp astex_codelet__1 , args[a].io=inout &
#pragma hmpp astex_codelet__1 , target=C &
#pragma hmpp astex_codelet__1 , version=1.4.0

void astex_codelet__1(char *a, char *b, unsigned width)
{
  char  tmp;
astex_thread_begin:  {
    if (a != b)
      {
        while (width--)
          {
            tmp = *a;
            *a++ = *b;
            *b++ = tmp;
          }
      }
  }
astex_thread_end:;
}

