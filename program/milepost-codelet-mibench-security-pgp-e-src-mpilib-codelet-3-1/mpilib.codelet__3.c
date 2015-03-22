typedef unsigned char  boolean;

typedef unsigned short  word16;

typedef word16  unit;

typedef short  signedunit;

typedef unit  *unitptr;

#pragma hmpp astex_codelet__3 codelet &
#pragma hmpp astex_codelet__3 , args[__astex_addr__astex_do_return].io=out &
#pragma hmpp astex_codelet__3 , args[__astex_addr__astex_what_return].io=out &
#pragma hmpp astex_codelet__3 , args[r1].io=inout &
#pragma hmpp astex_codelet__3 , target=C &
#pragma hmpp astex_codelet__3 , version=1.4.0

void astex_codelet__3(unitptr r1, int precision, unsigned int mcarry, unsigned int nextcarry, boolean __astex_addr__astex_what_return[1], int __astex_addr__astex_do_return[1])
{
  int  astex_do_return;
  astex_do_return = 0;
  boolean  astex_what_return;
astex_thread_begin:  {
    while (precision--)
      {
        nextcarry = (((signedunit ) *r1) < 0);
        *r1 = (*r1 << 1) | mcarry;
        mcarry = nextcarry;
        (++(r1));
      }
    {
      astex_what_return = nextcarry;
      astex_do_return = 1;
goto astex_thread_end;
    }
  }
astex_thread_end:;
  __astex_addr__astex_what_return[0] = astex_what_return;
  __astex_addr__astex_do_return[0] = astex_do_return;
}

