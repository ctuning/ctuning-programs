typedef unsigned short  MULTUNIT;

#pragma hmpp astex_codelet__1 codelet &
#pragma hmpp astex_codelet__1 , args[multiplicand].io=in &
#pragma hmpp astex_codelet__1 , args[prod].io=inout &
#pragma hmpp astex_codelet__1 , target=C &
#pragma hmpp astex_codelet__1 , version=1.4.0

void astex_codelet__1(MULTUNIT *prod, MULTUNIT *multiplicand, MULTUNIT multiplier, short munit_prec, unsigned long carry)
{
  unsigned long  p;
  short  i;
astex_thread_begin:  {
    for (i = 0 ; i < munit_prec ; ++i)
      {
        p = (unsigned long ) multiplier * *((multiplicand)++);
        p += *prod + carry;
        *((prod)++) = (MULTUNIT ) p;
        carry = p >> 16;
      }
    *prod += (MULTUNIT ) carry;
  }
astex_thread_end:;
}

