#pragma hmpp astex_codelet__1 codelet &
#pragma hmpp astex_codelet__1 , args[__astex_addr__i].io=out &
#pragma hmpp astex_codelet__1 , args[__astex_addr__e].io=inout &
#pragma hmpp astex_codelet__1 , args[__astex_addr__r].io=inout &
#pragma hmpp astex_codelet__1 , args[__astex_addr__a].io=inout &
#pragma hmpp astex_codelet__1 , args[__astex_addr__x].io=inout &
#pragma hmpp astex_codelet__1 , target=C &
#pragma hmpp astex_codelet__1 , version=1.4.0

void astex_codelet__1(unsigned long __astex_addr__x[1], unsigned long __astex_addr__a[1], unsigned long __astex_addr__r[1], unsigned long __astex_addr__e[1], int __astex_addr__i[1])
{
  int  i;
  unsigned long  e = __astex_addr__e[0];
  unsigned long  r = __astex_addr__r[0];
  unsigned long  a = __astex_addr__a[0];
  unsigned long  x = __astex_addr__x[0];
astex_thread_begin:  {
    for (i = 0 ; i < 32 ; i++)
      {
        r = (r << 2) + ((x & (3L << (32 - 2))) >> (32 - 2));
        x <<= 2;
        a <<= 1;
        e = (a << 1) + 1;
        if (r >= e)
          {
            r -= e;
            a++;
          }
      }
  }
astex_thread_end:;
  __astex_addr__x[0] = x;
  __astex_addr__a[0] = a;
  __astex_addr__r[0] = r;
  __astex_addr__e[0] = e;
  __astex_addr__i[0] = i;
}

