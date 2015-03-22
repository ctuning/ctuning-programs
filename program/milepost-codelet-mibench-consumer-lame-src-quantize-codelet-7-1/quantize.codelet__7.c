typedef double  FLOAT8;

typedef FLOAT8  D576[576];

typedef FLOAT8  D192_3[192][3];

#pragma hmpp astex_codelet__7 codelet &
#pragma hmpp astex_codelet__7 , args[__astex_addr__temp].io=inout &
#pragma hmpp astex_codelet__7 , args[xrpow].io=inout &
#pragma hmpp astex_codelet__7 , args[__astex_addr__i].io=out &
#pragma hmpp astex_codelet__7 , args[xr].io=in &
#pragma hmpp astex_codelet__7 , target=C &
#pragma hmpp astex_codelet__7 , version=1.4.0

void astex_codelet__7(FLOAT8 xr[576], int __astex_addr__i[1], FLOAT8 xrpow[576], FLOAT8 __astex_addr__temp[1])
{
  FLOAT8  temp = __astex_addr__temp[0];
  int  i;
astex_thread_begin:  {
    for (i = 0 ; i < 576 ; i++)
      {
        temp = fabs(xr[i]);
        xrpow[i] = sqrt(sqrt(temp) * temp);
      }
  }
astex_thread_end:;
  __astex_addr__i[0] = i;
  __astex_addr__temp[0] = temp;
}

