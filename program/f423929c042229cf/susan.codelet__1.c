typedef int  TOTAL_TYPE;

typedef unsigned char  uchar;

#pragma hmpp astex_codelet__1 codelet &
#pragma hmpp astex_codelet__1 , args[__astex_addr__total].io=inout &
#pragma hmpp astex_codelet__1 , args[cp].io=in &
#pragma hmpp astex_codelet__1 , args[dpt].io=in &
#pragma hmpp astex_codelet__1 , args[ip].io=in &
#pragma hmpp astex_codelet__1 , args[__astex_addr__tmp].io=out &
#pragma hmpp astex_codelet__1 , target=C &
#pragma hmpp astex_codelet__1 , version=1.4.0

void astex_codelet__1(int increment, int mask_size, int area, int __astex_addr__tmp[1], uchar *ip, uchar *dpt, uchar *cp, TOTAL_TYPE __astex_addr__total[1])
{
  TOTAL_TYPE  total = __astex_addr__total[0];
  int  tmp;
  int  brightness;
  int  y;
  int  x;
astex_thread_begin:  {
    for (y =  -mask_size ; y <= mask_size ; y++)
      {
        for (x =  -mask_size ; x <= mask_size ; x++)
          {
            brightness = *ip++;
            tmp = *dpt++ * *(cp - brightness);
            area += tmp;
            total += tmp * brightness;
          }
        ip += increment;
      }
    tmp = area - 10000;
  }
astex_thread_end:;
  __astex_addr__tmp[0] = tmp;
  __astex_addr__total[0] = total;
}

