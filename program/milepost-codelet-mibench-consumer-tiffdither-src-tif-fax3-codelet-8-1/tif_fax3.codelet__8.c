typedef unsigned char  __u_char;

typedef unsigned long int  __u_long;

typedef __u_char  u_char;

typedef __u_long  u_long;

typedef int  int32;

typedef int32  tsize_t;

typedef int32  toff_t;

#pragma hmpp astex_codelet__8 codelet &
#pragma hmpp astex_codelet__8 , args[__astex_addr__astex_do_return].io=out &
#pragma hmpp astex_codelet__8 , args[__astex_addr__astex_what_return].io=out &
#pragma hmpp astex_codelet__8 , args[zeroruns].io=in &
#pragma hmpp astex_codelet__8 , args[bp].io=in &
#pragma hmpp astex_codelet__8 , target=C &
#pragma hmpp astex_codelet__8 , version=1.4.0

void astex_codelet__8(u_char *bp, int32 bs, const u_char zeroruns[256], int32 bits, int32 __astex_addr__astex_what_return[1], int __astex_addr__astex_do_return[1])
{
  int  astex_do_return;
  astex_do_return = 0;
  int32  astex_what_return;
  int32  span;
  int32  n;
astex_thread_begin:  {
    bp += bs >> 3;
    if (bits > 0 && (n = (bs & 7)))
      {
        span = zeroruns[(*bp << n) & 0xff];
        if (span > 8 - n)
          span = 8 - n;
        if (span > bits)
          span = bits;
        if (n + span < 8)
          {
            astex_what_return = (span);
            astex_do_return = 1;
goto astex_thread_end;
          }
        bits -= span;
        bp++;
      }
    else     span = 0;
    if (bits >= 2 * 8 * sizeof (long ))
      {
        long  *lp;
        while ( !((((u_long ) (bp)) & (sizeof (long ) - 1)) == 0))
          {
            if (*bp != 0x00)
              {
                astex_what_return = (span + zeroruns[*bp]);
                astex_do_return = 1;
goto astex_thread_end;
              }
            span += 8, bits -= 8;
            bp++;
          }
        lp = (long *) bp;
        while (bits >= 8 * sizeof (long ) && *lp == 0)
          {
            span += 8 * sizeof (long ), bits -= 8 * sizeof (long );
            lp++;
          }
        bp = (u_char *) lp;
      }
    while (bits >= 8)
      {
        if (*bp != 0x00)
          {
            astex_what_return = (span + zeroruns[*bp]);
            astex_do_return = 1;
goto astex_thread_end;
          }
        span += 8, bits -= 8;
        bp++;
      }
    if (bits > 0)
      {
        n = zeroruns[*bp];
        span += (n > bits?bits:n);
      }
    {
      astex_what_return = (span);
      astex_do_return = 1;
goto astex_thread_end;
    }
  }
astex_thread_end:;
  __astex_addr__astex_what_return[0] = astex_what_return;
  __astex_addr__astex_do_return[0] = astex_do_return;
}

