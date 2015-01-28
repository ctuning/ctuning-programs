typedef unsigned int  uint;

typedef uint  bits32;

typedef uint  gs_logical_operation_t;

#pragma hmpp astex_codelet__1 codelet &
#pragma hmpp astex_codelet__1 , args[__astex_addr__x].io=inout &
#pragma hmpp astex_codelet__1 , args[__astex_addr__b].io=inout &
#pragma hmpp astex_codelet__1 , args[__astex_addr__g].io=inout &
#pragma hmpp astex_codelet__1 , args[__astex_addr__r].io=inout &
#pragma hmpp astex_codelet__1 , target=C &
#pragma hmpp astex_codelet__1 , version=1.4.0

void astex_codelet__1(bits32 pixel, uint __astex_addr__r[1], uint __astex_addr__g[1], uint __astex_addr__b[1], uint bpe, uint mask, uint __astex_addr__x[1])
{
  uint  x = __astex_addr__x[0];
  uint  b = __astex_addr__b[0];
  uint  g = __astex_addr__g[0];
  uint  r = __astex_addr__r[0];
astex_thread_begin:  {
    ++x;
    b = pixel & mask;
    pixel >>= bpe;
    g = pixel & mask;
    pixel >>= bpe;
    r = pixel & mask;
  }
astex_thread_end:;
  __astex_addr__r[0] = r;
  __astex_addr__g[0] = g;
  __astex_addr__b[0] = b;
  __astex_addr__x[0] = x;
}

