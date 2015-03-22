struct _NODE  {
  int  iDist;
  int  iPrev;
} ;

typedef struct _NODE  NODE;

#pragma hmpp astex_codelet__5 codelet &
#pragma hmpp astex_codelet__5 , args[__astex_addr__ch].io=inout &
#pragma hmpp astex_codelet__5 , args[rgnNodes].io=inout &
#pragma hmpp astex_codelet__5 , target=C &
#pragma hmpp astex_codelet__5 , version=1.4.0

void astex_codelet__5(int NUM_NODES, NODE *rgnNodes, int __astex_addr__ch[1])
{
  int  ch = __astex_addr__ch[0];
astex_thread_begin:  {
    for (ch = 0 ; ch < NUM_NODES ; ch++)
      {
        rgnNodes[ch].iDist = 9999;
        rgnNodes[ch].iPrev = 9999;
      }
  }
astex_thread_end:;
  __astex_addr__ch[0] = ch;
}

