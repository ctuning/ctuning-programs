#pragma hmpp astex_codelet__9 codelet &
#pragma hmpp astex_codelet__9 , args[__astex_addr__v].io=inout &
#pragma hmpp astex_codelet__9 , args[__astex_addr__c2].io=inout &
#pragma hmpp astex_codelet__9 , args[__astex_addr__c1].io=inout &
#pragma hmpp astex_codelet__9 , args[others].io=inout &
#pragma hmpp astex_codelet__9 , args[codesize].io=inout &
#pragma hmpp astex_codelet__9 , args[freq].io=inout &
#pragma hmpp astex_codelet__9 , target=C &
#pragma hmpp astex_codelet__9 , version=1.4.0

void astex_codelet__9(long freq[], int codesize[257], int others[257], int __astex_addr__c1[1], int __astex_addr__c2[1], long __astex_addr__v[1])
{
  long  v = __astex_addr__v[0];
  int  i;
  int  c2 = __astex_addr__c2[0];
  int  c1 = __astex_addr__c1[0];
astex_thread_begin:  {
    for ( ;  ; )
      {
        c1 =  -1;
        v = 1000000000L;
        for (i = 0 ; i <= 256 ; i++)
          {
            if (freq[i] && freq[i] <= v)
              {
                v = freq[i];
                c1 = i;
              }
          }
        c2 =  -1;
        v = 1000000000L;
        for (i = 0 ; i <= 256 ; i++)
          {
            if (freq[i] && freq[i] <= v && i != c1)
              {
                v = freq[i];
                c2 = i;
              }
          }
        if (c2 < 0)
          break;
        freq[c1] += freq[c2];
        freq[c2] = 0;
        codesize[c1]++;
        while (others[c1] >= 0)
          {
            c1 = others[c1];
            codesize[c1]++;
          }
        others[c1] = c2;
        codesize[c2]++;
        while (others[c2] >= 0)
          {
            c2 = others[c2];
            codesize[c2]++;
          }
      }
  }
astex_thread_end:;
  __astex_addr__c1[0] = c1;
  __astex_addr__c2[0] = c2;
  __astex_addr__v[0] = v;
}

