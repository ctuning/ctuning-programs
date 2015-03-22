typedef short  JCOEF;

typedef JCOEF  JBLOCK[64];

typedef JCOEF  *JCOEFPTR;

typedef int  DCTELEM;

#pragma hmpp astex_codelet__13 codelet &
#pragma hmpp astex_codelet__13 , args[workspace].io=in &
#pragma hmpp astex_codelet__13 , args[divisors].io=in &
#pragma hmpp astex_codelet__13 , args[output_ptr].io=inout &
#pragma hmpp astex_codelet__13 , target=C &
#pragma hmpp astex_codelet__13 , version=1.4.0

void astex_codelet__13(JCOEFPTR output_ptr, DCTELEM *divisors, DCTELEM workspace[64])
{
  int  i;
  DCTELEM  qval;
  DCTELEM  temp;
astex_thread_begin:  {
    for (i = 0 ; i < 64 ; i++)
      {
        qval = divisors[i];
        temp = workspace[i];
        if (temp < 0)
          {
            temp =  -temp;
            temp += qval >> 1;
            if (temp >= qval)
              temp /= qval;
            else             temp = 0;
            temp =  -temp;
          }
        else         {
          temp += qval >> 1;
          if (temp >= qval)
            temp /= qval;
          else           temp = 0;
        }
        output_ptr[i] = (JCOEF ) temp;
      }
  }
astex_thread_end:;
}

