typedef double  FLOAT8;

typedef FLOAT8  D576[576];

typedef FLOAT8  D192_3[192][3];

#pragma hmpp astex_codelet__6 codelet &
#pragma hmpp astex_codelet__6 , args[ix].io=inout &
#pragma hmpp astex_codelet__6 , args[xr].io=in &
#pragma hmpp astex_codelet__6 , target=C &
#pragma hmpp astex_codelet__6 , version=1.4.0

void astex_codelet__6(FLOAT8 xr[576], int ix[576], const FLOAT8 compareval0, const FLOAT8 istep)
{
  int  j;
astex_thread_begin:  {
    for (j = 576 ; j > 0 ; j--)
      {
        if (compareval0 > *xr)
          {
            *(ix++) = 0;
            xr++;
          }
        else         ((*(ix++)) = (int ) (istep * (*(xr++)) + 0.4054));
      }
  }
astex_thread_end:;
}

