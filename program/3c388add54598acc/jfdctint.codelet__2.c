typedef long  INT32;

typedef int  DCTELEM;

#pragma hmpp astex_codelet__2 codelet &
#pragma hmpp astex_codelet__2 , args[dataptr].io=inout &
#pragma hmpp astex_codelet__2 , target=C &
#pragma hmpp astex_codelet__2 , version=1.4.0

void astex_codelet__2(DCTELEM *dataptr)
{
  int  ctr;
  INT32  z5;
  INT32  z4;
  INT32  z3;
  INT32  z2;
  INT32  z1;
  INT32  tmp13;
  INT32  tmp12;
  INT32  tmp11;
  INT32  tmp10;
  INT32  tmp7;
  INT32  tmp6;
  INT32  tmp5;
  INT32  tmp4;
  INT32  tmp3;
  INT32  tmp2;
  INT32  tmp1;
  INT32  tmp0;
astex_thread_begin:  {
    for (ctr = 8 - 1 ; ctr >= 0 ; ctr--)
      {
        tmp0 = dataptr[8 * 0] + dataptr[8 * 7];
        tmp7 = dataptr[8 * 0] - dataptr[8 * 7];
        tmp1 = dataptr[8 * 1] + dataptr[8 * 6];
        tmp6 = dataptr[8 * 1] - dataptr[8 * 6];
        tmp2 = dataptr[8 * 2] + dataptr[8 * 5];
        tmp5 = dataptr[8 * 2] - dataptr[8 * 5];
        tmp3 = dataptr[8 * 3] + dataptr[8 * 4];
        tmp4 = dataptr[8 * 3] - dataptr[8 * 4];
        tmp10 = tmp0 + tmp3;
        tmp13 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp1 - tmp2;
        dataptr[8 * 0] = (DCTELEM ) (((tmp10 + tmp11) + (((INT32 ) 1) << ((2) - 1))) >> (2));
        dataptr[8 * 4] = (DCTELEM ) (((tmp10 - tmp11) + (((INT32 ) 1) << ((2) - 1))) >> (2));
        z1 = ((tmp12 + tmp13) * (((INT32 ) 4433)));
        dataptr[8 * 2] = (DCTELEM ) (((z1 + ((tmp13) * (((INT32 ) 6270)))) + (((INT32 ) 1) << ((13 + 2) - 1))) >> (13 + 2));
        dataptr[8 * 6] = (DCTELEM ) (((z1 + ((tmp12) * ( -((INT32 ) 15137)))) + (((INT32 ) 1) << ((13 + 2) - 1))) >> (13 + 2));
        z1 = tmp4 + tmp7;
        z2 = tmp5 + tmp6;
        z3 = tmp4 + tmp6;
        z4 = tmp5 + tmp7;
        z5 = ((z3 + z4) * (((INT32 ) 9633)));
        tmp4 = ((tmp4) * (((INT32 ) 2446)));
        tmp5 = ((tmp5) * (((INT32 ) 16819)));
        tmp6 = ((tmp6) * (((INT32 ) 25172)));
        tmp7 = ((tmp7) * (((INT32 ) 12299)));
        z1 = ((z1) * ( -((INT32 ) 7373)));
        z2 = ((z2) * ( -((INT32 ) 20995)));
        z3 = ((z3) * ( -((INT32 ) 16069)));
        z4 = ((z4) * ( -((INT32 ) 3196)));
        z3 += z5;
        z4 += z5;
        dataptr[8 * 7] = (DCTELEM ) (((tmp4 + z1 + z3) + (((INT32 ) 1) << ((13 + 2) - 1))) >> (13 + 2));
        dataptr[8 * 5] = (DCTELEM ) (((tmp5 + z2 + z4) + (((INT32 ) 1) << ((13 + 2) - 1))) >> (13 + 2));
        dataptr[8 * 3] = (DCTELEM ) (((tmp6 + z2 + z3) + (((INT32 ) 1) << ((13 + 2) - 1))) >> (13 + 2));
        dataptr[8 * 1] = (DCTELEM ) (((tmp7 + z1 + z4) + (((INT32 ) 1) << ((13 + 2) - 1))) >> (13 + 2));
        dataptr++;
      }
  }
astex_thread_end:;
}

