#pragma hmpp astex_codelet__3 codelet &
#pragma hmpp astex_codelet__3 , args[ImagOut].io=inout &
#pragma hmpp astex_codelet__3 , args[RealOut].io=inout &
#pragma hmpp astex_codelet__3 , target=C &
#pragma hmpp astex_codelet__3 , version=1.4.0

void astex_codelet__3(unsigned NumSamples, int InverseTransform, float *RealOut, float *ImagOut, unsigned BlockEnd, double angle_numerator)
{
  double  ti;
  double  tr;
  unsigned  BlockSize;
  unsigned  n;
  unsigned  k;
  unsigned  j;
  unsigned  i;
astex_thread_begin:  {
    for (BlockSize = 2 ; BlockSize <= NumSamples ; BlockSize <<= 1)
      {
        double  delta_angle = angle_numerator / (double ) BlockSize;
        double  sm2 = sin( -2 * delta_angle);
        double  sm1 = sin( -delta_angle);
        double  cm2 = cos( -2 * delta_angle);
        double  cm1 = cos( -delta_angle);
        double  w = 2 * cm1;
        double  ar[3], ai[3];
        double  temp;
        for (i = 0 ; i < NumSamples ; i += BlockSize)
          {
            ar[2] = cm2;
            ar[1] = cm1;
            ai[2] = sm2;
            ai[1] = sm1;
            for (j = i, n = 0 ; n < BlockEnd ; j++, n++)
              {
                ar[0] = w * ar[1] - ar[2];
                ar[2] = ar[1];
                ar[1] = ar[0];
                ai[0] = w * ai[1] - ai[2];
                ai[2] = ai[1];
                ai[1] = ai[0];
                k = j + BlockEnd;
                tr = ar[0] * RealOut[k] - ai[0] * ImagOut[k];
                ti = ar[0] * ImagOut[k] + ai[0] * RealOut[k];
                RealOut[k] = RealOut[j] - tr;
                ImagOut[k] = ImagOut[j] - ti;
                RealOut[j] += tr;
                ImagOut[j] += ti;
              }
          }
        BlockEnd = BlockSize;
      }
    if (InverseTransform)
      {
        double  denom = (double ) NumSamples;
        for (i = 0 ; i < NumSamples ; i++)
          {
            RealOut[i] /= denom;
            ImagOut[i] /= denom;
          }
      }
  }
astex_thread_end:;
}

