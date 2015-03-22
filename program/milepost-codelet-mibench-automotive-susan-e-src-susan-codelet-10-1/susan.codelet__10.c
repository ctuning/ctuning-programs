typedef unsigned char  uchar;

#pragma hmpp astex_codelet__10 codelet &
#pragma hmpp astex_codelet__10 , args[mid].io=inout &
#pragma hmpp astex_codelet__10 , args[in].io=inout &
#pragma hmpp astex_codelet__10 , target=C &
#pragma hmpp astex_codelet__10 , version=1.4.0

void astex_codelet__10(uchar *in, uchar *mid, int x_size, int y_size, int drawing_mode)
{
  uchar  *midp;
  uchar  *inp;
  int  i;
astex_thread_begin:  {
    if (drawing_mode == 0)
      {
        midp = mid;
        for (i = 0 ; i < x_size * y_size ; i++)
          {
            if (*midp < 8)
              {
                inp = in + (midp - mid) - x_size - 1;
                *inp++ = 255;
                *inp++ = 255;
                *inp = 255;
                inp += x_size - 2;
                *inp++ = 255;
                *inp++;
                *inp = 255;
                inp += x_size - 2;
                *inp++ = 255;
                *inp++ = 255;
                *inp = 255;
              }
            midp++;
          }
      }
    midp = mid;
    for (i = 0 ; i < x_size * y_size ; i++)
      {
        if (*midp < 8)
          *(in + (midp - mid)) = 0;
        midp++;
      }
  }
astex_thread_end:;
}

