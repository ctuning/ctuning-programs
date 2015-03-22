typedef unsigned int  __u_int;

typedef __u_int  u_int;

typedef int  int32;

typedef int32  tsize_t;

typedef void  *tdata_t;

typedef int32  toff_t;

typedef void  *thandle_t;

typedef tsize_t  (*TIFFReadWriteProc)(thandle_t , tdata_t , tsize_t );

#pragma hmpp astex_codelet__4 codelet &
#pragma hmpp astex_codelet__4 , args[cp].io=inout &
#pragma hmpp astex_codelet__4 , target=C &
#pragma hmpp astex_codelet__4 , version=1.4.0

void astex_codelet__4(tsize_t cc, tsize_t stride, char *cp)
{
astex_thread_begin:  {
    if (cc > stride)
      {
        cc -= stride;
        if (stride == 3)
          {
            u_int  cr = cp[0];
            u_int  cg = cp[1];
            u_int  cb = cp[2];
            do
              {
                cc -= 3, cp += 3;
                cp[0] = (cr += cp[0]);
                cp[1] = (cg += cp[1]);
                cp[2] = (cb += cp[2]);
              }
            while ((int32 ) cc > 0);
          }
        else         if (stride == 4)
          {
            u_int  cr = cp[0];
            u_int  cg = cp[1];
            u_int  cb = cp[2];
            u_int  ca = cp[3];
            do
              {
                cc -= 4, cp += 4;
                cp[0] = (cr += cp[0]);
                cp[1] = (cg += cp[1]);
                cp[2] = (cb += cp[2]);
                cp[3] = (ca += cp[3]);
              }
            while ((int32 ) cc > 0);
          }
        else         {
          do
            {
              switch (stride)                
                  {
                    default:
                    {
                      int  i;
                      for (i = stride - 4 ; i > 0 ; i--)
                        {
                          cp[stride] += *cp;
                          cp++;
                        }
                    }
                    case 4:
                    cp[stride] += *cp;
                    cp++;
                    case 3:
                    cp[stride] += *cp;
                    cp++;
                    case 2:
                    cp[stride] += *cp;
                    cp++;
                    case 1:
                    cp[stride] += *cp;
                    cp++;
                    case 0:
                    ;
                  }
              cc -= stride;
            }
          while ((int32 ) cc > 0);
        }
      }
  }
astex_thread_end:;
}

