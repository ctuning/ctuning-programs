#pragma hmpp astex_codelet__3 codelet &
#pragma hmpp astex_codelet__3 , args[x].io=inout &
#pragma hmpp astex_codelet__3 , args[solutions].io=inout &
#pragma hmpp astex_codelet__3 , target=C &
#pragma hmpp astex_codelet__3 , version=1.4.0

void astex_codelet__3(int *solutions, double *x, long double a1, long double Q, long double R, double R2_Q3)
{
  double  theta;
astex_thread_begin:  {
    if (R2_Q3 <= 0)
      {
        *solutions = 3;
        theta = acos(R / sqrt(Q * Q * Q));
        x[0] =  -2.0 * sqrt(Q) * cos(theta / 3.0) - a1 / 3.0;
        x[1] =  -2.0 * sqrt(Q) * cos((theta + 2.0 * (4 * atan(1))) / 3.0) - a1 / 3.0;
        x[2] =  -2.0 * sqrt(Q) * cos((theta + 4.0 * (4 * atan(1))) / 3.0) - a1 / 3.0;
      }
    else     {
      *solutions = 1;
      x[0] = pow(sqrt(R2_Q3) + fabs(R), 1 / 3.0);
      x[0] += Q / x[0];
      x[0] *= (R < 0.0)?1: -1;
      x[0] -= a1 / 3.0;
    }
  }
astex_thread_end:;
}

