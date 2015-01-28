typedef double  FLOAT8;

typedef FLOAT8  D576[576];

typedef FLOAT8  D192_3[192][3];

#pragma hmpp astex_codelet__10 codelet &
#pragma hmpp astex_codelet__10 , args[cos_l0].io=in &
#pragma hmpp astex_codelet__10 , args[all].io=in &
#pragma hmpp astex_codelet__10 , args[in].io=in &
#pragma hmpp astex_codelet__10 , args[out].io=inout &
#pragma hmpp astex_codelet__10 , target=C &
#pragma hmpp astex_codelet__10 , version=1.4.0

void astex_codelet__10(FLOAT8 *out, FLOAT8 *in, const int all[], int j, FLOAT8 *cos_l0)
{
  FLOAT8  s5;
  FLOAT8  s4;
  FLOAT8  s3;
  FLOAT8  s2;
  FLOAT8  s1;
  FLOAT8  s0;
astex_thread_begin:  {
    do
      {
        out[all[j]] = in[0] * cos_l0[0] + in[1] * cos_l0[1] + in[2] * cos_l0[2] + in[3] * cos_l0[3] + in[4] * cos_l0[4] + in[5] * cos_l0[5] + in[6] * cos_l0[6] + in[7] * cos_l0[7] + in[8] * cos_l0[8] + in[9] * cos_l0[9] + in[10] * cos_l0[10] + in[11] * cos_l0[11] + in[12] * cos_l0[12] + in[13] * cos_l0[13] + in[14] * cos_l0[14] + in[15] * cos_l0[15] + in[16] * cos_l0[16] + in[17] * cos_l0[17];
        cos_l0 += 18;
      }
    while (--j >= 0);
    s0 = in[0] + in[5] + in[15];
    s1 = in[1] + in[4] + in[16];
    s2 = in[2] + in[3] + in[17];
    s3 = in[6] - in[9] + in[14];
    s4 = in[7] - in[10] + in[13];
    s5 = in[8] - in[11] + in[12];
    out[16] = s0 * cos_l0[0] + s1 * cos_l0[1] + s2 * cos_l0[2] + s3 * cos_l0[3] + s4 * cos_l0[4] + s5 * cos_l0[5];
    cos_l0 += 6;
    out[10] = s0 * cos_l0[0] + s1 * cos_l0[1] + s2 * cos_l0[2] + s3 * cos_l0[3] + s4 * cos_l0[4] + s5 * cos_l0[5];
    cos_l0 += 6;
    out[7] = s0 * cos_l0[0] + s1 * cos_l0[1] + s2 * cos_l0[2] + s3 * cos_l0[3] + s4 * cos_l0[4] + s5 * cos_l0[5];
    cos_l0 += 6;
    out[1] = s0 * cos_l0[0] + s1 * cos_l0[1] + s2 * cos_l0[2] + s3 * cos_l0[3] + s4 * cos_l0[4] + s5 * cos_l0[5];
    cos_l0 += 6;
    s0 = s0 - s1 + s5;
    s2 = s2 - s3 - s4;
    out[13] = s0 * cos_l0[0] + s2 * cos_l0[1];
    out[4] = s0 * cos_l0[2] + s2 * cos_l0[3];
  }
astex_thread_end:;
}

