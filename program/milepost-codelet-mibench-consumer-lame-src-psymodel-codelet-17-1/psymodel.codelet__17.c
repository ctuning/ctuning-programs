typedef float  FLOAT;

typedef double  FLOAT8;

typedef enum sound_file_format_e  {
sf_unknown, sf_wave, sf_aiff, sf_mp3, sf_raw}
sound_file_format;

typedef struct   {
  unsigned long  num_samples;
  int  num_channels;
  int  in_samplerate;
  int  out_samplerate;
  int  gtkflag;
  int  bWriteVbrTag;
  int  quality;
  int  silent;
  int  mode;
  int  mode_fixed;
  int  force_ms;
  int  brate;
  int  copyright;
  int  original;
  int  error_protection;
  int  padding_type;
  int  extension;
  int  disable_reservoir;
  int  experimentalX;
  int  experimentalY;
  int  experimentalZ;
  int  VBR;
  int  VBR_q;
  int  VBR_min_bitrate_kbps;
  int  VBR_max_bitrate_kbps;
  int  lowpassfreq;
  int  highpassfreq;
  int  lowpasswidth;
  int  highpasswidth;
  sound_file_format  input_format;
  int  swapbytes;
  char  *inPath;
  char  *outPath;
  int  ATHonly;
  int  noATH;
  float  cwlimit;
  int  allow_diff_short;
  int  no_short_blocks;
  int  emphasis;
  long int  frameNum;
  long  totalframes;
  int  encoder_delay;
  int  framesize;
  int  version;
  int  padding;
  int  mode_gr;
  int  stereo;
  int  VBR_min_bitrate;
  int  VBR_max_bitrate;
  float  resample_ratio;
  int  bitrate_index;
  int  samplerate_index;
  int  mode_ext;
  float  lowpass1, lowpass2;
  float  highpass1, highpass2;
  int  lowpass_band;
  int  highpass_band;
  int  filter_type;
  int  quantization;
  int  noise_shaping;
  int  noise_shaping_stop;
  int  psymodel;
  int  use_best_huffman;
} lame_global_flags;

typedef FLOAT8  D576[576];

typedef FLOAT8  D192_3[192][3];

typedef struct   {
  FLOAT8  l[21 + 1];
  FLOAT8  s[12 + 1][3];
} III_psy_xmin;

#pragma hmpp astex_codelet__17 codelet &
#pragma hmpp astex_codelet__17 , args[__astex_addr__sblock].io=out &
#pragma hmpp astex_codelet__17 , args[__astex_addr__sb].io=out &
#pragma hmpp astex_codelet__17 , args[__astex_addr__k].io=inout &
#pragma hmpp astex_codelet__17 , args[__astex_addr__j].io=inout &
#pragma hmpp astex_codelet__17 , args[__astex_addr__b].io=out &
#pragma hmpp astex_codelet__17 , args[uselongblock].io=inout &
#pragma hmpp astex_codelet__17 , args[pe].io=inout &
#pragma hmpp astex_codelet__17 , args[numlines_l].io=in &
#pragma hmpp astex_codelet__17 , args[numlines_s].io=in &
#pragma hmpp astex_codelet__17 , args[s3ind_s].io=in &
#pragma hmpp astex_codelet__17 , args[s3ind].io=in &
#pragma hmpp astex_codelet__17 , args[bo_s].io=in &
#pragma hmpp astex_codelet__17 , args[bu_s].io=in &
#pragma hmpp astex_codelet__17 , args[bo_l].io=in &
#pragma hmpp astex_codelet__17 , args[bu_l].io=in &
#pragma hmpp astex_codelet__17 , args[w2_s].io=in &
#pragma hmpp astex_codelet__17 , args[w1_s].io=in &
#pragma hmpp astex_codelet__17 , args[w2_l].io=in &
#pragma hmpp astex_codelet__17 , args[w1_l].io=in &
#pragma hmpp astex_codelet__17 , args[thr].io=inout &
#pragma hmpp astex_codelet__17 , args[cb].io=in &
#pragma hmpp astex_codelet__17 , args[eb].io=inout &
#pragma hmpp astex_codelet__17 , args[energy_s].io=in &
#pragma hmpp astex_codelet__17 , args[en].io=inout &
#pragma hmpp astex_codelet__17 , args[thm].io=inout &
#pragma hmpp astex_codelet__17 , args[s3_l].io=in &
#pragma hmpp astex_codelet__17 , args[s3_s].io=in &
#pragma hmpp astex_codelet__17 , args[nb_2].io=inout &
#pragma hmpp astex_codelet__17 , args[nb_1].io=inout &
#pragma hmpp astex_codelet__17 , args[qthr_s].io=in &
#pragma hmpp astex_codelet__17 , args[qthr_l].io=in &
#pragma hmpp astex_codelet__17 , args[minval].io=in &
#pragma hmpp astex_codelet__17 , args[gfp].io=in &
#pragma hmpp astex_codelet__17 , target=C &
#pragma hmpp astex_codelet__17 , version=1.4.0

void astex_codelet__17(lame_global_flags *gfp, FLOAT8 minval[63], FLOAT8 qthr_l[63], FLOAT8 qthr_s[63], FLOAT8 nb_1[4][63], FLOAT8 nb_2[4][63], FLOAT8 s3_s[63 + 1][63 + 1], FLOAT8 s3_l[63 + 1][63 + 1], III_psy_xmin thm[4], III_psy_xmin en[4], FLOAT energy_s[3][129], FLOAT8 eb[63], FLOAT8 cb[63], FLOAT8 thr[63], FLOAT8 w1_l[21], FLOAT8 w2_l[21], FLOAT8 w1_s[12], FLOAT8 w2_s[12], int bu_l[21], int bo_l[21], int bu_s[12], int bo_s[12], int npart_l, int npart_s, int npart_s_orig, int s3ind[63][2], int s3ind_s[63][2], int numlines_s[63], int numlines_l[63], FLOAT8 pe[4], int uselongblock[2], int chn, int __astex_addr__b[1], int __astex_addr__j[1], int __astex_addr__k[1], int __astex_addr__sb[1], int __astex_addr__sblock[1])
{
  int  sblock;
  int  sb;
  int  k = __astex_addr__k[0];
  int  j = __astex_addr__j[0];
  int  b;
astex_thread_begin:  {
    for (b = 0 ; b < npart_l ; b++)
      {
        FLOAT8  tbb, ecb, ctb;
        FLOAT8  temp_1;
        ecb = 0;
        ctb = 0;
        for (k = s3ind[b][0] ; k <= s3ind[b][1] ; k++)
          {
            ecb += s3_l[b][k] * eb[k];
            ctb += s3_l[b][k] * cb[k];
          }
        tbb = ecb;
        if (tbb != 0)
          {
            tbb = ctb / tbb;
            if (tbb <= 0.04875584301)
              {
                tbb = exp(-0.2302585093 * (18 - 6));
              }
            else             if (tbb > 0.4989003827)
              {
                tbb = 1;
              }
            else             {
              tbb = log(tbb);
              tbb = exp(((18 - 6) * (0.2302585093 * 0.299)) + ((18 - 6) * (0.2302585093 * 0.43)) * tbb);
            }
          }
        tbb = ((minval[b]) < (tbb)?(minval[b]):(tbb));
        ecb *= tbb;
        temp_1 = ((ecb) < (((2 * nb_1[chn][b]) < (16 * nb_2[chn][b])?(2 * nb_1[chn][b]):(16 * nb_2[chn][b])))?(ecb):(((2 * nb_1[chn][b]) < (16 * nb_2[chn][b])?(2 * nb_1[chn][b]):(16 * nb_2[chn][b]))));
        thr[b] = ((qthr_l[b]) > (temp_1)?(qthr_l[b]):(temp_1));
        nb_2[chn][b] = nb_1[chn][b];
        nb_1[chn][b] = ecb;
        if (thr[b] < eb[b])
          {
            pe[chn] -= numlines_l[b] * log(thr[b] / eb[b]);
          }
      }
    if (chn < 2)
      {
        if (gfp->no_short_blocks)
          {
            uselongblock[chn] = 1;
          }
        else         {
          if (pe[chn] > 3000)
            {
              uselongblock[chn] = 0;
            }
          else           {
            FLOAT  mn, mx, ma = 0, mb = 0, mc = 0;
            for (j = 129 / 2 ; j < 129 ; j++)
              {
                ma += energy_s[0][j];
                mb += energy_s[1][j];
                mc += energy_s[2][j];
              }
            mn = ((ma) < (mb)?(ma):(mb));
            mn = ((mn) < (mc)?(mn):(mc));
            mx = ((ma) > (mb)?(ma):(mb));
            mx = ((mx) > (mc)?(mx):(mc));
            uselongblock[chn] = 1;
            if (mx > 30 * mn)
              {
                uselongblock[chn] = 0;
              }
            else             if ((mx > 10 * mn) && (pe[chn] > 1000))
              {
                uselongblock[chn] = 0;
              }
          }
        }
      }
    for (sb = 0 ; sb < 21 ; sb++)
      {
        FLOAT8  enn = w1_l[sb] * eb[bu_l[sb]] + w2_l[sb] * eb[bo_l[sb]];
        FLOAT8  thmm = w1_l[sb] * thr[bu_l[sb]] + w2_l[sb] * thr[bo_l[sb]];
        for (b = bu_l[sb] + 1 ; b < bo_l[sb] ; b++)
          {
            enn += eb[b];
            thmm += thr[b];
          }
        en[chn].l[sb] = enn;
        thm[chn].l[sb] = thmm;
      }
    for (sblock = 0 ; sblock < 3 ; sblock++)
      {
        j = 0;
        for (b = 0 ; b < npart_s_orig ; b++)
          {
            int  i;
            FLOAT  ecb = energy_s[sblock][j++];
            for (i = numlines_s[b] ; i > 0 ; i--)
              {
                ecb += energy_s[sblock][j++];
              }
            eb[b] = ecb;
          }
        for (b = 0 ; b < npart_s ; b++)
          {
            FLOAT8  ecb = 0;
            for (k = s3ind_s[b][0] ; k <= s3ind_s[b][1] ; k++)
              {
                ecb += s3_s[b][k] * eb[k];
              }
            thr[b] = ((qthr_s[b]) > (ecb)?(qthr_s[b]):(ecb));
          }
        for (sb = 0 ; sb < 12 ; sb++)
          {
            FLOAT8  enn = w1_s[sb] * eb[bu_s[sb]] + w2_s[sb] * eb[bo_s[sb]];
            FLOAT8  thmm = w1_s[sb] * thr[bu_s[sb]] + w2_s[sb] * thr[bo_s[sb]];
            for (b = bu_s[sb] + 1 ; b < bo_s[sb] ; b++)
              {
                enn += eb[b];
                thmm += thr[b];
              }
            en[chn].s[sb][sblock] = enn;
            thm[chn].s[sb][sblock] = thmm;
          }
      }
  }
astex_thread_end:;
  __astex_addr__b[0] = b;
  __astex_addr__j[0] = j;
  __astex_addr__k[0] = k;
  __astex_addr__sb[0] = sb;
  __astex_addr__sblock[0] = sblock;
}

