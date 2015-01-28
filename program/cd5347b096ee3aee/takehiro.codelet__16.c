typedef struct   {
  unsigned  part2_3_length;
  unsigned  big_values;
  unsigned  count1;
  unsigned  global_gain;
  unsigned  scalefac_compress;
  unsigned  window_switching_flag;
  unsigned  block_type;
  unsigned  mixed_block_flag;
  unsigned  table_select[3];
  int  subblock_gain[3];
  unsigned  region0_count;
  unsigned  region1_count;
  unsigned  preflag;
  unsigned  scalefac_scale;
  unsigned  count1table_select;
  unsigned  part2_length;
  unsigned  sfb_lmax;
  unsigned  sfb_smax;
  unsigned  count1bits;
  unsigned  *sfb_partition_table;
  unsigned  slen[4];
} gr_info;

struct huffcodetab  {
  unsigned int  xlen;
  unsigned int  linmax;
  unsigned long int  *table;
  unsigned char  *hlen;
} ;

#pragma hmpp astex_codelet__16 codelet &
#pragma hmpp astex_codelet__16 , args[__astex_addr__astex_do_return].io=out &
#pragma hmpp astex_codelet__16 , args[__astex_addr__astex_what_return].io=out &
#pragma hmpp astex_codelet__16 , args[__astex_addr__bits].io=inout &
#pragma hmpp astex_codelet__16 , args[__astex_addr__i].io=inout &
#pragma hmpp astex_codelet__16 , args[ht].io=in &
#pragma hmpp astex_codelet__16 , args[gi].io=inout &
#pragma hmpp astex_codelet__16 , args[ix].io=in &
#pragma hmpp astex_codelet__16 , target=C &
#pragma hmpp astex_codelet__16 , version=1.4.0

void astex_codelet__16(int ix[576], gr_info *gi, struct huffcodetab ht[34], int __astex_addr__i[1], int __astex_addr__bits[1], int __astex_addr__astex_what_return[1], int __astex_addr__astex_do_return[1])
{
  int  astex_do_return;
  astex_do_return = 0;
  int  astex_what_return;
  int  bits = __astex_addr__bits[0];
  int  a2;
  int  a1;
  int  i = __astex_addr__i[0];
astex_thread_begin:  {
    for ( ; i > 1 ; i -= 2)
      if (ix[i - 1] | ix[i - 2])
        break;
    gi->count1 = i;
    a1 = 0;
    for ( ; i > 3 ; i -= 4)
      {
        int  p, v;
        if ((unsigned int ) (ix[i - 1] | ix[i - 2] | ix[i - 3] | ix[i - 4]) > 1)
          break;
        v = ix[i - 1];
        p = v;
        bits += v;
        v = ix[i - 2];
        if (v != 0)
          {
            p += 2;
            bits++;
          }
        v = ix[i - 3];
        if (v != 0)
          {
            p += 4;
            bits++;
          }
        v = ix[i - 4];
        if (v != 0)
          {
            p += 8;
            bits++;
          }
        a1 += ht[32].hlen[p];
      }
    a2 = gi->count1 - i;
    if (a1 < a2)
      {
        bits += a1;
        gi->count1table_select = 0;
      }
    else     {
      bits += a2;
      gi->count1table_select = 1;
    }
    gi->count1bits = bits;
    gi->big_values = i;
    if (i == 0)
      {
        astex_what_return = bits;
        astex_do_return = 1;
goto astex_thread_end;
      }
  }
astex_thread_end:;
  __astex_addr__i[0] = i;
  __astex_addr__bits[0] = bits;
  __astex_addr__astex_what_return[0] = astex_what_return;
  __astex_addr__astex_do_return[0] = astex_do_return;
}

