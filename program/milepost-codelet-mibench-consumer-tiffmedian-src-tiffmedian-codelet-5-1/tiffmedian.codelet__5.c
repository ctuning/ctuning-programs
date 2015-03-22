typedef unsigned short  uint16;

typedef uint16  tdir_t;

typedef uint16  tsample_t;

typedef struct   {
  int  num_ents;
  int  entries[256][2];
} C_cell;

#pragma hmpp astex_codelet__5 codelet &
#pragma hmpp astex_codelet__5 , args[cell].io=in &
#pragma hmpp astex_codelet__5 , args[histp].io=inout &
#pragma hmpp astex_codelet__5 , args[bm].io=in &
#pragma hmpp astex_codelet__5 , args[gm].io=in &
#pragma hmpp astex_codelet__5 , args[rm].io=in &
#pragma hmpp astex_codelet__5 , target=C &
#pragma hmpp astex_codelet__5 , version=1.4.0

void astex_codelet__5(uint16 rm[256], uint16 gm[256], uint16 bm[256], int *histp, C_cell *cell, int dist, int ir, int ig, int ib)
{
  int  i;
  int  d2;
  int  tmp;
  int  j;
astex_thread_begin:  {
    for (i = 0 ; i < cell->num_ents && dist > cell->entries[i][1] ; ++i)
      {
        j = cell->entries[i][0];
        d2 = rm[j] - (ir << (8 - 5));
        d2 *= d2;
        tmp = gm[j] - (ig << (8 - 5));
        d2 += tmp * tmp;
        tmp = bm[j] - (ib << (8 - 5));
        d2 += tmp * tmp;
        if (d2 < dist)
          {
            dist = d2;
            *histp = j;
          }
      }
  }
astex_thread_end:;
}

