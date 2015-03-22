typedef struct   {
  int  num_ents;
  int  entries[256][2];
} C_cell;

#pragma hmpp astex_codelet__1 codelet &
#pragma hmpp astex_codelet__1 , args[__astex_addr__astex_do_return].io=out &
#pragma hmpp astex_codelet__1 , args[__astex_addr__astex_what_return].io=out &
#pragma hmpp astex_codelet__1 , args[ptr].io=inout &
#pragma hmpp astex_codelet__1 , target=C &
#pragma hmpp astex_codelet__1 , version=1.4.0

void astex_codelet__1(C_cell *ptr, C_cell *__astex_addr__astex_what_return[1], int __astex_addr__astex_do_return[1])
{
  int  astex_do_return;
  astex_do_return = 0;
  C_cell  *astex_what_return;
  int  n;
  int  tmp;
  int  next_n;
  int  i;
astex_thread_begin:  {
    for (n = ptr->num_ents - 1 ; n > 0 ; n = next_n)
      {
        next_n = 0;
        for (i = 0 ; i < n ; ++i)
          if (ptr->entries[i][1] > ptr->entries[i + 1][1])
            {
              tmp = ptr->entries[i][0];
              ptr->entries[i][0] = ptr->entries[i + 1][0];
              ptr->entries[i + 1][0] = tmp;
              tmp = ptr->entries[i][1];
              ptr->entries[i][1] = ptr->entries[i + 1][1];
              ptr->entries[i + 1][1] = tmp;
              next_n = i;
            }
      }
    {
      astex_what_return = (ptr);
      astex_do_return = 1;
goto astex_thread_end;
    }
  }
astex_thread_end:;
  __astex_addr__astex_what_return[0] = astex_what_return;
  __astex_addr__astex_do_return[0] = astex_do_return;
}

