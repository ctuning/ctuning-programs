typedef struct colorbox  {
  struct colorbox  *next, *prev;
  int  rmin, rmax;
  int  gmin, gmax;
  int  bmin, bmax;
  int  total;
} Colorbox;

#pragma hmpp astex_codelet__6 codelet &
#pragma hmpp astex_codelet__6 , args[__astex_addr__astex_do_return].io=out &
#pragma hmpp astex_codelet__6 , args[__astex_addr__astex_what_return].io=out &
#pragma hmpp astex_codelet__6 , args[b].io=in &
#pragma hmpp astex_codelet__6 , target=C &
#pragma hmpp astex_codelet__6 , version=1.4.0

void astex_codelet__6(Colorbox *usedboxes, Colorbox *b, int size, Colorbox *__astex_addr__astex_what_return[1], int __astex_addr__astex_do_return[1])
{
  int  astex_do_return;
  astex_do_return = 0;
  Colorbox  *astex_what_return;
  Colorbox  *p;
astex_thread_begin:  {
    for (p = usedboxes ; p != ((void *) 0) ; p = p->next)
      if ((p->rmax > p->rmin || p->gmax > p->gmin || p->bmax > p->bmin) && p->total > size)
        size = (b = p)->total;
    {
      astex_what_return = (b);
      astex_do_return = 1;
goto astex_thread_end;
    }
  }
astex_thread_end:;
  __astex_addr__astex_what_return[0] = astex_what_return;
  __astex_addr__astex_do_return[0] = astex_do_return;
}

