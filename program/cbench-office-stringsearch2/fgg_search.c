#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef XOPENME
#include <xopenme.h>
#endif

static char str_misc[1024];

/*
 * Copyright (c) 1989, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by the University of
 *	California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * $FreeBSD: src/sys/libkern/strncmp.c,v 1.7 1999/08/28 00:46:38 peter Exp $
 */

int
local_strncmp(s1, s2, n)
	register const char *s1, *s2;
	register size_t n;
{

	if (n == 0)
		return (0);
	do {
		if (*s1 != *s2++)
			return (*(const unsigned char *)s1 -
				*(const unsigned char *)(s2 - 1));
		if (*s1++ == 0)
			break;
	} while (--n != 0);
	return (0);
}


char *strsearch2(char *string, char *search)
{
   int i;

   for (i=0; i<strlen(string); i++)
   {
     if (local_strncmp(&string[i], search, strlen(search))==0)
        return &string[i];
   }

   return NULL;
}

int main(int argc, char* argv[])
{
      char *here = NULL;
      int i,j;

      long ct_repeat=0;
      long ct_repeat_max=1;
      int ct_return=0;
      FILE* fmisc1=NULL;
      FILE* fmisc2=NULL;
      FILE* fmisc3=NULL;

      long size1=0;
      long size2=0;
      char* search_strings[30000];
      char* find_strings[30000];
      long i1=0;
      long i2=0;
      char* a1x;
      char* a2x;

      if (argc<3)
      {
        fprintf(stderr, "Error: too few parameters!\n");
	exit(1);
      }

#ifdef XOPENME
  xopenme_init(1,0);
#endif

      if (getenv("CT_REPEAT_MAIN")!=NULL) ct_repeat_max=atol(getenv("CT_REPEAT_MAIN"));

      if ((fmisc1=fopen(argv[1],"rt"))==NULL)
      {
        fprintf(stderr,"\nError: Can't find text!\n");
        exit(1);
      }

      if ((fmisc2=fopen(argv[2],"rt"))==NULL)
      {
        fprintf(stderr,"\nError: Can't find search strings!\n");
        exit(1);
      }

      //counting file1
      i1=0;
      while ((fgets(str_misc, 1024, fmisc1)!=NULL) && (feof(fmisc1)==0))
      {
        i1++;
	size1+=strlen(str_misc)+1;
      }

      //counting file2
      i2=0;
      while ((fgets(str_misc, 1024, fmisc2)!=NULL) && (feof(fmisc2)==0))
      {
        i2++;
	size2+=strlen(str_misc)+1;
      }

      fclose(fmisc1);
      fclose(fmisc2);
      
      printf("Size1=%lu, size2=%lu\n", size1, size2);

      a1x=(char*) malloc(sizeof(char)*size1);
      a2x=(char*) malloc(sizeof(char)*size2);

      if ((fmisc1=fopen(argv[1],"rt"))==NULL)
      {
        fprintf(stderr,"\nError: Can't find text!\n");
        exit(1);
      }

      if ((fmisc2=fopen(argv[2],"rt"))==NULL)
      {
        fprintf(stderr,"\nError: Can't find search strings!\n");
        exit(1);
      }

      //loading file1
      i1=0;
      size1=0;
      while ((fgets(str_misc, 1023, fmisc1)!=NULL) && (feof(fmisc1)==0))
      {
        if (strlen(str_misc)>0) str_misc[strlen(str_misc)-1]=0;
        strcpy(&a1x[size1], str_misc);
        search_strings[i1]=&a1x[size1];
        i1++;
	size1+=strlen(str_misc)+1;
      }
      search_strings[i1]=0;

      //loading file2
      i2=0;
      size2=0;
      while ((fgets(str_misc, 1023, fmisc2)!=NULL) && (feof(fmisc2)==0))
      {
        if (strlen(str_misc)>0) str_misc[strlen(str_misc)-1]=0;
        strcpy(&a2x[size2], str_misc);
        find_strings[i2]=&a2x[size2];
        i2++;
	size2+=strlen(str_misc)+1;
      }
      find_strings[i2]=0;

      fclose(fmisc1);
      fclose(fmisc2);

      if ((fmisc3=fopen(argv[3],"w"))==NULL)
      {
        fprintf(stderr,"\nError: Can't open file for writing!\n");
        exit(1);
      }

#ifdef XOPENME
  xopenme_clock_start(0);
#endif

      for (i = 0; i<i1; i++)
      {
         for (j = 0; j<i2; j++)
         {
            for (ct_repeat=0; ct_repeat<ct_repeat_max; ct_repeat++)
              here = strsearch2(search_strings[i], find_strings[j]);
	      
	    if (here!=NULL)
               fprintf(fmisc3, "\"%s\" is in \"%s\"\n", find_strings[j], search_strings[i]); 
         }
      }

#ifdef XOPENME
  xopenme_clock_end(0);

  xopenme_dump_state();
  xopenme_finish();
#endif

      fclose(fmisc3);

      return 0;
}

