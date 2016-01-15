#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "blowfish.h"

#ifdef XOPENME
#include <xopenme.h>
#endif

int
main(int argc, char *argv[])
{
	BF_KEY key;
	unsigned char ukey[32]; /* FGG changed mistake */
	unsigned char indata[40], outdata[40], ivec[32]={0}; /* FGG changed mistake */
	int num=0; /* FGG changed mistake */
	int by=0,i=0;
	int encordec=-1;
	char *cp,ch;
	FILE *fp,*fp2;

        long ct_repeat=0;
        long ct_repeat_max=1;
        int ct_return=0;

#ifdef XOPENME
  xopenme_init(1,0);
#endif

        if (getenv("CT_REPEAT_MAIN")!=NULL) ct_repeat_max=atol(getenv("CT_REPEAT_MAIN"));
       			  
if (argc<3)
{
	fprintf(stderr, "Usage: blowfish {e|d} <intput> <output> key\n");
	exit(EXIT_FAILURE);
}

if (*argv[1]=='e' || *argv[1]=='E')
	encordec = 1;
else if (*argv[1]=='d' || *argv[1]=='D')
	encordec = 0;
else
{
	fprintf(stderr, "Usage: blowfish {e|d} <intput> <output> key\n");
	exit(EXIT_FAILURE);
}
					

/* Read the key */
cp = argv[4];
while(i < 64 && *cp)    /* the maximum key length is 32 bytes and   */
{                       /* hence at most 64 hexadecimal digits      */
	ch = toupper(*cp++);            /* process a hexadecimal digit  */
	if(ch >= '0' && ch <= '9')
		by = (by << 4) + ch - '0';
	else if(ch >= 'A' && ch <= 'F')
		by = (by << 4) + ch - 'A' + 10;
	else                            /* error if not hexadecimal     */
	{
		printf("key must be in hexadecimal notation\n");
		exit(EXIT_FAILURE);
	}

	/* store a key byte for each pair of hexadecimal digits         */
	if(i++ & 1)
		ukey[i / 2 - 1] = by & 0xff;
}

BF_set_key(&key,8,ukey);

if(*cp)
{
	printf("Bad key value.\n");
	exit(EXIT_FAILURE);
}

/* open the input and output files */
if ((fp = fopen(argv[2],"r"))==0)
{
        fprintf(stderr, "Usage: blowfish {e|d} <intput> <output> key\n");
	exit(EXIT_FAILURE);
};
if ((fp2 = fopen(argv[3],"w"))==0)
{
        fprintf(stderr, "Usage: blowfish {e|d} <intput> <output> key\n");
	exit(EXIT_FAILURE);
};

i=0;
while(!feof(fp))
{
    int           current_num;
    unsigned char current_ivec[32];

	int j;
	while(!feof(fp) && i<40)
		indata[i++]=getc(fp);

    /* backup for multiple loop_wrap run */
    current_num = num;
    memcpy(current_ivec, ivec, 32);

#ifdef XOPENME
  xopenme_clock_start(0);
#endif

for (ct_repeat=0; ct_repeat<ct_repeat_max; ct_repeat++)
{
  /* The call to BF_cfb64_encrypt modifies ivec and num. We need to make a
     copy and to restore it before each iteration of the kernel to make sure
     we do not alter the output of the application. */
        num = current_num;
        memcpy(ivec, current_ivec, 32);
	BF_cfb64_encrypt(indata,outdata,i,&key,ivec,&num,encordec);
}

#ifdef XOPENME
  xopenme_clock_end(0);

  xopenme_dump_state();
  xopenme_finish();
#endif

	for(j=0;j<i;j++)
	{
		/*printf("%c",outdata[j]);*/
		fputc(outdata[j],fp2);
	}
	i=0;
}

fclose(fp);
fclose(fp2);

return 0;
}



