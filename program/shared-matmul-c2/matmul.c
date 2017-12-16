/*
 Copyright (C) 2000-2013 by Grigori G.Fursin
 http://cTuning.org/lab/people/gfursin
*/

#include <stdio.h>
#include <stdlib.h>

#define Q 16

#ifdef OPENME
#include <openme.h>
#endif
#ifdef XOPENME
#include <xopenme.h>
#endif

#define min(a,b) (a<b?a:b)

void matmul(float* A, float* B, float* C, int N, int BS);
void naive_matmul(float* A, float* B, float* C, int N, int II, int JJ, int KK);

int main(int argc, char* argv[])
{
  FILE* fgg=NULL;
  int N=0;
  int BS=1;
  float QQ[Q];
  int i=0;
  int j=0;
  int k=0;
  long l=0;

  float* A;
  float* B;
  float* C;

  long ct_repeat=0;
  long ct_repeat_max=1;
  int ct_return=0;

  char* fn;

#ifdef OPENME
  openme_init(NULL,NULL,NULL,0);
  openme_callback("PROGRAM_START", NULL);
#endif
#ifdef XOPENME
  xopenme_init(1,1);
#endif

  fn=argv[1];

  if ((getenv("CT_REPEAT_MAIN")!=NULL) && (getenv("CT_MATRIX_DIMENSION")!=NULL))
  {
    ct_repeat_max=atol(getenv("CT_REPEAT_MAIN"));
    N=atol(getenv("CT_MATRIX_DIMENSION"));
  } 
  else
  {
    if (argc<3)
    {
       printf("Usage:\n");
       printf("  matmul <data file> <matrix dimension> <repetitions>\n");
       return 1;
    }

    N=atoi(argv[2]);
    ct_repeat_max=atol(argv[3]);
  }

  if (getenv("CT_BLOCK_SIZE")!=NULL)
    BS=atol(getenv("CT_BLOCK_SIZE"));

  if ((fgg=fopen(fn,"rt"))==NULL)
  {
    fprintf(stderr,"\nError: Can't find data!\n");
    return 1;
  }

  for (i=0; i<Q; i++)
  {
    fscanf(fgg, "%f", &QQ[i]);
  }

  fclose(fgg);

  A=malloc(N*N*sizeof(float));
  B=malloc(N*N*sizeof(float));
  C=malloc(N*N*sizeof(float));

  k=0;
  for (l=0; l<N*N; l++)
  {
    A[l]=QQ[k++];
    if (k>=Q) k=0;
    B[l]=QQ[k++];
    if (k>=Q) k=0;
    C[l]=0;
  }

#ifdef OPENME
  openme_callback("KERNEL_START", NULL);
#endif
#ifdef XOPENME
  xopenme_clock_start(0);
#endif
  for (ct_repeat=0; ct_repeat<ct_repeat_max; ct_repeat++)
  {
    matmul(A,B,C,N,BS);
  }
#ifdef XOPENME
  xopenme_clock_end(0);
#endif
#ifdef OPENME
  openme_callback("KERNEL_END", NULL);
#endif

  //Print array to avoid dead code elimination
  for (i=0; i<N; i++)
  {
    printf("%u) %f %f %f\n", i, C[i*N+i], A[i*N+i], B[i*N+i]);
  }

  free(C);
  free(B);
  free(A);

#ifdef XOPENME
  xopenme_dump_state();
  xopenme_finish();
#endif
#ifdef OPENME
  openme_callback("PROGRAM_END", NULL);
#endif

  return 0;
}

/* Naive reference implementation */

#if USE_BLOCKED_MATMUL != YES

void matmul(float* A, float* B, float* C, int N, int BS)
{
  naive_matmul(A,B,C,N,N,N,N);
}

#else

void matmul(float* A, float* B, float* C, int N, int BS)
{
  int i,j,k;

  for (i=0; i<N; i+=BS)
  {
    for (j=0; j<N; j+=BS)
    {
      for (k=0; k<N; k+=BS)
      {
         /* fix iteration space */
         int II=min(BS,N-i);
         int JJ=min(BS,N-j);
         int KK=min(BS,N-k);

         naive_matmul(A+i+k*N,B+k+j*N,C+i+j*N,N,II,JJ,KK);
      }
    }
  }
}

#endif

void naive_matmul(float* A, float* B, float* C, int N, int II, int JJ, int KK)
{
  int i,j,k;
  float tmp;

  for (i=0; i<II; i++)
  {
    for (j=0; j<JJ; j++)
    {
      tmp=C[i+j*N];
      for (k=0; k<KK; k++)
      {
        tmp+=A[i+k*N]*B[k+j*N];
      }
      C[i+j*N]=tmp;
    }
  }
}
