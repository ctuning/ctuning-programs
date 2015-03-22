/*
   This program was modified (incrementally simplified) in 2013-2014
   by Grigori Fursin to understand performance regressions
   for different images ...
*/

/* This program detects the edges in a 256 gray-level 128 x 128 pixel image.
   The program relies on a 2D-convolution routine to convolve the image with
   kernels (Sobel operators) that expose horizontal and vertical edge
   information.

   The following is a block diagram of the steps performed in edge detection,


            +---------+       +----------+
   Input    |Smoothing|       |Horizontal|-------+
   Image -->| Filter  |---+-->| Gradient |       |
            +---------+   |   +----------+  +----x-----+   +---------+  Binary
                          |                 | Gradient |   |  Apply  |  Edge
                          |                 | Combining|-->|Threshold|->Detected
                          |   +----------+  +----x-----+   +----x----+  Output
                          |   | Vertical |       |              |
                          +-->| Gradient |-------+              |
                              +----------+                   Threshold
                                                               Value


    This program is based on the routines and algorithms found in the book
    "C Language Algorithms for Digital Signal Processing" by P.M. Embree
    and B. Kimble.

    Copyright (c) 1992 -- Mazen A.R. Saghir -- University of Toronto */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/* #include <string.h> */
/* #include "traps.h" */

#include "filter.h"

FILE *input_fp=NULL;
FILE *output_fp=NULL;

int input_dsp();   /* input_dsp  (dest,   char*) */
void output_dsp(); /* output_dsp (source) */

void filter_codelet();

int image_buffer1[N][N];
int image_buffer2[N][N];
int image_buffer3[N][N];
int filter[K][K];

int main(int argc, const char **argv)
{
  int *matrix_ptr1;
  int *matrix_ptr2;
  int *matrix_ptr3;
  int *filter_ptr;
  int temp1;
  int temp2;
  int temp3;
  int v1;
  int v2;
  int v3;
  int i;
  int j;

  void convolve2d();

  long rr=0, r=1;
  int tm=0;
  char *stm;

  if (argc<1)
  {
    printf("Usage: ./a.out <input file name with an image in a raw format\n");
    return 1;
  }

  /* FGG adding kernel repetition */
  if (getenv("CT_REPEAT_MAIN")!=NULL) r=atol(getenv("CT_REPEAT_MAIN"));

  /* FGG adding "time of the day" feature */
  stm=getenv("CT_TIME");
  if (stm!=NULL)
  {
     if (strcmp(stm, "day")==0) tm=0;
     else if (strcmp(stm, "night")==0) tm=1;
  }

  /* Read input image. */
  input_dsp(image_buffer1, argv[1]);

  /* Initialize image_buffer2 and image_buffer3 */
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; ++j) {
       image_buffer2[i][j] = 0;
     }
  }

#pragma omp parallel for 
  printf("Time of the day feature: %u\n",tm);
  for (rr=0; rr<r; rr++) {
   if (tm==0) filter_codelet_day(image_buffer1, image_buffer2);
   else filter_codelet_night(image_buffer1, image_buffer2);
  }

  /* Store binary image. */

  output_dsp(image_buffer2, N*N);

  return 0;
}


/* This function convolves the input image by the kernel and stores the result
   in the output image. */

void convolve2d(input_image, kernel, output_image)
int *input_image;
int *kernel;
int *output_image;
{
  int *kernel_ptr;
  int *input_image_ptr;
  int *output_image_ptr;
  int *kernel_offset;
  int *input_image_offset;
  int *output_image_offset;
  int i;
  int j;
  int c;
  int r;
  int row;
  int col;
  int normal_factor;
  int sum;
  int temp1;
  int temp2;
  int dead_rows;
  int dead_cols;

  /* Set the number of dead rows and columns. These represent the band of rows
     and columns around the edge of the image whose pixels must be formed from
     less than a full kernel-sized compliment of input image pixels. No output
     values for these dead rows and columns since  they would tend to have less
     than full amplitude values and would exhibit a "washed-out" look known as
     convolution edge effects. */

  dead_rows = K / 2;
  dead_cols = K / 2;

  /* Calculate the normalization factor of the kernel matrix. */

  normal_factor = 0;
  kernel_ptr = kernel;
  for (r = 0; r < K; r++) {
    kernel_offset = kernel_ptr;
    temp1 = *kernel_offset++;
    for (c = 1; c < K; c++) {
      normal_factor += abs(temp1);
      temp1 = *kernel_offset++;
    }
    normal_factor += abs(temp1);
    kernel_ptr += K;
  }

  if (normal_factor == 0)
    normal_factor = 1;

  /* Convolve the input image with the kernel. */

  row = 0;
  output_image_ptr = output_image;
  output_image_ptr += (N * dead_rows);
  for (r = 0; r < N - K + 1; r++) {
    output_image_offset = output_image_ptr;
    output_image_offset += dead_cols;
    col = 0;
    for (c = 0; c < N - K + 1; c++) {
      input_image_ptr = input_image;
      input_image_ptr += (N * row);
      kernel_ptr = kernel;
      sum = 0;
      for (i = 0; i < K; i++) {
        input_image_offset = input_image_ptr;
        input_image_offset += col;
        kernel_offset = kernel_ptr;
        temp1 = *input_image_offset++;
        temp2 = *kernel_offset++;
        for (j = 1; j < K; j++) {
          sum += temp1 * temp2;
          temp1 = *input_image_offset++;
          temp2 = *kernel_offset++;
        }
        sum += temp1 * temp2;
        kernel_ptr += K;
        input_image_ptr += N;
      }
      *output_image_offset++ = (sum / normal_factor);
      col++;
    }
    output_image_ptr += N;
    row++;
  }
}

int input_dsp (int *dest, char *fgg_file)
{
  int success;

  input_fp=fopen(fgg_file,"rb");

  if (input_fp==NULL) {
    printf ("Error: cannot open input image file %s ...\n", fgg_file);
    exit(1);
  }

  success=fread(dest, N, N*sizeof(int), input_fp);
  fclose(input_fp);

  return success;
}

void output_dsp (int *src)
{
  output_fp=fopen("image_output.bin","wb");
  fwrite(src, N, N*sizeof(int), output_fp);
  fclose(output_fp);
}
