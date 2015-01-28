/**
 * ludcmp.h: This file is part of the PolyBench/C 3.2 test suite.
 *
 *
 * Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://polybench.sourceforge.net
 */
#ifndef LUDCMP_H
# define LUDCMP_H

/* Default to STANDARD_DATASET. */
# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#  define STANDARD_DATASET
# endif

/* Do not define anything if the user manually defines the size. */
# ifndef N
/* Define the possible dataset sizes. */
#  ifdef MINI_DATASET
#   define N 32
#  endif

#  ifdef SMALL_DATASET
#   define N 128
#  endif

#  ifdef STANDARD_DATASET /* Default if unspecified. */
#   define N 1024
#  endif

#  ifdef LARGE_DATASET
#   define N 2000
#  endif

#  ifdef EXTRALARGE_DATASET
#   define N 4000
#  endif
# endif /* !N */

# define _PB_N POLYBENCH_LOOP_BOUND(N,n)

# ifndef DATA_TYPE
#  define DATA_TYPE double
# endif

# if DATA_TYPE == double
#  define DATA_PRINTF_MODIFIER "%0.2lf "
# elif DATA_TYPE == float
#  define DATA_PRINTF_MODIFIER "%0.2f "
# elif DATA_TYPE == long
#  define DATA_PRINTF_MODIFIER "%0.2u "
# elif DATA_TYPE == int
#  define DATA_PRINTF_MODIFIER "%0.2u "
# endif

#endif /* !LUDCMP */
