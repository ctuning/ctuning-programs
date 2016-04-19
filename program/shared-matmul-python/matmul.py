#
# Copyright (C) 2016 by Grigori G.Fursin
# http://fursin.net
#

def prep(A,B,C):
    import math

    for j in range(0,N):
        for i in range(0, N):
            D=float(i+j+2);

            A[i][j]=math.sin(D+1);
            B[i][j]=math.sin(D+2);
            C[i][j]=math.sin(D+3);

    return

def matmulsub(A,B,C):
    for i in range(0,N):
        for j in range(0,N):
            for k in range(0,N):
               A[i][j]=A[i][j]+B[i][k]*C[k][j];

    return

##############################################################################
if __name__ == "__main__":
   import os

   print("Basic Matmul test")
   print("")

   ct_repeat_max=1
   s_ct_repeat_max=os.environ.get("CT_REPEAT_MAIN", "")
   if s_ct_repeat_max!="":
      ct_repeat_max=long(s_ct_repeat_max)

   sN=os.environ.get("CT_MATRIX_DIMENSION", "")  
   N=128
   if sN!="":
      N=int(sN)

   print(N)

   A=[[0.0 for x in range(N)] for y in range(N)]
   B=[[0.0 for x in range(N)] for y in range(N)]
   C=[[0.0 for x in range(N)] for y in range(N)]

   print("Preparing ...")

   prep(A,B,C)

   print("Calculating ...")

   for i in range(0, ct_repeat_max):
       matmulsub(A,B,C)

   print("")
   print("X="+str(A[N/2][N/2]))

   print("")
   print("End")

   exit(0)
