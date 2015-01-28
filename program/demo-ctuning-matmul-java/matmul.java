/*
 Copyright (C) 2000-2013 by Grigori G.Fursin
 http://cTuning.org/lab/people/gfursin
*/

public class matmul
{
  static int N=16;
  static double[][] A=new double [N][N];
  static double[][] B=new double [N][N];
  static double[][] C=new double [N][N];

  // *******************************************************************
  public static void main(String args[]) 
  {
    System.out.println("Matmul from Grigori's PhD");
    System.out.println("");

    System.out.println("Preparing ...");

    prep();

    System.out.println("Calculating ...");

    long ct_repeat_max=1;
    String s_ct_repeat_max=System.getenv("CT_REPEAT_MAIN");
    if (s_ct_repeat_max!=null && s_ct_repeat_max!="") ct_repeat_max=Long.valueOf(s_ct_repeat_max);

    String sN=System.getenv("CT_MATRIX_DIMENSION");  
    if (sN!=null && sN!="") N=Integer.valueOf(sN);

    System.out.println(N);

    A=new double [N][N];
    B=new double [N][N];
    C=new double [N][N];

    for (int i=1;i<=ct_repeat_max;i++)
      matmulsub();

    System.out.println("");
    System.out.println("X="+A[N/2][N/2]);

    System.out.println("");
    System.out.println("End");
  }

  static void prep()
  {
    for (int j=0; j<N; j++)
    {           
      for (int i=0; i<N; i++)
      {
        double D=i+j+2;

        A[i][j]=Math.sin(D+1);
        B[i][j]=Math.sin(D+2);
        C[i][j]=Math.sin(D+3);
      }
    }
  }

  static void matmulsub()
  {
    for (int i=0; i<N; i++)
    {
      for (int j=0; j<N; j++)
      {
        for (int k=0; k<N; k++)
        {
          A[i][j]=A[i][j]+B[i][k]*C[k][j];
        }
      }
    }
  }
}
