
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

static
void init_array (int n,
   double A[ 2000 + 0],
   double B[ 2000 + 0])
{
 
  for (int i = 0; i < n; i++)
      {
 A[i] = ((double) i+ 2) / n;
 B[i] = ((double) i+ 3) / n;
      }
}




static
void print_array(int n,
   double A[ 2000 + 0])

{
 
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "A");
  for (int i = 0; i < n; i++)
    {
      if (i % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2lf ", A[i]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "A");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}




static
void kernel_jacobi_1d(int tsteps,
       int n,
       double A[ 2000 + 0],
       double B[ 2000 + 0])
{
  
  for (int t = 0; t < tsteps; t++)
    {

      for (int i = 1; i < n - 1; i++){
 B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
}
      for (int i = 1; i < n - 1; i++){
 A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
}
    }

}


int main(int argc, char** argv)
{

  int n = 2000;
  int tsteps = 500;


  double (*A)[2000 + 0]; A = (double(*)[2000 + 0]) malloc ( (2000 + 0) * sizeof(double));;
  double (*B)[2000 + 0]; B = (double(*)[2000 + 0]) malloc ( (2000 + 0) * sizeof(double));;



  init_array (n, *A, *B);


  kernel_jacobi_1d(tsteps, n, *A, *B);

  if (argc > 42 && ! strcmp(argv[0], "")) print_array(n, *A);


  free((void*)A);;
  free((void*)B);;

  return 0;
}
