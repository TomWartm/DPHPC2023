
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>


static
void init_array (int n,
   double A[ 1300 + 0][1300 + 0],
   double B[ 1300 + 0][1300 + 0])
{
  
  for (int i = 0; i < n; i++)
   { for (int j = 0; j < n; j++)
      {
 A[i][j] = ((double) i*(j+2) + 2) / n;
 B[i][j] = ((double) i*(j+3) + 3) / n;
      }
    }
}







static
void kernel_jacobi_2d(int tsteps,
       int n,
       double A[ 1300 + 0][1300 + 0],
       double B[ 1300 + 0][1300 + 0])
{
  
#pragma scop
  for (int t = 0; t < tsteps; t++)
    {
      for (int i = 1; i < n - 1; i++)
{ for (int j = 1; j < n - 1; j++){
   B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);}}
      for (int i = 1; i < n - 1; i++){
 for (int j = 1; j < n - 1; j++){
   A[i][j] = 0.2 * (B[i][j] + B[i][j-1] + B[i][1+j] + B[1+i][j] + B[i-1][j]);}}
    }
#pragma endscop

}


int main(int argc, char** argv)
{

  int n = 1300;
  int tsteps = 500;


  double (*A)[1300 + 0][1300 + 0]= (double(*)[1300 + 0][1300 + 0])malloc (((1300 + 0) * (1300 + 0))* sizeof(double));;
  double (*B)[1300 + 0][1300 + 0]= (double(*)[1300 + 0][1300 + 0])malloc (((1300 + 0) * (1300 + 0))* sizeof(double));;



  init_array (n, *A, *B);


  kernel_jacobi_2d(tsteps, n, *A, *B);

  free((void*)A);;
  free((void*)B);;

  return 0;
}
