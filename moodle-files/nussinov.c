/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* nussinov.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include <stdlib.h>

typedef char base;





static
void init_array (int n,
                 base seq[ 2500 + 0],
   int table[ 2500 + 0][2500 + 0])
{
  

  for (int i=0; i <n; i++) {
     seq[i] = (base)((i+1)%4);
  }

  for (int i=0; i <n; i++)
     for (int j=0; j <n; j++)
       table[i][j] = 0;
}




static
void print_array(int n,
   int table[ 2500 + 0][2500 + 0])

{
  int i, j;
  int t = 0;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "table");
  for (i = 0; i < n; i++) {
    for (j = i; j < n; j++) {
      if (t % 20 == 0) fprintf (stderr, "\n");
      fprintf (stderr, "%d ", table[i][j]);
      t++;
    }
  }
  fprintf(stderr, "\nend   dump: %s\n", "table");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static
void kernel_nussinov(int n, base seq[ 2500 + 0],
      int table[ 2500 + 0][2500 + 0])
{
int tmp;
int tmp2;
int tmp3;
#pragma scop
 for (int i = n-1; i >= 0; i--) {
  for (int j=i+1; j<n; j++) {

   if (j-1>=0){
      tmp=(table[i][j] >= table[i][j-1]);
      if tmp!=0{
      table[i][j] = table[i][j];
      }
      else{      table[i][j] = table[i][j-1];}
      }
   if (i+1<n){
      tmp=(table[i][j] >= table[i+1][j]);
      if tmp!=0{
      table[i][j] =  table[i][j] ;
       }else{
       table[i][j] = table[i+1][j];}
       }
    tmp=j-1>=0 && i+1<n;
   if (tmp!=0) {

     if (i<j-1)

     {  tmp2=((seq[i])+(seq[j])) == 3;
        if tmp2!=0{tmp3=1;}
        else {tmp3=0;}
        tmp=table[i][j] >= table[i+1][j-1]+tmp3;
        if (tmp!=0)
        {table[i][j] = table[i][j];}
        else{
         table[i][j] = table[i+1][j-1]+tmp3;
        }
     }
     else
        tmp=(table[i][j] >= table[i+1][j-1]);
        if tmp!=0
            {table[i][j] =  table[i][j];}
        else{table[i][j] =  table[i+1][j-1]);}

   }

   for (int k=i+1; k<j; k++) {
   tmp=(table[i][j] >= table[i][k] + table[k+1][j]);
   if (tmp!=0){
      table[i][j] =  table[i][j];}
       else{
      table[i][j] =  table[i][k] + table[k+1][j];}
   }
  }
 }
#pragma endscop

}


int main(int argc, char** argv)
{

  int n = 2500;


  base (*seq)[2500 + 0]; seq = (base(*)[2500 + 0])malloc ((2500 + 0)* sizeof(base));;
  int (*table)[2500 + 0][2500 + 0]; table = (int(*)[2500 + 0][2500 + 0])malloc ((2500 + 0) * (2500 + 0)* sizeof(int));;


  init_array (n, *seq, *table);



  kernel_nussinov (n, *seq, *table);




  if (argc > 42 && ! strcmp(argv[0], "")) 
print_array(n, *table);

  free((void*)seq);;
  free((void*)table);;

  return 0;
}
