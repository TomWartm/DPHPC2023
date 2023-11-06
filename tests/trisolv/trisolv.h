/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_trisolv(int n,
		    double* A,
		    double* x,
		    double* b)
{
  int i, j;

#pragma scop
	for (i = 0; i < n; i++)
    {
		x[i] = b[i];
		for (j = 0; j < i; j++)
			x[i] -= A[i * n + j] * x[j];
		x[i] = x[i] / A[i * n + i];
    }
#pragma endscop

}