void init_trisolve(int n, double* L, double* x, double* b)
{
    for (int i = 0; i < n; i++)
    {
        x[i] = -999;
        b[i] = i ;
        for (int j = 0; j <= i; j++)
            L[i * n + j] = (double) (i + n - j + 1) * 2 / n;
    }
}