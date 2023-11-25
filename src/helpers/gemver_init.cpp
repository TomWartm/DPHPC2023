void init_gemver(int n, double *alpha, double *beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z) {

    *alpha = 1.5;
    *beta = 1.2;

    double fn = (double) n;

    for (int i = 0; i < n; i++)
    {
        u1[i] = i;
        u2[i] = ((i + 1) / fn) / 2.0;
        v1[i] = ((i + 1) / fn) / 4.0;
        v2[i] = ((i + 1) / fn) / 6.0;
        y[i] = ((i + 1) / fn) / 8.0;
        z[i] = ((i + 1) / fn) / 9.0;
        x[i] = 0.0;
        w[i] = 0.0;
        for (int j = 0; j < n; j++)
            A[i * n + j] = (double)(i * j % n) / n;
    }
}

void init_gemver(int n, int m,  double *alpha, double *beta, double *A, double *u1, double *v1, double *u2, double *v2, double *w, double *x, double *y, double *z) {
    // init version of not square matrix
    *alpha = 1.5;
    *beta = 1.2;

    double fn = (double) n;

    for (int i = 0; i < n; i++)
    {
        u1[i] = i;
        u2[i] = ((i + 1) / fn) / 2.0;
        w[i] = 0.0;
        
        y[i] = ((i + 1) / fn) / 8.0;
        
        for (int j = 0; j < m; j++)
            A[i * m + j] = (double)(i * j % n) / n;
    }

    for (int j = 0; j < m; j++){
        v1[j] = ((j + 1) / fn) / 4.0;
        v2[j] = ((j + 1) / fn) / 6.0;
        z[j] = ((j + 1) / fn) / 9.0;
        x[j] = 0.0;
        
    }
}