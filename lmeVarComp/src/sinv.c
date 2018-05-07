#include <stdlib.h>
#include <string.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>


/**
  * Computes the inverse of a symmetric matrix
  * A: symmetric matrix, n by n
  */

void sinv(double *A, const int *_n)
{
    const int n = *_n;
    const char upper = 'U';

    int *ipiv = (int *)malloc(n * sizeof(int));
    if (ipiv == 0) return;

    /* factorization using Bunch-Kaufman method */
    int info, lwork = -1;
    double temp;
    F77_CALL(dsytrf)(&upper, &n, A, &n, ipiv, &temp, &lwork, &info);
    lwork = (info == 0) ? (int)temp : (4 * n);
    if (lwork < n) lwork = n;

    double *work = (double *)malloc(lwork * sizeof(double));
    if (work == 0) {free(ipiv); return;}
    F77_CALL(dsytrf)(&upper, &n, A, &n, ipiv, work, &lwork, &info);

    /* inverse */
    F77_CALL(dsytri)(&upper, &n, A, &n, ipiv, work, &info);

    for (int j = 0; j < n; ++j) {
        double *restrict d = A + j * n;
        for (int i = j + 1; i < n; ++i)
            d[i] = A[i * n + j];
    }

    free(work);
    free(ipiv);
}

