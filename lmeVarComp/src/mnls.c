#include <stdlib.h>
#include <string.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>


/**
  * Computes the minimum-norm least squares solution
  * A: predictor matrix, m by n
  * B: response matrix, m by nrhs
  * X: solution, n by nrhs
  * m: number of observations
  * n: number of predictors
  * nrhs: number of responses
  * rcond: reciprocal condition number for singularity
  *        should be a small number like 1e-10
  * rank: estimated rank of A
  */

void mnls(
    double *A, double *B, double *X,
    const int *_m, const int *_n, const int *_nrhs,
    const double *_rcond, int *_rank)
{
    const int m = *_m, n = *_n, nrhs = *_nrhs;
    const double rcond = *_rcond;
    const char entire = 'E';

    int *jpvt = (int *)malloc(n * sizeof(int));
    if (jpvt == 0) return;
    memset(jpvt, 0, n * sizeof(int));
    int rank = 0;

    double *data;
    int k;
    if (m >= n) {
        data = B; k = m;
    } else { /* i.e. m < n */
        data = X; k = n;
        F77_CALL(dlacpy)(&entire, &m, &nrhs, B, &m, X, &n);
    }

    int info, lwork = -1;
    double temp;
    F77_CALL(dgelsy)(&m, &n, &nrhs, A, &m, data, &k,
        jpvt, &rcond, &rank, &temp, &lwork, &info);
    lwork = (info == 0) ? (int)temp : (4 * (m + n + nrhs));

    double *work = (double *)malloc(lwork * sizeof(double));
    if (work == 0) {free(jpvt); return;}
    F77_CALL(dgelsy)(&m, &n, &nrhs, A, &m, data, &k,
        jpvt, &rcond, &rank, work, &lwork, &info);

    if (m >= n)
        F77_CALL(dlacpy)(&entire, &n, &nrhs, B, &m, X, &n);

    *_rank = rank;
    free(work);
    free(jpvt);
}

