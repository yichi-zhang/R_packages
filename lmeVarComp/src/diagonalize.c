#include <stdlib.h>
#include <stdbool.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>


/**
  * Performs simultaneous diagonalization
  * A: k matrices of size p by p to be simultaneously diagonalized,
  *    column major and aligned one by one in memory
  * p: number of rows/columns of a matrix in A
  * k: number of matrices in A
  * tol: tolerance for equality or zero in eigenvalues,
  *      should be a small number like 1e-10
  * D: a p by k matrix for diagonal entries
  * fail: equal to 0 if successful
  */

void diagonalize(
    double *A, const int *_p, const int *_k,
    const double *_tol, double *D, int *fail)
{
    const int p = *_p, k = *_k;
    const int p2 = p * p;
    const char job = 'V', upper = 'U', entire = 'E';
    const char trans = 'T', no_trans = 'N';
    const double tol = *_tol;
    const double one_tol = 1.0 + tol;
    const double zero = 0.0, one = 1.0;

    /* eigenvalue decomposition of A_0 */
    int lwork = p2 + 10 * p;
    double *work = (double *)malloc(lwork * sizeof(double));
    if (work == 0) {
        *fail = -1; return;
    }
    bool *jump = (bool *)malloc(p * sizeof(bool));
    if (jump == 0) {
        *fail = -1; free(work); return;
    }

    F77_CALL(dsyev)(&job, &upper, &p, A, &p, D,
        work, &lwork, fail);
    if (D[0] < tol) D[0] = 0.0;
    for (int t = 1; t < p; ++t) {
        if (D[t] < tol) D[t] = 0.0;
        jump[t] = (D[t] > D[t - 1] * one_tol + tol);
    }

    /* A_j <- t(U) A_j U */
    double *U = A, *B;
    for (int j = 1; j < k; ++j) {
        B = A + p2 * j;
        F77_CALL(dgemm)(&trans, &no_trans, &p, &p, &p,
            &one, U, &p, B, &p, &zero, work, &p);
        F77_CALL(dgemm)(&no_trans, &no_trans, &p, &p, &p,
            &one, work, &p, U, &p, &zero, B, &p);
    }

    /* dealing with the rest A_i */
    double *subA;
    int info;
    for (int i = 1; i < k; ++i) {
        A += p2;
        D += p;

        int j1 = 0, j2, s;
        while (j1 < p) {
            j2 = j1 + 1;
            while ((j2 < p) && (!jump[j2])) ++j2;
            s = j2 - j1;

            if (s > 1) {
                /* eigenvalue decomposition of a subblock of A_i */
                subA = A + p * j1 + j1;
                F77_CALL(dsyev)(&job, &upper, &s, subA, &p, D + j1,
                    work, &lwork, &info);
                *fail += (info != 0);

                /* update U */
                B = U + p * j1;
                F77_CALL(dgemm)(&no_trans, &no_trans, &p, &s, &s,
                    &one, B, &p, subA, &p, &zero, work, &p);
                F77_CALL(dlacpy)(&entire, &p, &s,
                    work, &p, B, &p);

                /* update A_j <- t(U) A_j U */
                for (int j = 1; j < k - i; ++j) {
                    B = A + p2 * j + j1;
                    F77_CALL(dgemm)(&trans, &no_trans, &s, &p, &s,
                        &one, subA, &p, B, &p, &zero, work, &s);
                    F77_CALL(dlacpy)(&entire, &s, &p,
                        work, &s, B, &p);
                    B = A + p2 * j + p * j1;
                    F77_CALL(dgemm)(&no_trans, &no_trans, &p, &s, &s,
                        &one, B, &p, subA, &p, &zero, work, &p);
                    F77_CALL(dlacpy)(&entire, &p, &s,
                        work, &p, B, &p);
                }
            } else { /* i.e. s == 1 */
                D[j1] = A[p * j1 + j1];
            }

            j1 = j2;
        } /* while (j1 < p) */

        if (D[0] < tol) D[0] = 0.0;
        for (int t = 1; t < p; ++t) {
            if (D[t] < tol) D[t] = 0.0;
            jump[t] |= (D[t] > D[t - 1] * one_tol + tol);
        }
    } /* for (int i = 1; i < k; ++i) */

    free(jump);
    free(work);
}

