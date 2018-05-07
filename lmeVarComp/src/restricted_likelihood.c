#include <stdlib.h>
#include <math.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>
#include "restricted_likelihood.h"




double nrl_f(const double *tau, int m, void *extra)
{
    static const char no_trans = 'N';
    static const int inc_one = 1;
    static const double zero = 0.0, one = 1.0;

    rl_extra *e = (rl_extra *)extra;
    const int n_minus_p = e->n_minus_p;
    const double *restrict D = e->D;
    const double *const restrict w2 = e->w2;
    double *const restrict Dtau = e->Dtau;

    /* Dtau <- c(D %*% tau) */
    F77_CALL(dgemv)(&no_trans, &n_minus_p, &m, &one, D, &n_minus_p,
        tau, &inc_one, &zero, Dtau, &inc_one);

    /* v1 <- sum(log(1 + Dtau)); v2 <- sum(w2 / (1 + Dtau)) */
    double v1 = 0.0, v2 = 0.0, temp;
    for (int i = 0; i < n_minus_p; ++i) {
        temp = 1.0 + Dtau[i];
        v1 += log(temp);
        v2 += w2[i] / temp;
    }

    /* v1 + n_minus_p * log(v2) */
    return (v1 + n_minus_p * log(v2));
}




void nrl_g(const double *tau, int m, void *extra, double *g)
{
    static const char no_trans = 'N';
    static const int inc_one = 1;
    static const double zero = 0.0, one = 1.0;

    rl_extra *e = (rl_extra *)extra;
    const int n_minus_p = e->n_minus_p;
    const double *restrict D = e->D;
    const double *const restrict w2 = e->w2;
    double *const restrict Dtau = e->Dtau;
    double *const restrict w2_Dtau = e->w2_Dtau;
    double *const restrict w2_Dtau2 = e->w2_Dtau2;

    /* Dtau <- c(D %*% tau) */
    F77_CALL(dgemv)(&no_trans, &n_minus_p, &m, &one, D, &n_minus_p,
        tau, &inc_one, &zero, Dtau, &inc_one);

    /* Dtau <- 1 + Dtau */
    /* w2_Dtau <- w2 / Dtau; w2_Dtau2 <- w2_Dtau / Dtau */
    /* v1 <- sum(w2_Dtau) */
    double v1 = 0.0;
    for (int i = 0; i < n_minus_p; ++i) {
        Dtau[i] += 1.0;
        w2_Dtau[i] = w2[i] / Dtau[i];
        w2_Dtau2[i] = w2_Dtau[i] / Dtau[i];
        v1 += w2_Dtau[i];
    }

    for (int k = 0; k < m; ++k) {
        /* v2 <- sum(D[, k] / Dtau); v3 <- sum(D[, k] * w2_Dtau2) */
        double v2 = 0.0, v3 = 0.0;
        for (int i = 0; i < n_minus_p; ++i) {
            v2 += D[i] / Dtau[i];
            v3 += D[i] * w2_Dtau2[i];
        }
        /* g[k] <- v2 - n_minus_p / v1 * v3 */
        g[k] = v2 - n_minus_p / v1 * v3;
        D += n_minus_p;
    }
}




double rss(const double *tau, int m, void *extra)
{
    /* modified from nrl_f */
    static const char no_trans = 'N';
    static const int inc_one = 1;
    static const double zero = 0.0, one = 1.0;

    rl_extra *e = (rl_extra *)extra;
    const int n_minus_p = e->n_minus_p;
    const double *restrict D = e->D;
    const double *const restrict w2 = e->w2;
    double *const restrict Dtau = e->Dtau;

    /* Dtau <- c(D %*% tau) */
    F77_CALL(dgemv)(&no_trans, &n_minus_p, &m, &one, D, &n_minus_p,
        tau, &inc_one, &zero, Dtau, &inc_one);

    /* v2 <- sum(w2 / (1 + Dtau)) */
    double v2 = 0.0;
    for (int i = 0; i < n_minus_p; ++i)
        v2 += w2[i] / (1.0 + Dtau[i]);

    return v2;
}



