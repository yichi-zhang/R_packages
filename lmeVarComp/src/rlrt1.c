#include <stdlib.h>
#include <string.h>
#include <math.h>
/* define MATHLIB_STANDALONE */
#include <Rmath.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>
#include <R_ext/Random.h>
#include "restricted_likelihood.h"
#include "projected_bfgs.h"


/**
  * Simulate samples from null distributions of RLRT and GFT when m0 >= 1.
  * Ytilde2: vector, of length (n - p)
  * D: matrix, of size (n - p) by m
  * S0: matrix, of size (m0 + 1) by (m0 + 1)
  * S1: matrix, of size (m1 + 1) by (m1 + 1)
  * tau0: vector, of length m0
  *       (IN) initial value
  *       (OUT) estimate of tau under H0
  * tau1: vector, of length m1
  *       (IN) initial value
  *       (OUT) estimate of tau under H1
  * rlrt_obs: (OUT) observed test statistic
  * rlrt_sim: (OUT) vector, of length nsim, simulated test statistics
  * gft_obs: (OUT) observed test statistic
  * gft_sim: (OUT) vector, of length nsim, simulated test statistics
  * n: sample size, number of rows in X
  * p: number of fixed effects, i.e. number of columns in X
  * m0: number of variance components under H0
  * m1: number of variance components under H1
  * nsim: number of simulated test statistics
  * tol: tolerance in optimization
  * b: vector, of length 4, bounds of RLRT and GFT statistics
  *    for not performing simulation
  */

void rlrt1(
    double *Ytilde2, const double *D,
    const double *S0, const double *S1, double *tau0, double *tau1,
    double *rlrt_obs, double *rlrt_sim,
    double *gft_obs, double *gft_sim,
    int *_n, int *_p, int *_m0, int *_m1, int *_nsim,
    double *_tol, double *_b)
{
    const int n = *_n, p = *_p;
    const int n_minus_p = n - p;
    const int m0 = *_m0, m1 = *_m1;
    const int k0 = m0 + 1, k1 = m1 + 1;
    const int nsim = *_nsim;
    const double tol = *_tol;
    const double br1 = _b[0], br2 = _b[1], bg1 = _b[2], bg2 = _b[3];

    /* work space */
    int lwork = k1 * 2 + m1 * 3 + n_minus_p * 4;
    double *const work0 = (double *)malloc(lwork * sizeof(double));
    if (work0 == 0) {
        *rlrt_obs = *gft_obs = 0.0;
        return;
    }
    double *work = work0;

    rl_extra extra;
    extra.n_minus_p = n_minus_p;
    extra.D = D;
    extra.w2 = Ytilde2;
    double *const restrict Q     = work; work += k1;
    double *const restrict guess = work; work += k1;
    double *const restrict init  = work; work += m1;
    double *const restrict lower = work; work += m1;
    double *const restrict upper = work; work += m1;
    double *const restrict Dtau0 = work; work += n_minus_p;
    extra.Dtau                   = work; work += n_minus_p;
    extra.w2_Dtau                = work; work += n_minus_p;
    extra.w2_Dtau2               = work; work += n_minus_p;

    for (int h = 0; h < m1; ++h) {
        init[h] = 0.0;
        lower[h] = 0.0; upper[h] = INFINITY;
    }

    /* observed test statistics */
    double sr, r0, r1, sg, g0, g1;

    /* under H0 */
    projected_bfgs(init, tau0, m0, &extra, nrl_f, nrl_g, 100, tol,
        lower, upper, &r0);
    g0 = rss(init, m0, &extra);
    for (int h = 0; h < m0; ++h)
        tau0[h] = init[h];

    /* under H1 */
    projected_bfgs(init, tau1, m1, &extra, nrl_f, nrl_g, 100, tol,
        lower, upper, &r1);
    g1 = rss(init, m1, &extra);
    for (int h = 0; h < m1; ++h)
        tau1[h] = init[h];

    /* difference */
    sr = r0 - r1;
    *rlrt_obs = sr;
    sg = (g0 - g1) / (g1 / n);
    *gft_obs = sg;

    if (((sr <= br1) || (sr >= br2)) && ((sg <= bg1) || (sg >= bg2))) {
        /* p-value is 1 if sr <= br1, or sg <= bg1 */
        /* p-value is 0 if sr >= br2, or sg >= bg2 */
        free(work0); return;
    }

    static const char no_trans = 'N', trans = 'T', upper_part = 'U';
    static const int inc_one = 1;
    static const double zero = 0.0, one = 1.0;

    /* Dtau0 <- 1 + c(D[, 1 : m0] %*% tau0) */
    F77_CALL(dgemv)(&no_trans, &n_minus_p, &m0, &one, D, &n_minus_p,
        tau0, &inc_one, &zero, Dtau0, &inc_one);
    for (int j = 0; j < n_minus_p; ++j)
        Dtau0[j] += 1.0;

    /* simulated test statistics */
    GetRNGstate();
    double *restrict w2 = Ytilde2;
    double sum_w2, z;

    for (int i = 0; i < nsim; ++i) {
        /* w2 <- Dtau0 * rnorm(n_minus_p) ^ 2; sum_w2 <- sum(w2) */
        sum_w2 = 0.0;
        for (int j = 0; j < n_minus_p; ++j) {
            z = norm_rand();
            w2[j] = Dtau0[j] * z * z;
            sum_w2 += w2[j];
        }

        /* Q <- c(t(D) %*% w2, sum_w2) */
        F77_CALL(dgemv)(&trans, &n_minus_p, &m1, &one, D,
            &n_minus_p, w2, &inc_one, &zero, Q, &inc_one);
        Q[m1] = sum_w2;

        for (int h = 0; h < m1; ++h)
            init[h] = 0.0;

        /* under H0 */
        z = Q[m0]; Q[m0] = sum_w2;
        F77_CALL(dsymv)(&upper_part, &k0, &one, S0, &k0,
            Q, &inc_one, &zero, guess, &inc_one);
        Q[m0] = z;
        z = guess[m0]; if (z < 1e-6) z = 1e-6;
        for (int h = 0; h < m0; ++h) {
            if (guess[h] > 0.0) guess[h] /= z; else guess[h] = 0.0;
        }

        projected_bfgs(init, guess, m0, &extra, nrl_f, nrl_g, 100, tol,
            lower, upper, &r0);
        g0 = rss(init, m0, &extra);

        /* under H1 */
        F77_CALL(dsymv)(&upper_part, &k1, &one, S1, &k1,
            Q, &inc_one, &zero, guess, &inc_one);
        z = guess[m1]; if (z < 1e-6) z = 1e-6;
        for (int h = 0; h < m1; ++h) {
            if (guess[h] > 0.0) guess[h] /= z; else guess[h] = 0.0;
        }

        projected_bfgs(init, guess, m1, &extra, nrl_f, nrl_g, 100, tol,
            lower, upper, &r1);
        g1 = rss(init, m1, &extra);

        /* simulated test statistics */
        sr = r0 - r1;
        rlrt_sim[i] = sr;
        sg = (g0 - g1) / (g1 / n); if (sg < 0.0) sg = 0.0;
        gft_sim[i] = sg;
    }
    PutRNGstate();

    free(work0);
}

