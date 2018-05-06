#include <math.h>
#include <stdlib.h>

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>

#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>

#define MALLOC_DOUBLE(size) (double *)malloc((size) * sizeof(double))


void quick_sort(
    double *restrict vdat,
    const int n);


void center_matrix(
    double *x,
    double *restrict var_x,
    const int n,
    const int p);

void get_t_statistics(
    const double *x,
    const double *y,
    const double *restrict var_x,
    const double *restrict var_y,
    double *t,
    const int n,
    const int p,
    const int q);


typedef void (*aggregate_function)(
    const double *restrict t,
    const double *restrict alpha,
    double *restrict g,
    double *temp,
    const int p,
    const int n_alpha,
    const int inc_g);

void aggregate_by_cum_sum(
    const double *restrict t,
    const double *restrict alpha,
    double *restrict g,
    double *temp,
    const int p,
    const int n_alpha,
    const int inc_g);

void aggregate_marginals(
    const double *restrict t,
    const double *restrict alpha,
    double *restrict g,
    aggregate_function aggregate,
    const int p,
    const int q,
    const int n_alpha);


double get_p_value(
    const double obs,
    const double *restrict null,
    const int n);

void detect_effect(
    double *restrict x,
    double *restrict y,
    double *restrict alpha,
    double *restrict extreme,
    aggregate_function aggregate,
    const int n,
    const int p,
    const int n_alpha,
    const int n_sim);


#define SWAP(j, k) \
    if (vdat[j] > vdat[k]) { \
        register double vtemp = vdat[j]; \
        vdat[j] = vdat[k]; \
        vdat[k] = vtemp; \
    }


void swap_sort(double *restrict vdat, const int n)
{
    switch (n) {
    case 1:
        break;

    case 2:
        SWAP(0, 1);
        break;

    case 3:
        SWAP(0, 2);
        SWAP(0, 1);
        SWAP(1, 2);
        break;

    case 4:
        SWAP(0, 2);
        SWAP(1, 3);
        SWAP(0, 1);
        SWAP(2, 3);
        SWAP(1, 2);
        break;

    case 5:
        SWAP(0, 4);
        SWAP(0, 2);
        SWAP(1, 3);
        SWAP(2, 4);
        SWAP(0, 1);
        SWAP(2, 3);
        SWAP(1, 4);
        SWAP(1, 2);
        SWAP(3, 4);
        break;

    case 6:
        SWAP(0, 4);
        SWAP(1, 5);
        SWAP(0, 2);
        SWAP(1, 3);
        SWAP(2, 4);
        SWAP(3, 5);
        SWAP(0, 1);
        SWAP(2, 3);
        SWAP(4, 5);
        SWAP(1, 4);
        SWAP(1, 2);
        SWAP(3, 4);
        break;

    case 7:
        SWAP(0, 4);
        SWAP(1, 5);
        SWAP(2, 6);
        SWAP(0, 2);
        SWAP(1, 3);
        SWAP(4, 6);
        SWAP(2, 4);
        SWAP(3, 5);
        SWAP(0, 1);
        SWAP(2, 3);
        SWAP(4, 5);
        SWAP(1, 4);
        SWAP(3, 6);
        SWAP(1, 2);
        SWAP(3, 4);
        SWAP(5, 6);
        break;

    case 8:
        SWAP(0, 4);
        SWAP(1, 5);
        SWAP(2, 6);
        SWAP(3, 7);
        SWAP(0, 2);
        SWAP(1, 3);
        SWAP(4, 6);
        SWAP(5, 7);
        SWAP(2, 4);
        SWAP(3, 5);
        SWAP(0, 1);
        SWAP(2, 3);
        SWAP(4, 5);
        SWAP(6, 7);
        SWAP(1, 4);
        SWAP(3, 6);
        SWAP(1, 2);
        SWAP(3, 4);
        SWAP(5, 6);
        break;
    }
}


void quick_sort(double *restrict vdat, const int n)
{
    /* sorts small arrays directly */
    if (n <= 8) {
        swap_sort(vdat, n);
        return;
    }

    /* chooses pivot and sets sentinels */
    const int imid = n / 2, ilast = n - 1;
    SWAP(0, imid);
    SWAP(imid, ilast);
    SWAP(0, imid);

    /* makes partition */
    const double vpivot = vdat[imid];
    vdat[imid] = vdat[n - 2];

    int i = 0, j = n - 2, k = n - 2;
    while (1) {
        while (vdat[++i] < vpivot);
        while (vdat[--j] > vpivot);
        if (i >= j) break;

        vdat[k] = vdat[i];
        vdat[i] = vdat[j];
        k = j;
    }

    if (k > i) {
        vdat[k] = vdat[i];
        vdat[i] = vpivot;
    } else {
        vdat[k] = vpivot;
        i = k;
    }

    /* sorts subarrays recursively */
    j = i;
    while ((vdat[--i] == vpivot) && (i > 0));
    while ((vdat[++j] == vpivot) && (j < n));
    quick_sort(vdat, i + 1);
    quick_sort(vdat + j, n - j);
}


void center_matrix(
    double *x,
    double *restrict var_x,
    const int n,
    const int p)
{
    /* x: n by p; var_x: p */

    for (int j = 0; j < p; ++j) {
        double *restrict z = x + n * j;

        double mean_z = 0.0, var_z = 0.0;
        for (int i = 0; i < n; ++i)
            mean_z += z[i];
        mean_z /= (double)n;

        for (int i = 0; i < n; ++i) {
            z[i] -= mean_z;
            var_z += z[i] * z[i];
        }

        var_x[j] = var_z / (double)n;
    }
}


void get_t_statistics(
    const double *x,
    const double *y,
    const double *restrict var_x,
    const double *restrict var_y,
    double *t,
    const int n,
    const int p,
    const int q)
{
    /* x: n by p, centered; y: n by q, centered */
    /* var_x: p; var_y: q; t: p by q */

    const char trans = 'T', no_trans = 'N';
    const double zero = 0.0, alpha = 1.0 / (double)n;

    /* DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC) */
    F77_CALL(dgemm)(&trans, &no_trans, &p, &q, &n, &alpha,
        x, &n, y, &n, &zero, t, &p);

    double theta, theta_var;
    for (int k = 0; k < q; ++k) {
        double *restrict t1 = t + p * k;
        for (int j = 0; j < p; ++j) {
            theta = t1[j] / var_x[j];
            theta_var = var_y[k] / var_x[j] - theta * theta;
            if (theta_var < 0.0) theta_var = 0.0;
            t1[j] = theta / sqrt(theta_var / (double)n);
        }
    }
}


void aggregate_by_cum_sum(
    const double *restrict t,
    const double *restrict alpha,
    double *restrict g,
    double *temp,
    const int p,
    const int n_alpha,
    const int inc_g)
{
    /* t: p; alpha: 0; g: p (== n_alpha); temp: p */
    /* n_alpha == p */

    for (int j = 0; j < p; ++j)
        temp[j] = -t[j] * t[j];
    quick_sort(temp, p);

    double s = 0.0;
    for (int j = 0; j < p; ++j) {
        s += temp[j];
        *g = -s;
        g += inc_g;
    }
}


void aggregate_marginals(
    const double *restrict t,
    const double *restrict alpha,
    double *restrict g,
    aggregate_function aggregate,
    const int p,
    const int q,
    const int n_alpha)
{
    /* t: p by q; alpha: n_alpha; g: q by n_alpha */

    const int ltemp = p;
    double *restrict temp = (double *)malloc(ltemp * sizeof(double));

    for (int k = 0; k < q; ++k) {
        aggregate(t + p * k, alpha, g + k, temp, p, n_alpha, q);
    }

    free(temp);
}


double get_p_value(
    const double obs,
    const double *restrict null,
    const int n)
{
    /* null: n, increasing, non-negative, larger means more extreme */

    if (obs <= null[0])
        return 1.0;
    else if (obs >= null[n - 1])
        return 1.0 / (double)n;

    int lower = 0, upper = n, k;
    while (upper > lower + 1) {
        k = 0.5 * (lower + upper);
        if (obs > null[k])
            lower = k;
        else
            upper = k;
    }

    double adj = (obs - null[lower]) / (null[upper] - null[lower]);
    return 1.0 - ((double)lower + adj) / (double)n;
}


void simulate_null(
    const double *restrict x,
    const double *restrict var_x,
    const double *restrict alpha,
    double *restrict epsilon,
    double *restrict var_epsilon,
    double *restrict t,
    double *restrict g,
    aggregate_function aggregate,
    const int n,
    const int p,
    const int q,
    const int n_alpha)
{
    /* x: n by p; var_x: p; alpha: n_alpha */
    /* epsilon: n by q; var_epsilon: q */
    /* t: p by q; g: q by n_alpha */

    for (int i = 0; i < n * q; ++i)
        epsilon[i] = norm_rand();
    center_matrix(epsilon, var_epsilon, n, q);

    get_t_statistics(x, epsilon, var_x, var_epsilon, t, n, p, q);
    aggregate_marginals(t, alpha, g, aggregate, p, q, n_alpha);
}


void detect_effect(double *restrict x,
    double *restrict y,
    double *restrict alpha,
    double *restrict extreme,
    aggregate_function aggregate,
    const int n,
    const int p,
    const int n_alpha,
    const int n_sim)
{
    /* gets observed test statistics */
    double *restrict var_x = MALLOC_DOUBLE(p);
    double var_y;
    double *restrict obs_t = MALLOC_DOUBLE(p);
    double *restrict obs_g = MALLOC_DOUBLE(n_alpha);

    center_matrix(x, var_x, n, p);
    center_matrix(y, &var_y, n, 1);
    get_t_statistics(x, y, var_x, &var_y, obs_t, n, p, 1);
    aggregate_marginals(obs_t, alpha, obs_g, aggregate, p, 1, n_alpha);

    /* simulates null samples and gets observed p-values */
    double *restrict epsilon = MALLOC_DOUBLE(n * n_sim);
    double *restrict var_epsilon = MALLOC_DOUBLE(n_sim);
    double *restrict sim_t = MALLOC_DOUBLE(p * n_sim);
    double *restrict null_g = MALLOC_DOUBLE(n_sim * n_alpha);

    simulate_null(x, var_x, alpha, epsilon, var_epsilon, sim_t, null_g,
        aggregate, n, p, n_sim, n_alpha);

    double obs_g_min = 1.0;
    for (int k = 0; k < n_alpha; ++k) {
        quick_sort(null_g + n_sim * k, n_sim);
        obs_g[k] = get_p_value(obs_g[k], null_g + n_sim * k, n_sim);
        if (obs_g[k] < obs_g_min) obs_g_min = obs_g[k];
    }

    /* gets simulated p-values */
    double *restrict sim_g = MALLOC_DOUBLE(n_sim * n_alpha);
    double *restrict sim_g_min = MALLOC_DOUBLE(n_sim);

    simulate_null(x, var_x, alpha, epsilon, var_epsilon, sim_t, sim_g,
        aggregate, n, p, n_sim, n_alpha);

    for (int i = 0; i < n_sim; ++i)
        sim_g_min[i] = 1.0;
    for (int k = 0; k < n_alpha; ++k) {
        const int offset = n_sim * k;
        for (int i = 0; i < n_sim; ++i) {
            sim_g[i + offset] = get_p_value(sim_g[i + offset],
                null_g + offset, n_sim);
            if (sim_g[i + offset] < sim_g_min[i])
                sim_g_min[i] = sim_g[i + offset];
        }
    }

    /* compares observed p-values to simulated p-values */
    for (int k = 0; k < n_alpha; ++k) {
        const int offset = n_sim * k;
        int count = 0;
        for (int i = 0; i < n_sim; ++i) {
            if (sim_g[i + offset] < obs_g[k]) ++count;
        }
        extreme[k] = (double)count / (double)n_sim;
    }

    int count = 0;
    for (int i = 0; i < n_sim; ++i) {
        if (sim_g_min[i] < obs_g_min) ++count;
    }
    extreme[n_alpha] = (double)count / (double)n_sim;

    free(var_x);
    free(obs_t);
    free(obs_g);

    free(epsilon);
    free(var_epsilon);
    free(sim_t);
    free(null_g);

    free(sim_g);
    free(sim_g_min);
}


SEXP R_detect_effect(
    SEXP R_x, SEXP R_y, SEXP R_alpha, SEXP R_extreme,
    SEXP R_n, SEXP R_p, SEXP R_num_alpha, SEXP R_num_sim)
{
    detect_effect(
	    REAL(R_x), REAL(R_y), 
		REAL(R_alpha), REAL(R_extreme), 
		aggregate_by_cum_sum,
        INTEGER(R_n)[0], INTEGER(R_p)[0], 
		INTEGER(R_num_alpha)[0], INTEGER(R_num_sim)[0]);
    return R_NilValue;
}


static const R_CallMethodDef g_call_methods[]  = {
  {"R_detect_effect", (DL_FUNC)&R_detect_effect, 8},
  {NULL, NULL, 0}
};


void attribute_visible R_init_sdat(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, g_call_methods, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
    R_forceSymbols(dll, TRUE);
}

