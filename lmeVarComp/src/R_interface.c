#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>

#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>


void diagonalize(
    double *A, const int *_p, const int *_k,
    const double *_tol, double *D, int *fail);

SEXP R_diagonalize(
    SEXP R_A, SEXP R_p, SEXP R_k,
    SEXP R_tol, SEXP R_D, SEXP R_fail)
{
    diagonalize(
        REAL(R_A), INTEGER(R_p), INTEGER(R_k),
        REAL(R_tol), REAL(R_D), INTEGER(R_fail));
    return R_NilValue;
}


void mnls(
    double *A, double *B, double *X,
    const int *_m, const int *_n, const int *_nrhs,
    const double *_rcond, int *_rank);

SEXP R_mnls(
    SEXP R_A, SEXP R_B, SEXP R_X,
    SEXP R_m, SEXP R_n, SEXP R_nrhs,
    SEXP R_rcond, SEXP R_rank)
{
    mnls(
        REAL(R_A), REAL(R_B), REAL(R_X),
        INTEGER(R_m), INTEGER(R_n), INTEGER(R_nrhs),
        REAL(R_rcond), INTEGER(R_rank));
    return R_NilValue;
}


void sinv(double *A, const int *_n);

SEXP R_sinv(SEXP R_A, SEXP R_n)
{
    sinv(REAL(R_A), INTEGER(R_n));
    return R_NilValue;
}


void rlrt0(
    double *Ytilde2, const double *D,
    const double *S1, double *tau1,
    double *rlrt_obs, double *rlrt_sim,
    double *gft_obs, double *gft_sim,
    int *_n, int *_p,
    int *_m1, int *_nsim,
    double *_tol, double *_b);

SEXP R_rlrt0(
    SEXP R_Ytilde2, SEXP R_D,
    SEXP R_S1, SEXP R_tau1,
    SEXP R_rlrt_obs, SEXP R_rlrt_sim,
    SEXP R_gft_obs, SEXP R_gft_sim,
    SEXP R_n, SEXP R_p, SEXP R_m1, SEXP R_nsim,
    SEXP R_tol, SEXP R_b)
{
    rlrt0(
        REAL(R_Ytilde2), REAL(R_D),
        REAL(R_S1), REAL(R_tau1),
        REAL(R_rlrt_obs), REAL(R_rlrt_sim),
        REAL(R_gft_obs), REAL(R_gft_sim),
        INTEGER(R_n), INTEGER(R_p),
        INTEGER(R_m1), INTEGER(R_nsim),
        REAL(R_tol), REAL(R_b));
    return R_NilValue;
}


void rlrt1(
    double *Ytilde2, const double *D,
    const double *S0, const double *S1, double *tau0, double *tau1,
    double *rlrt_obs, double *rlrt_sim,
    double *gft_obs, double *gft_sim,
    int *_n, int *_p,
    int *_m0, int *_m1, int *_nsim,
    double *_tol, double *_b);

SEXP R_rlrt1(
    SEXP R_Ytilde2, SEXP R_D,
    SEXP R_S0, SEXP R_S1, SEXP R_tau0, SEXP R_tau1,
    SEXP R_rlrt_obs, SEXP R_rlrt_sim,
    SEXP R_gft_obs, SEXP R_gft_sim,
    SEXP R_n, SEXP R_p,
    SEXP R_m0, SEXP R_m1, SEXP R_nsim,
    SEXP R_tol, SEXP R_b)
{
    rlrt1(
        REAL(R_Ytilde2), REAL(R_D),
        REAL(R_S0), REAL(R_S1), REAL(R_tau0), REAL(R_tau1),
        REAL(R_rlrt_obs), REAL(R_rlrt_sim),
        REAL(R_gft_obs), REAL(R_gft_sim),
        INTEGER(R_n), INTEGER(R_p),
        INTEGER(R_m0), INTEGER(R_m1), INTEGER(R_nsim),
        REAL(R_tol), REAL(R_b));
    return R_NilValue;
}


static const R_CallMethodDef g_call_methods[]  = {
  {"R_diagonalize", (DL_FUNC)&R_diagonalize, 6},
  {"R_mnls", (DL_FUNC)&R_mnls, 8},
  {"R_sinv", (DL_FUNC)&R_sinv, 2},
  {"R_rlrt0", (DL_FUNC)&R_rlrt0, 14},
  {"R_rlrt1", (DL_FUNC)&R_rlrt1, 17},
  {NULL, NULL, 0}
};


void attribute_visible R_init_lmeVarComp(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, g_call_methods, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
    R_forceSymbols(dll, TRUE);
}

