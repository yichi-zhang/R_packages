#ifndef PROJECTED_BFGS_H_INCLUDED
#define PROJECTED_BFGS_H_INCLUDED




#define MX_BFGS_ACCEPT      0.0001
#define MX_BFGS_SHRINKAGE   0.2
#define MX_BFGS_STRETCH     2
#define MX_BFGS_LONGEST     1e6
#define MX_BFGS_TINY        1e-12
#define MX_BFGS_NOT_UPDATE  0.25
#define MX_BFGS_RESTART     2.0




typedef double objective_function(const double *, int, void *);
typedef void objective_gradient(const double *, int, void *, double *);




void projected_bfgs(double *restrict x, double *restrict guess,
    const int p, void *extra,
    objective_function func, objective_gradient grad,
    const int max_iter, const double tol,
    const double *restrict lower, const double *restrict upper,
    double *value);




#endif /* PROJECTED_BFGS_H_INCLUDED */

