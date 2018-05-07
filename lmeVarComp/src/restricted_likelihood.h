#ifndef RESTRICTED_LIKELIHOOD_H_INCLUDED
#define RESTRICTED_LIKELIHOOD_H_INCLUDED




typedef struct {
    int n_minus_p;
    const double *D;    /* matrix, of size n_minus_p by m */
    const double *w2;   /* vector, of length n_minus_p */
    double *Dtau;       /* vector, of length n_minus_p */
    double *w2_Dtau;    /* vector, of length n_minus_p */
    double *w2_Dtau2;   /* vector, of length n_minus_p */
} rl_extra;




/* negative restricted likelihood */
double nrl_f(const double *tau, int m, void *extra);
void nrl_g(const double *tau, int m, void *extra, double *g);




/* residual sum of squares */
double rss(const double *tau, int m, void *extra);




#endif /* RESTRICTED_LIKELIHOOD_H_INCLUDED */

