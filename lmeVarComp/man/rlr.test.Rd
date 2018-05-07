\name{rlr.test}
\alias{rlr.test}
\title{
Restricted Likelihood Ratio Test and Generalized F-test for Zero Variance Components
}
\description{
\code{rlr.test} tests whether certain variance components are zeros
using restricted likelihood ratio test and generalized F-test.
}
\usage{
rlr.test(Y, X, Z, Sigma, m0, nsim = 5000L, seed = 130623L)
}
\arguments{
  \item{Y}{
  response vector of length \code{n}
  }
  \item{X}{
  fixed effects design matrix of dimension \code{n} by \code{p}
  }
  \item{Z}{
  a list of random effects design matrices. 
  Each matrix should have \code{n} rows.
  }
  \item{Sigma}{
  a list of random effects correlation structures.
  Each matrix should be symmetric and positive definite, and match
  the dimension of the corresponding random effects design matrix.
  }
  \item{m0}{
  an integer indicating the number of nuisance variance components.
  Should be between \code{0} and \code{length(Z) - 1}.
  The first \code{m0} variance components will be treated as nuisance.
  }
  \item{nsim}{
  number of simulations from the null distribution. 
  If zero, REML estimates are computed but tests are not performed.
  }
  \item{seed}{
  a seed to be set before simulating from the null distribution.
  }
}
\value{
A list containing the following components:
  \item{RLRT}{a vector of the test statistic and the p-value of 
    restricted likelihood ratio test.}
  \item{GFT}{a vector of the test statistic and the p-value of 
    generalized F-test.}
  \item{H0.estimate}{REML estimate of variance components 
    (including the error term) under the null hypothesis.}
  \item{H1.estimate}{REML estimate of variance components 
    (including the error term) under the alternative hypothesis.}
}
\author{
Yichi Zhang
}
\references{
Zhang, Y., Staicu, A.-M., and Maity, A. (2016). Testing for additivity in non-parametric regression. Canadian Journal of Statistics, 44: 445-462. \doi{10.1002/cjs.11295}
}
\examples{
# two-way random effects ANOVA
n1 <- 5L
n2 <- 6L
n0 <- 4L
n <- n1 * n2 * n0
X <- cbind(rep(1, n))
A <- gl(n1, n2 * n0)
Z1 <- model.matrix(~ -1 + A, contrasts.arg = contr.treatment)
B <- rep(gl(n2, n0), n1)
Z2 <- model.matrix(~ -1 + B, contrasts.arg = contr.treatment)
Z3 <- model.matrix(~ -1 + B : A, contrasts.arg = contr.treatment)
set.seed(1L)
Y <- (X \%*\% 1
  + Z1 \%*\% rnorm(ncol(Z1), 0, 0.7)
  + Z2 \%*\% rnorm(ncol(Z2), 0, 0.3)
  + Z3 \%*\% rnorm(ncol(Z3), 0, 0.5)
  + rnorm(n, 0, 1))
Z <- list(Z1, Z2, Z3)
Sigma <- lapply(Z, function(z) diag(ncol(z)))
# tests interaction effects
rlr.test(Y, X, Z, Sigma, 2L, 2000L, 2L)
# tests overall effects
rlr.test(Y, X, Z, Sigma, 1L, 2000L, 3L)
}
