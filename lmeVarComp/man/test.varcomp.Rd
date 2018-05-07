\name{test.varcomp}
\alias{test.varcomp}
\title{
Testing Zero Variance Components in Linear Mixed Models
}
\description{
\code{test.varcomp} tests whether certain variance components are zeros.
  This function provides a formula interface to the \code{rlr.test} function.
}
\usage{
test.varcomp(fixed, random, test, data = NULL, Sigma = NULL, 
  type = "RLR", nsim = 5000L, seed = 130623L, 
  keep.matrices = FALSE)
}
\arguments{
  \item{fixed}{
  a two-sided formula specifying the response and the fixed effects. 
  }
  \item{random}{
  a one-sided formula specifying the random effects 
  (not including the error term).
  }
  \item{test}{
  an integer vector of the indices of random effects to be tested.
  }
  \item{data}{
  an optional data frame, list or environment containing the variables in the model.
  }
  \item{Sigma}{
  an optional list of symmetric and positive definite matrices specifying
  the correlation structures of random effects.
  If \code{NULL}, default to identity matrices.
  }
  \item{type}{
  \code{RLR} for restricted likelihood ratio test and generalized F-test
  }
  \item{nsim}{
  number of simulations from the null distribution. 
  }
  \item{seed}{
  a seed to be set before simulating from the null distribution.
  }
  \item{keep.matrices}{
  whether the design matrices for fixed effects and random effects,
  as well as the response vector, will be returned.
  }
}
\value{
A list containing the following components:
  \item{RLRT}{A vector of the test statistic and the p-value of 
    restricted likelihood ratio test.}
  \item{GFT}{A vector of the test statistic and the p-value of 
    generalized F-test.}
  \item{Y}{If \code{keep.matrices} is \code{TRUE}, the response.}
  \item{X}{If \code{keep.matrices} is \code{TRUE}, the fixed effects design matrix.}
  \item{Z}{If \code{keep.matrices} is \code{TRUE}, a list of the random effects design matrices.}
  \item{Sigma}{If \code{keep.matrices} is \code{TRUE}, a list of the random effects correlation structures.}
}
\author{
Yichi Zhang
}
\references{
Zhang, Y., Staicu, A.-M., and Maity, A. (2016). Testing for additivity in non-parametric regression. Canadian Journal of Statistics, 44: 445-462. \doi{10.1002/cjs.11295}
}
\seealso{
\code{\link{rlr.test}}
}
\examples{
n1 <- 5L
n2 <- 6L
n0 <- 4L
A <- gl(n1, n2 * n0)
B <- rep(gl(n2, n0), n1)
set.seed(1L)
Y <- 1 + rnorm(n1, 0, 0.7)[A] + rnorm(n2, 0, 0.3)[B] + 
  rnorm(n1 * n2, 0, 0.5)[A : B] + rnorm(n1 * n2 * n0, 0, 1)
test.varcomp(Y ~ 1, ~ -1 + A + B + A:B, test = c(2L, 3L), 
  nsim = 2000L, seed = 2L)
}
