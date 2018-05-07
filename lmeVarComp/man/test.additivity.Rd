\name{test.additivity}
\alias{test.additivity}
\title{
Testing Additivity in Nonparametric Regression
}
\description{
\code{test.additivity} tests for additive model in nonparametric regression
  using mixed model representation and variance components testing.
}
\usage{
test.additivity(x, y, type = "RLR", 
  nbasis = 10L, kernel = c("gaussian", "polynomial", "spline"),
  nsim = 5000L, seed = 130623L)
}
\arguments{
  \item{x}{
  design matrix. Each column should be scaled to have range within \code{[0,1]}.
  }
  \item{y}{
  response vector.
  }
  \item{type}{
  \code{RLR} for restricted likelihood ratio test and generalized F-test
  }
  \item{nbasis}{
  number of basis functions in additive modeling.
  }
  \item{kernel}{
  reproducing kernel for non-additive modeling.
  }
  \item{nsim}{
  number of simulations from the null distribution. 
  }
  \item{seed}{
  a seed to be set before simulating from the null distribution.
  }
}
\value{
A vector of test statistic and p-value.
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
set.seed(20L)
x <- matrix(runif(200L), 100L, 2L)
y <- 4 * x[, 1L] * x[, 2L] + rnorm(100L)
test.additivity(x, y)
}
