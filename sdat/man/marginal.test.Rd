\name{marginal.test}
\alias{marginal.test}
\title{
Tests for signal detection via marginal approach
}
\description{
Conduct the sum-test, max-test and adaptive-test for testing beta = 0
in a linear model y = x^T beta + epsilon.
}
\usage{
marginal.test(x, y, num_sim = 5000L)
}
\arguments{
  \item{x}{
the predictors, an n by p matrix
}
  \item{y}{
the responses, a vector of length n
}
  \item{num_sim}{
the number of resampling simulations to obtain the null distribution
of the test statistic
}
}
\details{
See the reference for a detailed description of the method.
}
\value{
\code{marginal.test} returns a self-explanatory named vector.
}
\references{
Zhang, Y., Laber E. B. (2015). Comment on 
"An adaptive resampling test for detecting the presence of signifficant predictors". 
Journal of the American Statistical Association, 110(512), 1451-1454.
}
\examples{
n <- 100
p <- 10
x <- matrix(rnorm(n * p), n, p)
y <- 0.3 * x[, 1] + rnorm(n)
result <- marginal.test(x, y)
result[1 : 3]  # gives p-values of max-test, sum-test and adaptive-test
result[4]      # gives running time in seconds
}
