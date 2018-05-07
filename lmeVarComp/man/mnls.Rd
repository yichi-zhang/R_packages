\name{mnls}
\alias{mnls}
\title{
Minimum Norm Least Squares
}
\description{
\code{mnls} computes the minimum norm solution to the least squares problem.
}
\usage{
mnls(x, y, rcond = 1e-10)
}
\arguments{
  \item{x}{
  design matrix of dimension \code{n} by \code{p}.
  }
  \item{y}{
  response vector of length \code{n}, or 
  response matrix of dimension \code{n} by \code{q}.
  }
  \item{rcond}{
  reciprocal condition number to determine the effective rank of \code{x}.
  }
}
\details{
The underlying C code calls the LAPACK routine DGELSY.
}
\value{
The least squares solution, as a \code{p} by \code{q} matrix.
It has an attribute called \code{rank}, which is the effective rank of \code{x}.
}
\author{
Yichi Zhang
}
\examples{
x <- matrix(rnorm(500L), 100L, 5L)
x <- cbind(x, x[, 1L] + x[, 2L], x[, 1L] - x[, 3L])
b <- -3L : 3L
y <- c(x \%*\% b)
mnls(x, y)  # different to b
}
