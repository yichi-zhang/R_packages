mnls <- function(x, y, rcond = 1e-10)
{
  x <- copy_matrix(x, "double")
  y <- copy_matrix(y, "double")
  m <- as.integer(nrow(x))
  n <- as.integer(ncol(x))
  nrhs <- as.integer(ncol(y))
  beta <- double(n * nrhs)
  rcond <- as.double(rcond)
  rank <- integer(1L)
  
  result <- .Call(
    R_mnls, 
	x, y, beta, m, n, nrhs, rcond, rank,
    PACKAGE = "lmeVarComp")
  
  beta <- matrix(beta, n, nrhs)
  attr(beta, "rank") <- rank
  beta
}
