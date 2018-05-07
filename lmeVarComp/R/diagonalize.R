diagonalize <- function(x, tol = 1e-10)
{
  A <- copy_matrix(x, "double")
  p <- as.integer(nrow(x))
  k <- as.integer(ncol(x) %/% nrow(x))
  tol <- as.double(tol)
  D <- double(p * k)
  fail <- integer(1L)
  
  result <- .Call(
    R_diagonalize, 
	A, p, k, tol, D, fail, 
    PACKAGE = "lmeVarComp")
  
  list(U = matrix(A, p, p), D = matrix(D, p, k))
}
