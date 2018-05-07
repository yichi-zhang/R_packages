sinv <- function(x)
{
  x <- copy_matrix(x, "double")
  n <- as.integer(ncol(x))
  .Call(R_sinv, x, n, PACKAGE = "lmeVarComp")
  matrix(x, n, n)
}
