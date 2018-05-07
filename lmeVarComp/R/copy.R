copy_matrix <- function(x, c_type)
{
  m <- NROW(x)
  n <- NCOL(x)
  y <- matrix(0, m, n)
  y[seq_len(m), seq_len(n)] <- x
  storage.mode(y) <- c_type
  y
}


copy_vector <- function(x, c_type)
{
  n <- length(x)
  y <- double(n)
  y[seq_len(n)] <- x
  storage.mode(y) <- c_type
  y
}

