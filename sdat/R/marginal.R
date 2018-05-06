marginal.test <- function(x, y, num_sim = 5000L)
{
  time0 <- proc.time()[3L]
  
  if (!is.matrix(x)) x <- cbind(x)
  n <- nrow(x)
  p <- ncol(x)
  
  alpha <- seq_len(p)
  num_alpha <- length(alpha)
  extreme <- double(num_alpha + 1L)
  
  .Call(
    R_detect_effect, 
    as.double(x), as.double(y),
    as.double(alpha), extreme, 
    as.integer(n), as.integer(p), 
    as.integer(num_alpha), as.integer(num_sim),
	PACKAGE = "sdat")
  
  time0 <- as.double(proc.time()[3L] - time0)
  c("max" = extreme[1L], 
    "sum" = extreme[num_alpha], 
    "adaptive" = extreme[num_alpha + 1L],
    "time" = time0)
}

