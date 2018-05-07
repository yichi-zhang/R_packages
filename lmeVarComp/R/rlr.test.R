rlr.test <-
function(Y, X, Z, Sigma, m0, nsim = 5000L, seed = 130623L)
{
  # lengths
  n <- length(Y)
  p <- ncol(X)
  q <- sapply(Z, ncol)
  m1 <- length(Z)
  nmp <- n - p
  index0 <- seq_len(m0)
  
  # simultaneous diagonalization
  U0 <- qr.Q(qr(X, LAPACK = TRUE), 
    complete = TRUE)[, (p + 1L) : n, drop = FALSE]
  UtZSZtU <- matrix(0, nmp, nmp * m1)
  for (k in seq_len(m1)) {
    ZtU <- crossprod(Z[[k]], U0)
    UtZSZtU[, (nmp * (k - 1L) + 1L) : (nmp * k)] <- 
      crossprod(ZtU, Sigma[[k]] %*% ZtU)
      # t(U_0) %*% Z_k %*% Sigma_k %*% t(Z_k) %*% U_0
  }
  
  result <- diagonalize(UtZSZtU)
  U1 <- result$U
  D <- result$D
  U <- U0 %*% U1
  
  # checking redundant variance components
  if (any(colSums(D) < 1e-6)) {
    stop("Redundant variance component(s) observed.")
  }
  
  # checking assumption
  gap <- double(m1)
  for (k in seq_len(m1)) {
    gap[k] <- norm(UtZSZtU[, (nmp * (k - 1L) + 1L) : (nmp * k)] - 
      U1 %*% (t(U1) * D[, k]), "M")
  }
  
  if (any(gap > 1e-3)) {
    stop("Assumption violated.")
  }

  # initial values for tau under H0/H1 using MINQUE(0)
  # all components of tau1 and tau0 are non-negative
  result <- minque0(Y, D, U, m0, m1)
  Ytilde2 <- result$Ytilde2
  D <- result$D
  scaling <- result$scaling
  S0 <- result$S0
  S1 <- result$S1
  tau0 <- result$tau0
  tau1 <- result$tau1
  
  # null distribution
  tol <- 1e-6 / log(sum(Ytilde2)) / nmp
  if (nsim >= 1L) {
    dm <- m1 - m0
    b <- c(qchisq(0.2, 1), qchisq(0.998, dm), 
      qf(0.2, 1, 8), qf(0.998, dm, 8) * dm)
  } else {
    b <- c(1.0, 0.0, 1.0, 0.0)
  }
  
  Ytilde2 <- copy_vector(Ytilde2, "double")
  D <- copy_matrix(D, "double")
  S0 <- copy_matrix(S0, "double")
  S1 <- copy_matrix(S1, "double")
  tau0 <- copy_vector(tau0, "double")
  tau1 <- copy_vector(tau1, "double")
  rlrt_obs <- double(1L)
  rlrt_sim <- double(nsim)
  gft_obs <- double(1L)
  gft_sim <- double(nsim)
  n <- as.integer(n)
  p <- as.integer(p)
  m0 <- as.integer(m0)
  m1 <- as.integer(m1)
  nsim <- as.integer(nsim)
  tol <- as.double(tol)
  b <- as.double(b)
  
  set.seed(seed)
  if (m0 >= 1L) {
    .Call(
	  R_rlrt1, 
	  Ytilde2, D, S0, S1, tau0, tau1,
      rlrt_obs, rlrt_sim, gft_obs, gft_sim,
	  n, p, m0, m1, nsim, tol, b,
      PACKAGE = "lmeVarComp")
  } else {
    .Call(
	  R_rlrt0, 
	  Ytilde2, D, S1, tau1,
      rlrt_obs, rlrt_sim, gft_obs, gft_sim,
	  n, p, m1, nsim, tol, b,
      PACKAGE = "lmeVarComp")
  }
  
  # point estimates, test statistics and p-values
  j0 <- seq_len(m0)
  nmp <- n - p
  eps0 <- sum(Ytilde2 / (1 + D[, j0, drop = FALSE] %*% tau0)) / nmp
  eps1 <- sum(Ytilde2 / (1 + D %*% tau1)) / nmp
  H0.estimate <- c(tau0 * scaling[j0] * eps0, eps0)
  H1.estimate <- c(tau1 * scaling * eps1, eps1)
  
  if (nsim >= 1L) {
    if (((rlrt_obs > b[1L]) && (rlrt_obs < b[2L]))
      || ((gft_obs > b[3L]) && (gft_obs < b[4L]))) {
      rlrt_pv <- mean(rlrt_sim >= rlrt_obs)
      gft_pv <- mean(gft_sim >= gft_obs)
    } else {
      rlrt_pv <- (rlrt_obs <= b[1L]) * 1 + (rlrt_obs >= b[2L]) * 0
      gft_pv <- (gft_obs <= b[3L]) * 1 + (gft_obs >= b[4L]) * 0
    }
  } else {
    rlrt_pv <- NA_real_
    gft_pv <- NA_real_
  }

  list(
    "RLRT" = c(stat.obs = rlrt_obs, p.value = rlrt_pv),
    "GFT" = c(stat.obs = gft_obs, p.value = gft_pv),
    H0.estimate = H0.estimate, H1.estimate = H1.estimate)
}
