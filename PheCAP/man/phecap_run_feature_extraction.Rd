\name{phecap_run_feature_extraction}
\alias{phecap_run_feature_extraction}
\title{
Run Surrogate-Assisted Feature Extraction (SAFE)
}
\description{
Run surrogate-assisted feature extraction (SAFE)
using unlabeled data and subsampling.
}
\usage{
phecap_run_feature_extraction(
  data, surrogates,
  subsample_size = 1000L, num_subsamples = 200L,
  start_seed = 45600L, verbose = 50L)
}
\arguments{
  \item{data}{
An object of class PhecapData, obtained by calling PhecapData(...)
}
  \item{surrogates}{
A list of objects of class PhecapSurrogate, obtained by something like
list(PhecapSurrogate(...), PhecapSurrogate(...))
}
  \item{subsample_size}{
An integer scalar giving the size of each subsample
}
  \item{num_subsamples}{
The number of subsamples drawn for each surrogate
}
  \item{start_seed}{
in the i-th subsample, the seed is set to start_seed + i.
}
  \item{verbose}{
print progress every \code{verbose} subsample if \code{verbose} is positive,
or remain quiet if \code{verbose} is zero
}
}
\details{
In this unlabeled setting, the extremes of each surrogate
are used to define cases and controls.
The variables selected are
those selected in at least half of the subsamples.
}
\value{
A character vector consisting of the names of the variables selected,
with attribute frequency, which consists of the proportion of being selected
among all subsamples for each variable.
}
\seealso{
See \code{\link{PheCAP-package}} for code examples.
}