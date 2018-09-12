\name{phecap_train_phenotyping_model}
\alias{phecap_train_phenotyping_model}
\title{
Train Phenotyping Model using the Training Labels
}
\description{
Train the phenotyping model on the training dataset,
and evaluate its performance via random splits of the training dataset.
}
\usage{
phecap_train_phenotyping_model(
  data, surrogates, feature_selected,
  method = "lasso_bic", train_percent = 0.7, num_splits = 200L,
  start_seed = 78900L, verbose = 50L)
}
\arguments{
  \item{data}{
an object of class \code{PhecapData}, obtained by calling \code{PhecapData(...)}.
}
  \item{surrogates}{
a list of objects of class \code{PhecapSurrogate}, obtained by something like
\code{list(PhecapSurrogate(...), PhecapSurrogate(...))}.
}
  \item{feature_selected}{
a character vector of the features that should be included in the model,
probably returned by \code{phecap_run_feature_extraction}.
}
  \item{method}{
Either a character scalar or a list of two components.
If a character scalar is used, possible values are
\code{'plain'} (logistic regression without penalty),
\code{'lasso_cv'} (logistic regression with lasso penalty and CV tuning),
\code{'lasso_bic'} (logistic regression with lasso penalty and BIC tuning),
\code{'svm'} (support vector machine with CV tuning, package \code{e1071} needed),
and
\code{'rf'} (random forest with default parameters, package \code{randomForest} needed).
If a list is used, it should contain two named components:
\code{fit} --- a function for model fitting, and
\code{predict} ---- a function for prediction.
}
  \item{train_percent}{
The percentage (between 0 and 1) of labels that are used for model training
during random splits
}
  \item{num_splits}{
The number of random splits.
}
  \item{start_seed}{
in the i-th split, the seed is set to start_seed + i.
}
  \item{verbose}{
print progress every verbose splits if verbose is positive,
or remain quiet if verbose is zero
}
}
\value{
An object of class \code{PhecapModel}, with components
  \item{coefficients}{the fitted object}
  \item{method}{the method used for model training}
  \item{feature_selected}{the feature selected by SAFE}
  \item{train_roc}{ROC on training dataset}
  \item{train_auc}{AUC on training dataset}
  \item{split_roc}{average ROC on random splits of training dataset}
  \item{split_auc}{average AUC on random splits of training dataset}
  \item{fit_function}{the function used for fitting}
  \item{predict_function}{the function used for prediction}
}
\seealso{
See \code{\link{PheCAP-package}} for code examples.
}