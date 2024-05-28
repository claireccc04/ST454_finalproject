//stan model: linear hierarchical regression
data {
  int<lower=1> D; // number of predictors + intercept
  int<lower=0> N; // number of data
  int<lower=1> J; // number of regions
  array[N] real expenditure; // outcome
  array[N] int<lower=1, upper=J> gorx; // the class number
  array[N] row_vector[D] X; // the design matrix
}
parameters {
  array[D] real mu_beta; // the mean for predictors+intercept
  array[J] vector[D] beta; // the random effects
  real<lower=0> sigma_res; // the residual sd
}
model {
//priors
  mu_beta ~ normal(0, 100);
  sigma_res ~ cauchy(0,5);
// model for the random effects
  for(j in 1:J){
    for(d in 1:D)
    beta[j,d] ~ normal(mu_beta[d], 10);
    }
// likelihood
  for (n in 1:N) {
    expenditure[n] ~ normal(X[n] * beta[gorx[n]],sigma_res);
  }
}
generated quantities { //for use with the loo package
  vector[N] expenditure_rep;
  vector[N] log_lik;
  for (n in 1:N) {
    expenditure_rep[n] = normal_rng(X[n] * beta[gorx[n]],sigma_res);
    log_lik[n] = normal_lpdf(expenditure[n] | X[n] * beta[gorx[n]],sigma_res);
}
}
